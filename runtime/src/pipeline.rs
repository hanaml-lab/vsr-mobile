//! Decode, a two-frame queue, inference, encode.
//!
//! Decoding, inference and encoding run on three threads joined by bounded channels of depth
//! two: the decoder can be one frame ahead while a frame is in the engine, and memory never
//! grows with the length of the video. NV12 frames are read in place from the decoder's buffer
//! (no copy before colour conversion), which matters on a Jetson, where the decoder writes them
//! into memory the GPU can read.

use crate::gate::SceneCutGate;
use crate::{Engine, Planar, STATE_CHANNELS};
use std::sync::mpsc::sync_channel;
use std::thread;
use std::time::Instant;

/// A borrowed NV12 frame: a full-resolution Y plane followed by an interleaved half-resolution UV plane.
pub struct Nv12<'a> {
    pub width: usize,
    pub height: usize,
    pub data: &'a [u8],
}

impl<'a> Nv12<'a> {
    pub fn new(width: usize, height: usize, data: &'a [u8]) -> Result<Self, String> {
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err("NV12 needs even dimensions".into());
        }
        if data.len() != width * height * 3 / 2 {
            return Err(format!(
                "NV12 {width}x{height} needs {} bytes, got {}",
                width * height * 3 / 2,
                data.len()
            ));
        }
        Ok(Nv12 {
            width,
            height,
            data,
        })
    }

    pub fn luma(&self) -> &'a [u8] {
        &self.data[..self.width * self.height]
    }

    /// BT.709 limited range to RGB in [0, 1].
    pub fn to_planar(&self) -> Planar {
        let (w, h) = (self.width, self.height);
        let (y_plane, uv) = self.data.split_at(w * h);
        let mut out = Planar::new(w, h);
        let n = w * h;
        for y in 0..h {
            for x in 0..w {
                let luma = (y_plane[y * w + x] as f32 - 16.0) / 219.0;
                let i = (y / 2) * w + (x / 2) * 2;
                let cb = (uv[i] as f32 - 128.0) / 224.0;
                let cr = (uv[i + 1] as f32 - 128.0) / 224.0;
                let p = y * w + x;
                out.data[p] = (luma + 1.5748 * cr).clamp(0.0, 1.0);
                out.data[n + p] = (luma - 0.1873 * cb - 0.4681 * cr).clamp(0.0, 1.0);
                out.data[2 * n + p] = (luma + 1.8556 * cb).clamp(0.0, 1.0);
            }
        }
        out
    }
}

#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub frames: usize,
    pub resets: usize,
    pub infer_ms: Vec<f64>,
}

impl Stats {
    /// Mean frame time over the last `tail` frames: the sustained figure, not the first second.
    pub fn sustained_ms(&self, tail: usize) -> f64 {
        let t = &self.infer_ms[self.infer_ms.len().saturating_sub(tail)..];
        if t.is_empty() {
            0.0
        } else {
            t.iter().sum::<f64>() / t.len() as f64
        }
    }
}

/// Run `frames` (owned NV12 buffers) through the engine and hand each upscaled frame to `sink`.
pub fn run<E, I, S>(
    width: usize,
    height: usize,
    frames: I,
    mut engine: E,
    mut sink: S,
    cut_threshold: f32,
) -> Result<Stats, String>
where
    E: Engine + Send + 'static,
    I: Iterator<Item = Vec<u8>> + Send + 'static,
    S: FnMut(Planar) -> Result<(), String>,
{
    let (to_infer, from_decode) = sync_channel::<Vec<u8>>(2);
    let (to_encode, from_infer) = sync_channel::<Result<(Planar, f64, bool), String>>(2);
    let decoder = thread::spawn(move || {
        for f in frames {
            if to_infer.send(f).is_err() {
                break;
            }
        }
    });
    let infer = thread::spawn(move || {
        let mut state = vec![0f32; STATE_CHANNELS * width * height];
        let mut gate = SceneCutGate::new(cut_threshold);
        for buf in from_decode {
            let result = (|| {
                let nv12 = Nv12::new(width, height, &buf)?;
                let cut = gate.is_cut(nv12.luma());
                if cut {
                    state.iter_mut().for_each(|v| *v = 0.0);
                }
                let t0 = Instant::now();
                let out = engine.infer(&nv12.to_planar(), &mut state)?;
                Ok((out, t0.elapsed().as_secs_f64() * 1000.0, cut))
            })();
            let failed = result.is_err();
            if to_encode.send(result).is_err() || failed {
                break;
            }
        }
    });
    let mut stats = Stats::default();
    for item in from_infer {
        let (frame, ms, cut) = item?;
        sink(frame)?;
        stats.frames += 1;
        stats.resets += cut as usize;
        stats.infer_ms.push(ms);
    }
    infer.join().map_err(|_| "inference thread panicked")?;
    decoder.join().map_err(|_| "decoder thread panicked")?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BicubicEngine;

    fn grey_nv12(w: usize, h: usize, y: u8) -> Vec<u8> {
        let mut v = vec![y; w * h];
        v.extend(std::iter::repeat_n(128u8, w * h / 2));
        v
    }

    #[test]
    fn nv12_grey_converts_to_grey() {
        let buf = grey_nv12(4, 2, 126);
        let p = Nv12::new(4, 2, &buf).unwrap().to_planar();
        let v = (126.0 - 16.0) / 219.0;
        assert!(p.data.iter().all(|x| (x - v).abs() < 1e-5));
        assert!(Nv12::new(3, 2, &buf).is_err());
    }

    #[test]
    fn pipeline_upscales_every_frame_in_order_and_resets_at_cuts() {
        let (w, h) = (16, 8);
        let frames: Vec<Vec<u8>> = (0..6)
            .map(|i| grey_nv12(w, h, if i < 3 { 40 } else { 220 }))
            .collect();
        let mut seen = Vec::new();
        let stats = run(
            w,
            h,
            frames.into_iter(),
            BicubicEngine,
            |f| {
                assert_eq!((f.width, f.height), (64, 32));
                seen.push(f.data[0]);
                Ok(())
            },
            0.5,
        )
        .unwrap();
        assert_eq!(stats.frames, 6);
        assert_eq!(stats.resets, 1);
        assert!(seen[2] < seen[3]);
    }

    #[test]
    fn bad_frame_size_is_an_error() {
        let frames = vec![vec![0u8; 10]];
        assert!(run(16, 8, frames.into_iter(), BicubicEngine, |_| Ok(()), 0.5).is_err());
    }

    #[test]
    fn bicubic_preserves_flat_images() {
        let mut p = Planar::new(5, 3);
        p.data.iter_mut().for_each(|v| *v = 0.25);
        assert!(crate::bicubic_x4(&p)
            .data
            .iter()
            .all(|v| (v - 0.25).abs() < 1e-6));
    }
}
