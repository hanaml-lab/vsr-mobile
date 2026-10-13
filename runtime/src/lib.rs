//! The playback side of vsr-mobile: frames in NV12 from the decoder, x4 frames in RGB out.
//!
//! An [`Engine`] runs the network on one frame and its hidden state. [`BicubicEngine`] is the
//! dependency-free reference used in tests and as a fallback; with the `onnx` feature,
//! `OrtEngine` runs the exported model through ONNX Runtime.

pub mod budget;
pub mod gate;
pub mod pipeline;

/// A planar RGB image, f32 in [0, 1], channel-major: `data[c * h * w + y * w + x]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Planar {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl Planar {
    pub fn new(width: usize, height: usize) -> Self {
        Planar {
            width,
            height,
            data: vec![0.0; 3 * width * height],
        }
    }

    #[inline]
    pub fn at(&self, c: usize, y: usize, x: usize) -> f32 {
        self.data[c * self.width * self.height + y * self.width + x]
    }

    /// Interleaved 8-bit RGB, for the encoder.
    pub fn to_rgb24(&self) -> Vec<u8> {
        let n = self.width * self.height;
        let mut out = Vec::with_capacity(3 * n);
        for i in 0..n {
            for c in 0..3 {
                out.push((self.data[c * n + i].clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
        out
    }
}

/// Number of channels in the hidden state (3 for the previous frame, 13 of features).
pub const STATE_CHANNELS: usize = 16;

pub trait Engine {
    /// Upscale `frame` x4, reading and updating `state` (STATE_CHANNELS x h x w). An all-zero
    /// state means "no previous frame".
    fn infer(&mut self, frame: &Planar, state: &mut [f32]) -> Result<Planar, String>;
}

/// A boxed engine is an engine, so the command line can choose one at run time.
impl<E: Engine + ?Sized> Engine for Box<E> {
    fn infer(&mut self, frame: &Planar, state: &mut [f32]) -> Result<Planar, String> {
        (**self).infer(frame, state)
    }
}

/// Bicubic x4 (Catmull-Rom). Keeps the previous frame in the state like the network does, so the
/// pipeline and the gate can be exercised without a model.
pub struct BicubicEngine;

fn cubic(t: f32) -> [f32; 4] {
    let (t2, t3) = (t * t, t * t * t);
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

pub fn bicubic_x4(src: &Planar) -> Planar {
    let (w, h) = (src.width, src.height);
    let mut out = Planar::new(w * 4, h * 4);
    let (ow, oh) = (w * 4, h * 4);
    for c in 0..3 {
        for oy in 0..oh {
            let sy = (oy as f32 + 0.5) / 4.0 - 0.5;
            let y0 = sy.floor();
            let wy = cubic(sy - y0);
            for ox in 0..ow {
                let sx = (ox as f32 + 0.5) / 4.0 - 0.5;
                let x0 = sx.floor();
                let wx = cubic(sx - x0);
                let mut acc = 0.0;
                for (j, wyj) in wy.iter().enumerate() {
                    let yy = (y0 as isize + j as isize - 1).clamp(0, h as isize - 1) as usize;
                    for (i, wxi) in wx.iter().enumerate() {
                        let xx = (x0 as isize + i as isize - 1).clamp(0, w as isize - 1) as usize;
                        acc += wyj * wxi * src.at(c, yy, xx);
                    }
                }
                out.data[c * ow * oh + oy * ow + ox] = acc.clamp(0.0, 1.0);
            }
        }
    }
    out
}

impl Engine for BicubicEngine {
    fn infer(&mut self, frame: &Planar, state: &mut [f32]) -> Result<Planar, String> {
        let n = frame.width * frame.height;
        if state.len() != STATE_CHANNELS * n {
            return Err(format!(
                "state has {} values, want {}",
                state.len(),
                STATE_CHANNELS * n
            ));
        }
        state[..3 * n].copy_from_slice(&frame.data);
        Ok(bicubic_x4(frame))
    }
}

#[cfg(feature = "onnx")]
pub mod onnx {
    //! The exported model (python -m export.onnx) through ONNX Runtime.
    use super::{Engine, Planar, STATE_CHANNELS};
    use ort::session::Session;
    use ort::value::Tensor;

    pub struct OrtEngine {
        session: Session,
    }

    impl OrtEngine {
        pub fn load(path: &str) -> Result<Self, String> {
            let session = Session::builder()
                .and_then(|b| b.commit_from_file(path))
                .map_err(|e| e.to_string())?;
            Ok(OrtEngine { session })
        }
    }

    impl Engine for OrtEngine {
        fn infer(&mut self, frame: &Planar, state: &mut [f32]) -> Result<Planar, String> {
            let (w, h) = (frame.width, frame.height);
            let f = Tensor::from_array(([1usize, 3, h, w], frame.data.clone()))
                .map_err(|e| e.to_string())?;
            let s = Tensor::from_array(([1usize, STATE_CHANNELS, h, w], state.to_vec()))
                .map_err(|e| e.to_string())?;
            let outputs = self
                .session
                .run(ort::inputs!["frame" => f, "state" => s])
                .map_err(|e| e.to_string())?;
            let (_, up) = outputs["upscaled"]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;
            let (_, st) = outputs["state_out"]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;
            state.copy_from_slice(st);
            Ok(Planar {
                width: w * 4,
                height: h * 4,
                data: up.to_vec(),
            })
        }
    }
}
