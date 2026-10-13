//! vsr-runtime --model vsr.onnx --input clip_270p.mp4 --output clip_1080p.mp4 [--frame-budget-ms 33]
//!
//! Decoding and encoding go through ffmpeg (NV12 in, RGB out, over pipes). Without the `onnx`
//! feature, or with `--engine bicubic`, frames are upscaled with bicubic interpolation. With
//! `--frame-budget-ms`, a frame that overruns the budget sends the next `--fallback-frames` frames
//! to bicubic before the network is tried again.

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use vsr_runtime::budget::BudgetedEngine;
use vsr_runtime::{pipeline, BicubicEngine, Engine};

struct Args {
    model: Option<String>,
    input: String,
    output: String,
    engine: String,
    cut_threshold: f32,
    frame_budget_ms: Option<f64>,
    fallback_frames: usize,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        model: None,
        input: String::new(),
        output: String::new(),
        engine: "onnx".into(),
        cut_threshold: 0.45,
        frame_budget_ms: None,
        fallback_frames: 30,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => a.model = Some(val()?),
            "--input" => a.input = val()?,
            "--output" => a.output = val()?,
            "--engine" => a.engine = val()?,
            "--cut-threshold" => {
                a.cut_threshold = val()?.parse().map_err(|_| "bad --cut-threshold")?
            }
            "--frame-budget-ms" => {
                a.frame_budget_ms = Some(val()?.parse().map_err(|_| "bad --frame-budget-ms")?)
            }
            "--fallback-frames" => {
                a.fallback_frames = val()?.parse().map_err(|_| "bad --fallback-frames")?
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if a.input.is_empty() || a.output.is_empty() {
        return Err(
            "usage: vsr-runtime [--model vsr.onnx] --input IN --output OUT [--engine onnx|bicubic] \
             [--frame-budget-ms MS] [--fallback-frames N]"
                .into(),
        );
    }
    Ok(a)
}

fn probe(input: &str) -> Result<(usize, usize, String), String> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "csv=p=0",
            input,
        ])
        .output()
        .map_err(|e| format!("ffprobe: {e}"))?;
    let s = String::from_utf8_lossy(&out.stdout);
    let parts: Vec<&str> = s.trim().split(',').collect();
    if parts.len() < 3 {
        return Err(format!("cannot read video size from {input}"));
    }
    Ok((
        parts[0].parse().map_err(|_| "bad width")?,
        parts[1].parse().map_err(|_| "bad height")?,
        parts[2].to_string(),
    ))
}

fn engine(a: &Args) -> Result<Box<dyn Engine + Send>, String> {
    match a.engine.as_str() {
        "bicubic" => Ok(Box::new(BicubicEngine)),
        #[cfg(feature = "onnx")]
        "onnx" => Ok(Box::new(vsr_runtime::onnx::OrtEngine::load(a.model.as_deref().ok_or("--model is required")?)?)),
        #[cfg(not(feature = "onnx"))]
        "onnx" => Err("built without ONNX Runtime: cargo build --release -p vsr-runtime --features onnx, or --engine bicubic".into()),
        other => Err(format!("unknown engine {other}")),
    }
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("vsr-runtime: {e}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<(), String> {
    let a = parse()?;
    let (w, h, rate) = probe(&a.input)?;
    let (eng, fallbacks): (Box<dyn Engine + Send>, Arc<AtomicUsize>) = match a.frame_budget_ms {
        Some(budget) => {
            let budgeted =
                BudgetedEngine::new(engine(&a)?, BicubicEngine, budget, a.fallback_frames);
            let count = budgeted.fallbacks();
            (Box::new(budgeted), count)
        }
        None => (engine(&a)?, Arc::new(AtomicUsize::new(0))),
    };
    let mut dec = Command::new("ffmpeg")
        .args([
            "-v", "error", "-i", &a.input, "-f", "rawvideo", "-pix_fmt", "nv12", "-",
        ])
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("ffmpeg: {e}"))?;
    let mut enc = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{}x{}", w * 4, h * 4),
            "-r",
            &rate,
            "-i",
            "-",
        ])
        .args([
            "-i", &a.input, "-map", "0:v", "-map", "1:a?", "-c:a", "copy", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", &a.output,
        ])
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| format!("ffmpeg: {e}"))?;
    let mut stdout = dec.stdout.take().unwrap();
    let frame_bytes = w * h * 3 / 2;
    let frames = std::iter::from_fn(move || {
        let mut buf = vec![0u8; frame_bytes];
        stdout.read_exact(&mut buf).ok().map(|_| buf)
    });
    let mut stdin = enc.stdin.take().unwrap();
    let stats = pipeline::run(
        w,
        h,
        frames,
        eng,
        |f| stdin.write_all(&f.to_rgb24()).map_err(|e| e.to_string()),
        a.cut_threshold,
    )?;
    drop(stdin);
    enc.wait().map_err(|e| e.to_string())?;
    dec.wait().map_err(|e| e.to_string())?;
    eprintln!(
        "{} frames, {} scene resets, {} fallback frames, sustained {:.1} ms/frame over the last 300",
        stats.frames,
        stats.resets,
        fallbacks.load(Ordering::Relaxed),
        stats.sustained_ms(300)
    );
    Ok(())
}
