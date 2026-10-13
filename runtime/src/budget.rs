//! A hard frame-time limit. [`BudgetedEngine`] runs the network while it keeps to the frame budget;
//! when a frame overruns, the next `cooldown` frames go to a cheaper fallback (bicubic in
//! `vsr-runtime`), and then the network is tried again. Playback degrades to a softer picture for
//! a moment instead of dropping frames.
//!
//! The fallback computes no features, so a fallback frame keeps the frame in the hidden state and
//! clears the feature channels: the network starts afresh, as after a scene cut, rather than warping
//! features that belong to an older frame.

use crate::{Engine, Planar};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Monotonic time in milliseconds.
pub trait Clock {
    fn now_ms(&mut self) -> f64;
}

/// Wall-clock time since the clock was made.
pub struct SystemClock(Instant);

impl Default for SystemClock {
    fn default() -> Self {
        SystemClock(Instant::now())
    }
}

impl Clock for SystemClock {
    fn now_ms(&mut self) -> f64 {
        self.0.elapsed().as_secs_f64() * 1000.0
    }
}

pub struct BudgetedEngine<P, F, C = SystemClock> {
    primary: P,
    fallback: F,
    clock: C,
    budget_ms: f64,
    cooldown: usize,
    left: usize,
    fallbacks: Arc<AtomicUsize>,
}

impl<P: Engine, F: Engine> BudgetedEngine<P, F> {
    /// `budget_ms` per frame for `primary`; `cooldown` frames of `fallback` after each overrun.
    pub fn new(primary: P, fallback: F, budget_ms: f64, cooldown: usize) -> Self {
        Self::with_clock(
            primary,
            fallback,
            budget_ms,
            cooldown,
            SystemClock::default(),
        )
    }
}

impl<P: Engine, F: Engine, C: Clock> BudgetedEngine<P, F, C> {
    pub fn with_clock(primary: P, fallback: F, budget_ms: f64, cooldown: usize, clock: C) -> Self {
        BudgetedEngine {
            primary,
            fallback,
            clock,
            budget_ms,
            cooldown,
            left: 0,
            fallbacks: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// The number of frames the fallback has produced; readable after the engine has moved to
    /// another thread.
    pub fn fallbacks(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.fallbacks)
    }
}

impl<P: Engine, F: Engine, C: Clock> Engine for BudgetedEngine<P, F, C> {
    fn infer(&mut self, frame: &Planar, state: &mut [f32]) -> Result<Planar, String> {
        if self.left > 0 {
            self.left -= 1;
            self.fallbacks.fetch_add(1, Ordering::Relaxed);
            let out = self.fallback.infer(frame, state)?;
            let n = frame.width * frame.height;
            state[..3 * n].copy_from_slice(&frame.data);
            state[3 * n..].iter_mut().for_each(|v| *v = 0.0);
            return Ok(out);
        }
        let t0 = self.clock.now_ms();
        let out = self.primary.infer(frame, state)?;
        if self.clock.now_ms() - t0 > self.budget_ms {
            self.left = self.cooldown;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BicubicEngine, Planar, STATE_CHANNELS};
    use std::sync::{Arc, Mutex};

    /// A clock that only moves when a fake engine says it spent time.
    #[derive(Clone, Default)]
    struct FakeClock(Arc<Mutex<f64>>);

    impl Clock for FakeClock {
        fn now_ms(&mut self) -> f64 {
            *self.0.lock().unwrap()
        }
    }

    /// Fills the output with `value`, the state's features with 1.0, and spends `costs[i]` ms on frame i.
    struct Scripted {
        value: f32,
        costs: Vec<f64>,
        calls: usize,
        clock: FakeClock,
    }

    impl Engine for Scripted {
        fn infer(&mut self, frame: &Planar, state: &mut [f32]) -> Result<Planar, String> {
            *self.clock.0.lock().unwrap() += self.costs.get(self.calls).copied().unwrap_or(0.0);
            self.calls += 1;
            let n = frame.width * frame.height;
            state[..3 * n].copy_from_slice(&frame.data);
            state[3 * n..].iter_mut().for_each(|v| *v = 1.0);
            let mut out = Planar::new(frame.width * 4, frame.height * 4);
            out.data.iter_mut().for_each(|v| *v = self.value);
            Ok(out)
        }
    }

    fn scripted(value: f32, costs: &[f64], clock: &FakeClock) -> Scripted {
        Scripted {
            value,
            costs: costs.to_vec(),
            calls: 0,
            clock: clock.clone(),
        }
    }

    fn run(engine: &mut impl Engine, frames: usize) -> Vec<f32> {
        let frame = Planar::new(4, 2);
        let mut state = vec![0f32; STATE_CHANNELS * 8];
        (0..frames)
            .map(|_| engine.infer(&frame, &mut state).unwrap().data[0])
            .collect()
    }

    #[test]
    fn a_primary_within_budget_runs_every_frame() {
        let clock = FakeClock::default();
        let mut engine = BudgetedEngine::with_clock(
            scripted(0.9, &[10.0; 5], &clock),
            scripted(0.1, &[], &clock),
            33.0,
            2,
            clock.clone(),
        );
        assert_eq!(run(&mut engine, 5), vec![0.9; 5]);
        assert_eq!(engine.fallbacks().load(Ordering::Relaxed), 0);
    }

    #[test]
    fn an_overrun_sends_the_next_cooldown_frames_to_the_fallback_then_retries() {
        let clock = FakeClock::default();
        let mut engine = BudgetedEngine::with_clock(
            scripted(0.9, &[50.0, 10.0, 10.0], &clock),
            scripted(0.1, &[], &clock),
            33.0,
            2,
            clock.clone(),
        );
        let fallbacks = engine.fallbacks();
        assert_eq!(run(&mut engine, 5), vec![0.9, 0.1, 0.1, 0.9, 0.9]);
        assert_eq!(fallbacks.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn a_fallback_frame_keeps_the_frame_in_the_state_and_clears_the_features() {
        let clock = FakeClock::default();
        let mut engine = BudgetedEngine::with_clock(
            scripted(0.9, &[50.0], &clock),
            BicubicEngine,
            33.0,
            1,
            clock.clone(),
        );
        let mut frame = Planar::new(4, 2);
        frame
            .data
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = i as f32 / 24.0);
        let mut state = vec![0f32; STATE_CHANNELS * 8];
        engine.infer(&frame, &mut state).unwrap(); // the primary overruns and leaves features of 1.0
        assert!(state[24..].iter().all(|v| *v == 1.0));
        engine.infer(&frame, &mut state).unwrap(); // the fallback
        assert_eq!(&state[..24], frame.data.as_slice());
        assert!(state[24..].iter().all(|v| *v == 0.0));
    }
}
