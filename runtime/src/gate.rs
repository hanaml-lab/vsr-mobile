//! A cheap scene-cut detector run before inference. The network's own confidence gate handles
//! fast motion and occlusion; a hard cut is caught here from the luma histograms alone, and the
//! hidden state is cleared so the first frame after the cut is not smeared with the last one.

pub struct SceneCutGate {
    threshold: f32,
    prev: Option<[f32; 32]>,
}

impl SceneCutGate {
    /// `threshold` is the total-variation distance between luma histograms (0..=1) above which
    /// two frames are treated as different shots.
    pub fn new(threshold: f32) -> Self {
        SceneCutGate {
            threshold,
            prev: None,
        }
    }

    fn histogram(luma: &[u8]) -> [f32; 32] {
        let mut h = [0f32; 32];
        for &v in luma {
            h[(v >> 3) as usize] += 1.0;
        }
        let n = luma.len().max(1) as f32;
        h.iter_mut().for_each(|x| *x /= n);
        h
    }

    /// Feed the luma plane of the next frame; returns true when the state must be reset.
    pub fn is_cut(&mut self, luma: &[u8]) -> bool {
        let h = Self::histogram(luma);
        let cut = match &self.prev {
            None => false,
            Some(p) => {
                p.iter().zip(&h).map(|(a, b)| (a - b).abs()).sum::<f32>() / 2.0 > self.threshold
            }
        };
        self.prev = Some(h);
        cut
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_a_cut_but_not_a_pan() {
        let mut gate = SceneCutGate::new(0.4);
        let dark: Vec<u8> = (0..1000).map(|i| (i % 64) as u8).collect();
        let panned: Vec<u8> = (0..1000).map(|i| ((i + 7) % 64) as u8).collect();
        let bright: Vec<u8> = (0..1000).map(|i| 180 + (i % 64) as u8).collect();
        assert!(!gate.is_cut(&dark));
        assert!(!gate.is_cut(&panned));
        assert!(gate.is_cut(&bright));
        assert!(!gate.is_cut(&bright));
    }
}
