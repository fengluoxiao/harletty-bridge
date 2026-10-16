// SPDX-License-Identifier: MIT
// Input-conditioning policy and band-zero coefficients adapted from
// TheM14/JustOneCacophony, revision 428772eb87275e3227efb3fc37911e7263234fda.
// See ../../JOC_INPUT_CONDITIONING_LICENSE for the retained notice.
// This is the auditioned five-input path, not a claim of Dolby conformance.

use super::qmf::{QMF_SUBBANDS, QmfSubbands};
pub(super) const INPUT_DELAY_SAMPLES: usize = 10 * QMF_SUBBANDS;
const HISTORY: usize = 21;
const DELAY: usize = 10;
const BAND0_REAL: [f64; HISTORY] = [
    0.0013996040215715766,
    0.003839150769636035,
    0.007512642536312342,
    0.012419373728334904,
    0.018367428332567215,
    0.0249701626598835,
    0.03167900815606117,
    0.03785000368952751,
    0.04283412545919418,
    0.04607561603188515,
    0.047200120985507965,
    0.04607561603188515,
    0.04283412545919418,
    0.03785000368952751,
    0.03167900815606117,
    0.0249701626598835,
    0.018367428332567215,
    0.012419373728334904,
    0.007512642536312342,
    0.003839150769636035,
    0.0013996040215715766,
];
const BAND0_IMAG: [f64; HISTORY] = [
    -0.0006242550443857908,
    -0.0019234686624258757,
    -0.0042654648423194885,
    -0.008168308064341545,
    -0.014327201060950756,
    -0.023759860545396805,
    -0.03757232800126076,
    -0.05577569454908371,
    -0.07568276673555374,
    -0.09172472357749939,
    -0.5979374051094055,
    -0.09172472357749939,
    -0.07568276673555374,
    -0.05577569454908371,
    -0.03757232800126076,
    -0.023759860545396805,
    -0.014327201060950756,
    -0.008168308064341545,
    -0.0042654648423194885,
    -0.0019234686624258757,
    -0.0006242550443857908,
];

#[derive(Debug)]
pub(super) struct JocInputConditioner {
    history: Vec<Vec<QmfSubbands>>,
    cursor: usize,
    filled: usize,
}
impl Default for JocInputConditioner {
    fn default() -> Self {
        Self {
            history: (0..5).map(|_| vec![QmfSubbands::zero(); HISTORY]).collect(),
            cursor: 0,
            filled: 0,
        }
    }
}
impl JocInputConditioner {
    pub fn reset(&mut self) {
        for channel in &mut self.history {
            for slot in channel {
                *slot = QmfSubbands::zero();
            }
        }
        self.cursor = 0;
        self.filled = 0;
    }
    pub fn process(&mut self, bands: &mut [QmfSubbands]) {
        debug_assert_eq!(bands.len(), 5);
        for (channel, value) in bands.iter().enumerate() {
            self.history[channel][self.cursor].clone_from(value);
        }
        for (channel, value) in bands.iter_mut().enumerate() {
            *value = if self.filled >= DELAY {
                self.history[channel][(self.cursor + HISTORY - DELAY) % HISTORY].clone()
            } else {
                QmfSubbands::zero()
            };
            if channel < 3 {
                continue;
            }
            for band in 1..QMF_SUBBANDS {
                let real = value.real[band];
                value.real[band] = value.imaginary[band];
                value.imaginary[band] = -real;
            }
            let (mut real, mut imag) = (0.0_f64, 0.0_f64);
            for k in 0..HISTORY {
                let age = HISTORY - 1 - k;
                if self.filled < age {
                    continue;
                }
                let x = &self.history[channel][(self.cursor + HISTORY - age) % HISTORY];
                real += f64::from(x.real[0]) * BAND0_REAL[k]
                    - f64::from(x.imaginary[0]) * BAND0_IMAG[k];
                imag += f64::from(x.real[0]) * BAND0_IMAG[k]
                    + f64::from(x.imaginary[0]) * BAND0_REAL[k];
            }
            value.real[0] = (2.0 * real) as f32;
            value.imaginary[0] = (2.0 * imag) as f32;
        }
        self.cursor = (self.cursor + 1) % HISTORY;
        self.filled = (self.filled + 1).min(HISTORY);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn delay_and_surround_rotation_have_the_expected_polarity() {
        let mut state = JocInputConditioner::default();
        for t in 0..35 {
            let mut bands = vec![QmfSubbands::zero(); 5];
            if t == 0 {
                for band in &mut bands {
                    band.real[17] = 3.0;
                    band.imaginary[17] = 4.0;
                }
            }
            state.process(&mut bands);
            for (channel, band) in bands.iter().enumerate() {
                let expected = if t != 10 {
                    (0.0, 0.0)
                } else if channel < 3 {
                    (3.0, 4.0)
                } else {
                    (4.0, -3.0)
                };
                assert_eq!((band.real[17], band.imaginary[17]), expected);
            }
        }
    }
    #[test]
    fn reset_clears_both_delay_and_band_zero_filter_history() {
        let mut dirty = JocInputConditioner::default();
        for _ in 0..30 {
            let mut bands = vec![
                QmfSubbands {
                    real: [1.0; 64],
                    imaginary: [2.0; 64]
                };
                5
            ];
            dirty.process(&mut bands);
        }
        dirty.reset();
        let mut fresh = JocInputConditioner::default();
        for _ in 0..40 {
            let mut a = vec![QmfSubbands::zero(); 5];
            let mut b = a.clone();
            dirty.process(&mut a);
            fresh.process(&mut b);
            assert_eq!(a, b);
        }
    }
}
