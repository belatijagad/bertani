//! The subset of `CPython`'s `random.Random` used by Kaggriculture.
//!
//! Merely using another MT19937 implementation is not sufficient for replay
//! compatibility: `CPython` seeds the generator with an array of little-endian
//! 32-bit words, builds floats from two outputs, and uses rejection sampling
//! for `_randbelow`.

const STATE_LEN: usize = 624;
const PERIOD_OFFSET: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// A safe, small transliteration of `CPython`'s `_randommodule.c`.
#[derive(Clone, Debug)]
pub(crate) struct PyRandom {
    state: [u32; STATE_LEN],
    index: usize,
}

impl PyRandom {
    /// Constructs the state used by `random.Random(seed)` for a nonnegative
    /// Python integer that fits in `u128`.
    pub(crate) fn new(seed: u128) -> Self {
        let mut words = [0_u32; 4];
        let mut remaining = seed;
        let mut word_count = 0;

        // CPython represents zero with one zero word for init_by_array.
        loop {
            words[word_count] = u32::try_from(remaining & u128::from(u32::MAX))
                .expect("masked seed word fits in u32");
            word_count += 1;
            remaining >>= 32;
            if remaining == 0 {
                break;
            }
        }

        let mut rng = Self {
            state: [0; STATE_LEN],
            index: STATE_LEN,
        };
        rng.init_by_array(&words[..word_count]);
        rng
    }

    /// Equivalent to `CPython`'s `Random.random()`.
    pub(crate) fn random(&mut self) -> f64 {
        let high = self.next_u32() >> 5;
        let low = self.next_u32() >> 6;
        (f64::from(high) * 67_108_864.0 + f64::from(low)) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Equivalent to `Random.getrandbits(bits)` for `bits <= 32`.
    ///
    /// As in `CPython`, requesting zero bits returns zero without advancing the
    /// generator.
    pub(crate) fn getrandbits(&mut self, bits: u32) -> u32 {
        assert!(bits <= 32, "getrandbits supports at most 32 bits");
        if bits == 0 {
            return 0;
        }
        self.next_u32() >> (32 - bits)
    }

    /// Equivalent to `CPython`'s `_randbelow_with_getrandbits(upper)`.
    pub(crate) fn randbelow(&mut self, upper: u32) -> u32 {
        assert!(upper > 0, "randbelow requires a positive upper bound");
        let bits = u32::BITS - upper.leading_zeros();
        loop {
            let candidate = self.getrandbits(bits);
            if candidate < upper {
                return candidate;
            }
        }
    }

    /// Returns the index selected by `Random.choice` for a sequence of `len`.
    pub(crate) fn choice_index(&mut self, len: usize) -> usize {
        let upper = u32::try_from(len).expect("choice length exceeds u32::MAX");
        usize::try_from(self.randbelow(upper)).expect("u32 does not fit in usize")
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= STATE_LEN {
            self.twist();
        }

        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }

    fn twist(&mut self) {
        for index in 0..STATE_LEN - PERIOD_OFFSET {
            let value = (self.state[index] & UPPER_MASK) | (self.state[index + 1] & LOWER_MASK);
            self.state[index] = self.state[index + PERIOD_OFFSET]
                ^ (value >> 1)
                ^ if value & 1 == 0 { 0 } else { MATRIX_A };
        }
        for index in STATE_LEN - PERIOD_OFFSET..STATE_LEN - 1 {
            let value = (self.state[index] & UPPER_MASK) | (self.state[index + 1] & LOWER_MASK);
            self.state[index] = self.state[index + PERIOD_OFFSET - STATE_LEN]
                ^ (value >> 1)
                ^ if value & 1 == 0 { 0 } else { MATRIX_A };
        }

        let value = (self.state[STATE_LEN - 1] & UPPER_MASK) | (self.state[0] & LOWER_MASK);
        self.state[STATE_LEN - 1] = self.state[PERIOD_OFFSET - 1]
            ^ (value >> 1)
            ^ if value & 1 == 0 { 0 } else { MATRIX_A };
        self.index = 0;
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for index in 1..STATE_LEN {
            let previous = self.state[index - 1];
            self.state[index] = 1_812_433_253_u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(u32::try_from(index).expect("MT index fits in u32"));
        }
        self.index = STATE_LEN;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        debug_assert!(!key.is_empty());
        self.init_genrand(19_650_218);

        let mut state_index = 1;
        let mut key_index = 0;
        for _ in 0..STATE_LEN.max(key.len()) {
            let previous = self.state[state_index - 1];
            self.state[state_index] = (self.state[state_index]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(key[key_index])
            .wrapping_add(u32::try_from(key_index).expect("seed index fits in u32"));

            state_index += 1;
            key_index += 1;
            if state_index >= STATE_LEN {
                self.state[0] = self.state[STATE_LEN - 1];
                state_index = 1;
            }
            if key_index >= key.len() {
                key_index = 0;
            }
        }

        for _ in 0..STATE_LEN - 1 {
            let previous = self.state[state_index - 1];
            self.state[state_index] = (self.state[state_index]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(u32::try_from(state_index).expect("MT index fits in u32"));

            state_index += 1;
            if state_index >= STATE_LEN {
                self.state[0] = self.state[STATE_LEN - 1];
                state_index = 1;
            }
        }

        // MSB is 1, guaranteeing a nonzero initial state.
        self.state[0] = UPPER_MASK;
        self.index = STATE_LEN;
    }
}

#[cfg(test)]
mod tests {
    use super::PyRandom;

    const LARGE_SEED: u128 = (u64::MAX as u128) * 1_000_003;

    #[test]
    fn full_words_match_cpython_for_integer_seed_words() {
        let cases: &[(u128, &[u32])] = &[
            (
                0,
                &[
                    3_626_764_237,
                    1_654_615_998,
                    3_255_389_356,
                    3_823_568_514,
                    1_806_341_205,
                    173_879_092,
                    1_112_038_970,
                    4_146_640_122,
                ],
            ),
            (
                1,
                &[
                    577_090_037,
                    2_444_712_010,
                    3_639_700_191,
                    3_445_702_192,
                    3_280_387_012,
                    271_041_745,
                    1_095_513_148,
                    506_456_969,
                ],
            ),
            (
                LARGE_SEED,
                &[
                    2_137_684_262,
                    1_042_549_217,
                    375_098_176,
                    4_191_977_717,
                    1_976_093_707,
                    470_682_954,
                    127_515_893,
                    641_012_018,
                ],
            ),
        ];

        for &(seed, expected) in cases {
            let mut rng = PyRandom::new(seed);
            let actual: Vec<_> = (0..expected.len()).map(|_| rng.getrandbits(32)).collect();
            assert_eq!(actual, expected, "seed {seed}");
        }
    }

    #[test]
    fn random_f64_bits_match_cpython() {
        let cases: &[(u128, &[u64])] = &[
            (
                0,
                &[
                    4_605_781_095_417_019_838,
                    4_605_002_265_878_567_962,
                    4_601_247_963_976_755_576,
                    4_598_335_849_438_463_398,
                    4_602_780_372_834_555_655,
                    4_600_966_264_840_031_036,
                    4_605_235_049_612_297_461,
                    4_599_135_616_238_489_068,
                ],
            ),
            (
                1,
                &[
                    4_594_009_002_368_267_652,
                    4_605_808_224_069_059_832,
                    4_605_054_689_724_112_659,
                    4_598_266_534_995_001_180,
                    4_602_596_585_012_360_058,
                    4_601_768_931_085_461_634,
                    4_604_044_247_283_579_354,
                    4_605_279_407_925_821_028,
                ],
            ),
            (
                LARGE_SEED,
                &[
                    4_602_637_717_576_163_406,
                    4_590_957_524_307_120_152,
                    4_601_959_957_626_122_346,
                    4_584_214_667_040_819_328,
                    4_598_845_820_362_478_608,
                    4_600_249_133_998_619_018,
                    4_604_644_659_288_201_418,
                    4_599_313_998_048_923_038,
                ],
            ),
        ];

        for &(seed, expected) in cases {
            let mut rng = PyRandom::new(seed);
            let actual: Vec<_> = (0..expected.len())
                .map(|_| rng.random().to_bits())
                .collect();
            assert_eq!(actual, expected, "seed {seed}");
        }
    }

    #[test]
    fn varying_getrandbits_widths_match_cpython() {
        let widths = [0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 24, 31, 32];
        let expected = [
            0,
            1,
            1,
            6,
            14,
            13,
            5,
            66,
            494,
            33_506,
            63_691,
            6_793_667,
            1_971_893_209,
            3_366_389_305,
        ];
        let mut rng = PyRandom::new(0);

        let actual: Vec<_> = widths
            .into_iter()
            .map(|width| rng.getrandbits(width))
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn zero_getrandbits_does_not_advance_state() {
        let mut rng = PyRandom::new(1);
        assert_eq!(rng.getrandbits(0), 0);
        assert_eq!(rng.getrandbits(32), 577_090_037);
    }

    #[test]
    fn randbelow_matches_cpython_rejection_sampling() {
        // Eight is intentionally a power of two. CPython requests four bits,
        // rejecting values 8..=15, instead of masking down to three bits.
        let expected = [
            6, 6, 0, 4, 7, 6, 4, 7, 5, 3, 2, 4, 2, 1, 4, 2, 4, 1, 1, 5, 7, 1, 5, 6, 5, 3, 7, 7, 4,
            0, 0, 1,
        ];
        let mut rng = PyRandom::new(0);
        let actual: Vec<_> = (0..expected.len()).map(|_| rng.randbelow(8)).collect();
        assert_eq!(actual, expected);

        // This also checks the exact number of words consumed by rejections.
        assert_eq!(rng.getrandbits(32), 3_091_108_076);
    }

    #[test]
    fn mixed_randbelow_bounds_match_cpython() {
        let bounds = [1, 2, 3, 4, 5, 7, 8, 9, 10, 17, 255, 256, 257, 1_000];
        let expected = [0, 0, 0, 3, 0, 0, 2, 4, 6, 11, 201, 226, 160, 24];
        let mut rng = PyRandom::new(LARGE_SEED);
        let actual: Vec<_> = bounds
            .into_iter()
            .map(|bound| rng.randbelow(bound))
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn choice_indices_match_cpython() {
        let expected = [
            7, 3, 1, 7, 1, 0, 2, 4, 6, 5, 7, 5, 0, 0, 1, 7, 7, 7, 7, 3, 3, 3, 6, 4, 5, 3, 6, 4, 1,
            4, 3, 1,
        ];
        let mut rng = PyRandom::new(LARGE_SEED);
        let actual: Vec<_> = (0..expected.len()).map(|_| rng.choice_index(8)).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    #[should_panic(expected = "at most 32 bits")]
    fn getrandbits_rejects_unsupported_width() {
        PyRandom::new(0).getrandbits(33);
    }

    #[test]
    #[should_panic(expected = "positive upper bound")]
    fn randbelow_rejects_zero() {
        PyRandom::new(0).randbelow(0);
    }
}
