//! #LunnamaGenConfigBuilder
//! 
use llama_cpp_2::sampling::LlamaSampler;

#[derive(Debug, Clone, Default)]
pub struct LunnamaGenConfigBuilder {
    temperature: f32,
    top_k: i32,
    top_p: f32,
    is_greedy: bool,
    pub(crate) max_tokens: u32,
}

impl LunnamaGenConfigBuilder {
    pub fn temperature(mut self, temp: f32) -> Self {
        assert!((0.0..=2.0).contains(&temp), "temperature out of range");
        self.temperature = temp;
        self
    }
    pub fn top_k(mut self, top_k: i32) -> Self {
        assert!((top_k > 0), "top k should be higher than 0");
        self.top_k = top_k;
        self
    }
    pub fn top_p(mut self, top_p: f32) -> Self {
        assert!((0.0..=1.0).contains(&top_p), "top p out of range");
        self.top_p = top_p;
        self
    }
    pub fn greedy(mut self) -> Self {
        self.is_greedy = true;
        self
    }
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        assert!(max_tokens > 0, "Max tokens must be greater than 0");
        self.max_tokens = max_tokens;
        self
    }

    ///
    /// # Example
    /// Demonstrates how to create a sampler using `LunnamaGenConfigBuilder`
    /// and apply it to a token data array.
    /// ## ⚠️ Important
    /// This fluent builder is used inside `LunnamaModel` to choose **best** the token
    /// ```rust
    /// use twilight_llama::model::gen_config_builder::LunnamaGenConfigBuilder;
    ///
    /// let config = LunnamaGenConfigBuilder::default().greedy();
    /// ```
    pub(crate) fn get_sampler(&self) -> LlamaSampler {
        if self.is_greedy {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain(
                vec![
                    LlamaSampler::temp(self.temperature),
                    LlamaSampler::top_k(self.top_k),
                    LlamaSampler::top_p(self.top_p, 1),
                    LlamaSampler::dist(0),
                ],
                false,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_config() -> LunnamaGenConfigBuilder {
        LunnamaGenConfigBuilder::default()
    }

    mod temperature {
        use super::*;

        #[test]
        #[should_panic(expected = "temperature out of range")]
        fn should_panic_when_temperature_is_negative() {
            get_config().temperature(-0.1);
        }

        #[test]
        #[should_panic(expected = "temperature out of range")]
        fn should_panic_when_temperature_is_above_two() {
            get_config().temperature(2.1);
        }

        #[test]
        fn should_accept_valid_temperature() {
            let config = get_config().temperature(1.5);
            assert_eq!(config.temperature, 1.5);
        }
    }

    mod top_k {
        use super::*;

        #[test]
        #[should_panic(expected = "top k should be higher than 0")]
        fn should_panic_when_top_k_is_zero() {
            get_config().top_k(0);
        }

        #[test]
        fn should_accept_valid_top_k() {
            let config = get_config().top_k(42);
            assert_eq!(config.top_k, 42);
        }
    }

    mod top_p {
        use super::*;

        #[test]
        #[should_panic(expected = "top p out of range")]
        fn should_panic_when_top_p_is_below_zero() {
            get_config().top_p(-0.1);
        }

        #[test]
        #[should_panic(expected = "top p out of range")]
        fn should_panic_when_top_p_is_above_one() {
            get_config().top_p(1.1);
        }

        #[test]
        fn should_accept_valid_top_p() {
            let config = get_config().top_p(0.9);
            assert_eq!(config.top_p, 0.9);
        }
    }

    mod greedy {
        use super::*;

        #[test]
        fn should_set_is_greedy_to_true() {
            let config = get_config().greedy();
            assert!(config.is_greedy);
        }
    }

    mod max_tokens {
        use super::*;

        #[test]
        #[should_panic(expected = "Max tokens must be greater than 0")]
        fn should_panic_when_max_tokens_below_one() {
            get_config().max_tokens(0);
        }
        #[test]
        fn should_accept_and_modify_max_tokens() {
            let config = get_config().max_tokens(1000);
            assert_eq!(config.max_tokens, 1000);

            let config = config.max_tokens(999);
            assert_eq!(config.max_tokens, 999);
        }
    }

    mod get_sampler {
        use llama_cpp_2::token::{
            data::LlamaTokenData, data_array::LlamaTokenDataArray, LlamaToken,
        };

        use super::*;

        #[test]
        fn should_choose_the_higher_token_in_logits() {
            let mut data_array = LlamaTokenDataArray::new(
                vec![
                    LlamaTokenData::new(LlamaToken(0), 0., 0.),
                    LlamaTokenData::new(LlamaToken(1), 1., 0.),
                ],
                false,
            );
            let sampler = get_config().greedy().get_sampler();
            data_array.apply_sampler(&sampler);
            assert_eq!(data_array.selected_token(), Some(LlamaToken(1)));
        }
    }
}
