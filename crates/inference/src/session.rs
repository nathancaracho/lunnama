use super::{
    gen_config_builder::LunnamaGenConfigBuilder,
    model_builder::{LunnamaModel, LunnamaModelResult},
};
#[derive(Debug)]
pub struct LunnamaSession<'a> {
    pub(crate) model: &'a LunnamaModel,
    pub(crate) config: LunnamaGenConfigBuilder,
}

impl<'a> LunnamaSession<'a> {
    pub fn new(model: &'a LunnamaModel) -> Self {
        Self {
            model,
            config: LunnamaGenConfigBuilder::default(),
        }
    }
    pub fn with_config(mut self, config: LunnamaGenConfigBuilder) -> Self {
        self.config = config;
        self
    }
    pub fn generate(&self, prompt: &str) -> anyhow::Result<LunnamaModelResult> {
        self.model.generate(prompt, &self.config)
    }
}
