use std::num::NonZeroU32;

use anyhow::Ok;
use llama_cpp_2::{
    context::{params::LlamaContextParams, LlamaContext},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    token::LlamaToken,
};
use thiserror::Error;
use tracing::info;

use super::{gen_config_builder::LunnamaGenConfigBuilder, session::LunnamaSession};

#[derive(Debug, Error)]
pub enum LunnamaModelError {
    #[error("Prompt not provide")]
    PromptNotProvide,
    #[error("backend not initialized")]
    BackendNotInit,
    #[error("Inner not initialized")]
    InnerNotInit,
}

#[derive(Debug)]
pub struct LunnamaModel {
    inner: LlamaModel,
    backend: LlamaBackend,
}

#[derive(Debug, Clone)]
pub struct LunnamaModelResult {
    pub content: String,
    pub input_tokens: u32,
    pub out_tokens: u32,
}

impl LunnamaModel {
    pub fn from_file(path: &str) -> anyhow::Result<LunnamaSession> {
        let backend = LlamaBackend::init()?;
        let params = LlamaModelParams::default();
        let inner = LlamaModel::load_from_file(&backend, path, &params)?;
        let model = Box::leak(Box::new(LunnamaModel { inner, backend }));
        Ok(LunnamaSession::new(model))
    }
    fn get_new_ctx(&self, total_required_tokens: u32) -> anyhow::Result<LlamaContext> {
        // To create the context we need discovery the total used tokens prompt tks + max output tokens
        let ctx_size = NonZeroU32::new(total_required_tokens);
        let ctx = self.inner.new_context(
            &self.backend,
            LlamaContextParams::default().with_n_ctx(ctx_size),
        )?;
        Ok(ctx)
    }
    ///Create and fill the batch with prompt tokens and add flag true to last token
    fn initialize_batch(
        &self,
        tokenized_prompt: Vec<LlamaToken>,
        total_required_tokens: u32,
    ) -> anyhow::Result<LlamaBatch> {
        let mut batch = LlamaBatch::new(total_required_tokens as usize, 1);
        let last_index = (tokenized_prompt.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokenized_prompt.into_iter()) {
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)?;
        }
        Ok(batch)
    }
    pub(crate) fn generate(
        &self,
        prompt: &str,
        config_builder: &LunnamaGenConfigBuilder,
    ) -> anyhow::Result<LunnamaModelResult> {
        println!("prompt:\n {}", prompt);
        let tokenized_prompt = self.inner.str_to_token(prompt, AddBos::Always)?;
        let prompt_len = tokenized_prompt.len() as u32;
        let max_tokens = config_builder.max_tokens.max(1000);

        let total_required_tokens = prompt_len + max_tokens;

        let mut ctx = self.get_new_ctx(total_required_tokens)?;
        let mut batch = self.initialize_batch(tokenized_prompt, total_required_tokens)?;
        // adding the tokenized prompt to LLM context, generating the first
        // logits
        ctx.decode(&mut batch)?;

        let mut output = String::new();

        let mut cursor = batch.n_tokens();
        let mut sampler = config_builder.get_sampler();

        while cursor < total_required_tokens as i32 {
            let is_higher_than_max_tokens = cursor + 1 > ctx.n_ctx() as i32;
            if is_higher_than_max_tokens {
                println!("Context limit reached ({} tokens)", ctx.n_ctx());
                break;
            }
            //choose the next token based on logits
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);

            sampler.accept(token);
            if self.inner.is_eog_token(token) {
                info!("EOG token reached");
                break;
            }

            //parse token to String and push to output
            let decoded = self.inner.token_to_str(token, Special::Tokenize)?;
            output.push_str(&decoded);

            batch.clear();
            //add new token to batch
            batch.add(token, cursor, &[0], true)?;
            //get new logits
            ctx.decode(&mut batch)?;

            cursor += 1;
        }

        Ok(LunnamaModelResult {
            content: output,
            input_tokens: prompt_len,
            out_tokens: cursor as u32 - prompt_len,
        })
    }
}
