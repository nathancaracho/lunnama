use inference::{
    gen_config_builder::LunnamaGenConfigBuilder, model_builder::LunnamaModel,
    prompt_builder::LunnamaPromptBuilder,
};
fn main() -> anyhow::Result<()> {
    let model = LunnamaModel::from_file(
        "/Users/nathancaracho/Documents/projects/AI/models/tinyllama-1.1b-chat-v1.0.Q4_K_S.gguf",
    )?;
    let prompt = LunnamaPromptBuilder::new()
        .with_role("Your are RUST teacher.")
        .with_task("What is Rust lang")
        .build();
    let result = model
        .with_config(
            LunnamaGenConfigBuilder::default()
                .temperature(1.5)
                .max_tokens(1000),
        )
        .generate(&prompt)?;
    println!("the result: {:?}", result);
    Ok(())
}
