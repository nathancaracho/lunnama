use llama_cpp_2::{send_logs_to_tracing, LogOptions};
use tracing::info;
use inference::{
    gen_config_builder::LunnamaGenConfigBuilder, model_builder::LunnamaModel,
};
fn main() -> anyhow::Result<()> {
    send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
    let model = LunnamaModel::from_file(
        "/Users/nathancaracho/Documents/projects/AI/models/llama-3.2-1b-instruct-q8_0.gguf",
    )?
    .with_config(
        LunnamaGenConfigBuilder::default()
            .temperature(0.7)
            .max_tokens(1000),
    );

    let result = model.generate(
        r#"### Task:
        Explain the question in your own words. Do not answer yet.

        ### Question:
        How many days are there between February 1st and March 1st in a common (non‑leap) year?

        ### Explanation:"#,
    )?;

    let result = model.generate(
        format!(
            r#"### Instruction:
            Based on the explanation below, list the logical steps needed to solve the problem.

            ### Explanation:
            {}

            ### Steps:
            "#,
            result.content
        )
        .as_str(),
    )?;
    let result = model.generate(
        format!(
            r#"
            ## Task
                Solve the question by following the steps listed.
            ## Steps:
            {}

            "#,
            result.content
        )
        .as_str(),
    )?;
    info!("The final result 3:\n{}", result.content);
    Ok(())
}
