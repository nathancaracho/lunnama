#[derive(Debug, Default)]
pub struct LunnamaPromptBuilder {
    content: String,
}
impl LunnamaPromptBuilder {
    pub fn with_task(mut self, description: &str) -> Self {
        self.content.push_str(&format!("{}\n\n", description));
        self
    }
    pub fn with_role(mut self, role: &str) -> Self {
        self.content.push_str(&format!("You are {}\n\n", role));
        self
    }
    pub fn build(self) -> String {
        self.content
    }
    pub fn new() -> Self {
        Self {
            content: String::new(),
        }
    }
}

#[cfg(test)]
mod prompt {
    use rstest::rstest;

    use super::*;
    #[rstest]
    #[case("Doctor", "Diagnose the patient's condition")]
    #[case("Detective", "Investigate the mysterious disappearance")]
    #[case("Chef", "Prepare a traditional Italian pasta dish")]
    fn should_build_correctly(#[case] role: &str, #[case] task: &str) {
        let prompt = LunnamaPromptBuilder::new()
            .with_role(role)
            .with_task(task)
            .build();

        assert!(
            prompt.contains(role),
            "Prompt should contain the role '{}', but got:\n{}",
            role,
            prompt
        );

        assert!(
            prompt.contains(task),
            "Prompt should contain the task '{}', but got:\n{}",
            task,
            prompt
        );

        assert_eq!(
            prompt,
            format!("You are {}\n\n{}\n\n", role, task),
            "The prompt not format correctly"
        );
    }
}
