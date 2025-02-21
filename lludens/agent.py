import llm


class TotreLLM:
    def __init__(self, model_id: str, system: str = None, options: dict = None):
        # Retrieve the desired model
        self.model = llm.get_model(model_id)
        # Start a conversation to automatically maintain context
        self.conversation = self.model.conversation()
        # Default options include turning off streaming for simplicity
        self.options = options or {
            "temperature": 0.1,
        }
        # Seed the conversation with the system prompt if provided
        if system:
            self.system = system

    def interact(
        self,
        prompt: str,
        attachments: list = None,
    ) -> str:
        response = self.conversation.prompt(
            prompt, attachments=attachments, system=self.system, **self.options
        )
        return response.text().strip()

    def get_history(self):
        return self.conversation.responses

    def update_history(self, summary: str):
        # Update the agent's conversation context with the round summary.
        self.conversation.prompt(summary, **self.options)
