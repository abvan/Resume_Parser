import os
from dotenv import load_dotenv
from groq import Groq


class GroqLLMClient:
    def __init__(self, model: str = "llama-3.3-70b-versatile", temperature: float = 0.7):
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in the env file, please recheck.")
        
        # Initialize Groq SDK client
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: str = None, max_tokens: int = 1024) -> str:
        """
        Send a prompt to the Groq LLM and return the generated response.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return "Error: Failed to generate response from Groq API."