import os
import requests
from dotenv import load_dotenv

class GroqLLMClient:
    def __init__(self, model: str = "llama-3.1-8b-instant", temperature: float = 0.7):
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in the env file, please recheck.")
        
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = model
        self.temperature = temperature
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate(self, prompt: str, system_prompt: str = None, max_tokens: int = 1024) -> str:
        """
        Send a prompt to the Groq LLM and return the generated response.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=15)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return content.strip()
        except requests.exceptions.RequestException as e:
            print(f"[LLM ERROR] {e}")
            return "Error: Failed to generate response from Groq API."


        

