# modules/bias_clarity.py

from utils.llm_client import GroqLLMClient
import json


class BiasClarityChecker:
    """
    Analyzes a job description for bias, inclusiveness, and clarity.
    Produces structured results for easy integration into the JD builder workflow.
    """

    def __init__(self):
        self.llm = GroqLLMClient()

    def analyze_jd(self, jd_text: str) -> dict:
        """
        Analyze the given JD for bias and clarity.

        Args:
            jd_text (str): The job description text (either modular_jd output
                           or merged output after skill expansion).

        Returns:
            dict: {
                "bias_score": int,
                "clarity_score": int,
                "issues": list[str],
                "suggestions": list[str]
            }
        """

        # Prompt to request structured bias & clarity analysis
        prompt = (
            "Analyze the following job description for bias, inclusiveness, and clarity.\n\n"
            f"Job Description:\n{jd_text}\n\n"
            "Task:\n"
            "1. Identify biased language and unclear terms.\n"
            "2. Rate bias on a scale of 0 (no bias) to 100 (very biased).\n"
            "3. Rate clarity on a scale of 0 (very unclear) to 100 (very clear).\n"
            "4. Suggest alternative phrasing for any biased or unclear sections.\n"
            "5. Return your findings strictly as a JSON object with keys:\n"
            "{\n"
            "  \"bias_score\": int,\n"
            "  \"clarity_score\": int,\n"
            "  \"issues\": [list of strings],\n"
            "  \"suggestions\": [list of strings]\n"
            "}"
        )

        system_prompt = (
            "You are an expert HR and DEI consultant. "
            "You specialize in reviewing job descriptions for gender neutrality, inclusiveness, "
            "and readability while providing constructive suggestions for improvement."
        )

        # Get response from LLM
        raw_response = self.llm.generate(prompt, system_prompt=system_prompt)

        # Parse JSON output safely
        try:
            analysis = json.loads(raw_response)
        except json.JSONDecodeError:
            # In case LLM returns slightly off-format text
            analysis = {
                "bias_score": None,
                "clarity_score": None,
                "issues": [],
                "suggestions": [raw_response.strip()]
            }

        return analysis