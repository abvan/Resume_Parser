# modules/skill_expand.py

from utils.llm_client import GroqLLMClient


class SkillExpander:
    """
    Expands a job description by enriching skills into detailed, role-specific responsibilities.
    Designed to be called after modular_jd generation.
    """

    def __init__(self):
        self.llm = GroqLLMClient()

    def _normalize_skills(self, skills):
        """Ensure skills are in list form and stripped of whitespace."""
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",") if s.strip()]
        return skills

    def expand_from_jd(self, role: str, skills, existing_jd: str) -> str:
        """
        Given a role, skills, and existing JD, generate richer responsibilities tied to skills.
        
        Args:
            role (str): Job title
            skills (list or str): Skills relevant to the role
            existing_jd (str): The current generated JD text

        Returns:
            str: Enriched responsibilities text to be merged into the JD
        """
        skills_list = self._normalize_skills(skills)

        # Build prompt
        prompt = (
            f"Role: {role}\n"
            f"Skills: {', '.join(skills_list)}\n\n"
            f"Existing Job Description:\n{existing_jd}\n\n"
            "Task:\n"
            "1. Review the JD and identify where these skills are mentioned but not well-detailed.\n"
            "2. For each skill, create one or more role-specific, actionable responsibility statements.\n"
            "3. Use professional, concise language consistent with the JD style.\n"
            "4. Return only the new/enhanced responsibilities as bullet points."
        )

        system_prompt = (
            "You are an HR expert who specializes in crafting precise, role-specific job descriptions. "
            "You enhance existing descriptions by expanding skill mentions into clear, actionable responsibilities."
        )

        # Call LLM
        expanded_text = self.llm.generate(prompt, system_prompt=system_prompt)
        return expanded_text.strip()