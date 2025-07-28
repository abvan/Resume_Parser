from utils.llm_client import GroqLLMClient

llm = GroqLLMClient()


def generate_responsibilities(role: str, context: str = "") -> str:
    prompt = (
        f"Write a list of 5-7 key responsibilities for a {role}.\n"
        f"Context: {context}\n"
        "Make them action-oriented, clear, and relevant to the role."
    )

    system_prompt = (
        "You are an expert HR professional helping to craft job descriptions. "
        "Your responses should be clear, concise, and tailored to the specified role and context."
    )

    return llm.generate(prompt, system_prompt=system_prompt)


def generate_required_skills(role: str, context: str = "", responsibilities: str = "") -> str:
    prompt = (
        f"List the most essential skills and qualifications required for a {role}.\n"
        f"Context: {context}\n"
    )
    if responsibilities:
        prompt += f"\nRefer to these responsibilities while listing relevant skills:\n{responsibilities}\n"
    prompt += "Include technical and soft skills where applicable."

    system_prompt = (
        "You are assisting in writing job descriptions. "
        "Generate a clear and comprehensive list of required skills for the given role."
    )

    return llm.generate(prompt, system_prompt=system_prompt)


def generate_preferred_qualifications(
    role: str,
    context: str = "",
    required_skills: str = "",
    responsibilities: str = ""
) -> str:
    prompt = (
        f"List preferred (but not mandatory) qualifications for a {role}.\n"
        f"Context: {context}\n"
    )
    if responsibilities:
        prompt += f"\nResponsibilities:\n{responsibilities}\n"
    if required_skills:
        prompt += f"\nRequired Skills:\n{required_skills}\n"
    prompt += (
        "Think of certifications, experience, industry knowledge, or education."
    )

    system_prompt = (
        "You are helping create a job description. "
        "Provide 3–5 preferred qualifications that would strengthen a candidate’s application."
    )

    return llm.generate(prompt, system_prompt=system_prompt)


def generate_culture_benefits(
    role: str,
    context: str = "",
    responsibilities: str = "",
    required_skills: str = "",
    preferred_qualifications: str = ""
) -> str:
    prompt = (
        f"Write a short paragraph highlighting company culture and key benefits for a {role}.\n"
        f"Context: {context}\n"
    )
    if responsibilities:
        prompt += f"\nResponsibilities:\n{responsibilities}\n"
    if required_skills:
        prompt += f"\nRequired Skills:\n{required_skills}\n"
    if preferred_qualifications:
        prompt += f"\nPreferred Qualifications:\n{preferred_qualifications}\n"

    prompt += (
        "Mention aspects like team environment, remote work, career growth, or wellness programs."
    )

    system_prompt = (
        "You're generating the 'Company Culture & Benefits' section of a job description. "
        "Be positive, inclusive, and briefly highlight perks and values."
    )

    return llm.generate(prompt, system_prompt=system_prompt)