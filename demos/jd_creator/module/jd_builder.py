# Old non agentic jd builder

from module.modular_jd import (
    generate_responsibilities,
    generate_required_skills,
    generate_preferred_qualifications,
    generate_culture_benefits
)


def build_full_jd(role: str, context: str = "") -> str:
    """
    Generate a complete job description by composing key sections.
    """

    # Generate each section using the modular functions
    responsibilities = generate_responsibilities(role, context)
    required_skills = generate_required_skills(role, context)
    preferred_qualifications = generate_preferred_qualifications(role, context)
    culture_and_benefits = generate_culture_benefits(role, context)

    # Format and concatenate all parts
    full_jd = f"""
    **Job Title:** {role}

    ---

    ### Key Responsibilities:
    {responsibilities.strip()}

    ---

    ### Required Skills & Qualifications:
    {required_skills.strip()}

    ---

    ### Preferred Qualifications:
    {preferred_qualifications.strip()}

    ---

    ### Company Culture & Benefits:
    {culture_and_benefits.strip()}
    """.strip()

    return full_jd