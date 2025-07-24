import spacy
import re

nlp = spacy.load("en_core_web_sm")

def extract_skills(text):
    text = text.lower()

    # Define common skill section headings
    section_headings = [
        "skills",
        "technical skills",
        "key skills",
        "core competencies",
        "areas of expertise",
        "technologies"
    ]

    # Try to extract the skill-related section
    skill_section = ""
    for heading in section_headings:
        pattern = rf"{heading}[:\n-]+(.*?)(\n[A-Z][^:\n]{0,50}[:\n-]|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            skill_section = match.group(1)
            break

    if not skill_section:
        # fallback to parsing whole text if nothing found
        skill_section = text

    # Apply spaCy to extract phrases
    doc = nlp(skill_section)
    noun_phrases = set(chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 1)

    return list(noun_phrases)
