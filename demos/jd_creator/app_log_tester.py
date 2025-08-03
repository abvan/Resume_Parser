# # app_log_tester.py

# import os
# from utils.llm_client import GroqLLMClient
# from module.modular_jd import (
#     generate_responsibilities,
#     generate_required_skills,
#     generate_preferred_qualifications,
#     generate_culture_benefits
# )

# # 1️⃣ Test Groq API connectivity
# def test_groq_connection():
#     print("🔍 Testing Groq API connectivity...")
#     llm = GroqLLMClient()
#     try:
#         result = llm.generate("hi there", system_prompt="You are a friendly assistant.")
#         print("✅ Groq API is working. Response:")
#         print(result)
#     except Exception as e:
#         print("❌ Error connecting to Groq API:", e)
#         exit(1)

# # 2️⃣ Ask for input variables
# def get_user_inputs():
#     role = input("Enter job role/title: ").strip()
#     context = input("Enter context (optional, press Enter to skip): ").strip()
#     return role, context

# # 3️⃣ Check template file existence
# def check_template_file():
#     template_path = os.path.join("data", "templates", "template.pdf")
#     if os.path.exists(template_path):
#         print(f"📄 Found template file at: {template_path}")
#     else:
#         print(f"⚠️ No template file found at: {template_path}")
#         print("Please place a template JD at this path before testing TemplateRAG.")

# # 4️⃣ Run modular JD generation
# def run_modular_jd(role, context):
#     print("\n🛠 Generating JD sections...\n")

#     responsibilities = generate_responsibilities(role, context)
#     print("=== Key Responsibilities ===")
#     print(responsibilities, "\n")

#     required_skills = generate_required_skills(role, context)
#     print("=== Required Skills ===")
#     print(required_skills, "\n")

#     preferred_qualifications = generate_preferred_qualifications(role, context)
#     print("=== Preferred Qualifications ===")
#     print(preferred_qualifications, "\n")

#     culture_benefits = generate_culture_benefits(role, context)
#     print("=== Company Culture & Benefits ===")
#     print(culture_benefits, "\n")

# # Main execution
# if __name__ == "__main__":
#     test_groq_connection()
#     role, context = get_user_inputs()
#     check_template_file()
#     run_modular_jd(role, context)

# Flow with Agentic

# app_log_tester.py

import os
import sys
from dotenv import load_dotenv

# Load environment variables for Groq API
load_dotenv()

# Import your modules
from module.modular_jd import (
    generate_responsibilities,
    generate_required_skills,
    generate_preferred_qualifications,
    generate_culture_benefits
)
from module.jd_builder_agentic import decide_expansion, run_template_rag
from module.skill_expand import SkillExpander
from module.bias_clarity import BiasClarityChecker
from utils.file_io import save_jd_to_pdf


def log_step(title, content):
    """Helper for formatted logging."""
    print(f"\n{'='*80}")
    print(f"📌 {title}")
    print(f"{'-'*80}")
    print(content if content else "[EMPTY]")
    print(f"{'='*80}\n")


def run_agentic_jd_flow():
    """
    Run the JD creation process end-to-end with logging at every decision step.
    """
    # 1️⃣ User Inputs
    role = input("Enter Job Role: ").strip()
    context = input("Enter Context (optional): ").strip()
    skills = input("Enter comma-separated skills: ").strip()

    # 2️⃣ Modular JD Creation
    print("\n[STEP] Generating modular JD sections...")
    responsibilities = generate_responsibilities(role, context)
    required_skills = generate_required_skills(role, context)
    preferred_qualifications = generate_preferred_qualifications(role, context)
    culture_benefits = generate_culture_benefits(role, context)

    jd_text = f"""
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
{culture_benefits.strip()}
""".strip()

    log_step("INITIAL JD", jd_text)

    # 3️⃣ Create State Object
    state = {
        "role": role,
        "context": context,
        "skills": skills,
        "jd_text": jd_text,
        "needs_expansion": False,
        "error": None
    }

    # 4️⃣ Agentic Decision: Skill Expansion Needed?
    print("\n[STEP] Deciding if Skill Expansion is needed...")
    state = decide_expansion(state)
    log_step("DECISION - NEEDS EXPANSION", state.get("needs_expansion"))

    if state.get("error"):
        log_step("ERROR", state["error"])
        sys.exit(1)

    # 5️⃣ Skill Expansion (if needed)
    if state["needs_expansion"]:
        print("\n[STEP] Expanding skills...")
        expander = SkillExpander()
        expanded_jd = expander.expand_from_jd(
            role=state["role"],
            skills=state["skills"],
            existing_jd=state["jd_text"]
        )
        state["jd_text"] = expanded_jd
        log_step("EXPANDED JD", state["jd_text"])
    else:
        print("\n[INFO] Skill Expansion not required.")

    # 6️⃣ Bias & Clarity Check
    print("\n[STEP] Running Bias & Clarity analysis...")
    checker = BiasClarityChecker()
    bias_report = checker.analyze_jd(state["jd_text"])
    log_step("BIAS & CLARITY REPORT", bias_report)

    # 7️⃣ Optional Template RAG Comparison
    if os.path.exists("data/templates/template.pdf"):
        print("\n[STEP] Comparing JD with uploaded template (Template RAG)...")
        state["uploaded_template"] = "data/templates/template.pdf"
        state = run_template_rag(state)
        log_step("TEMPLATE COMPARISON RESULT", state.get("template_comparison"))
    else:
        print("\n[INFO] No template uploaded for comparison.")

    # 8️⃣ Save to PDF
    print("\n[STEP] Saving JD to PDF...")
    pdf_path = save_jd_to_pdf(state["jd_text"])
    print(f"[SUCCESS] JD saved to: {pdf_path}")


if __name__ == "__main__":
    print("\n🚀 Running Agentic JD Builder Tester...\n")
    run_agentic_jd_flow()