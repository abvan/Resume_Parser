# modules/jd_builder_agentic.py

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from module import modular_jd
from module.skill_expand import SkillExpander
from module.bias_clarity import BiasClarityChecker
from module.template_rag import TemplateRAG
from utils.llm_client import GroqLLMClient


# -------------------------
# Define Agentic State
# -------------------------
class JDState(TypedDict):
    role: str
    skills: str
    context: Optional[str]
    uploaded_template: Optional[str]  # Path or text
    jd_text: Optional[str]            # Current JD
    bias_clarity_report: Optional[dict]
    template_comparison: Optional[dict]
    needs_expansion: Optional[bool]


# -------------------------
# Step 1: Generate Modular JD
# -------------------------
def generate_modular_jd(state: JDState) -> JDState:
    jd = modular_jd.generate_responsibilities(state["role"], state.get("context", ""))

    required_skills = modular_jd.generate_required_skills(
        state["role"], state.get("context", ""), jd
    )

    preferred_qualifications = modular_jd.generate_preferred_qualifications(
        state["role"], state.get("context", ""), required_skills, jd
    )

    culture_benefits = modular_jd.generate_culture_benefits(
        state["role"], state.get("context", ""),
        responsibilities=jd,
        required_skills=required_skills,
        preferred_qualifications=preferred_qualifications
    )

    full_jd = f"""
    **Job Title:** {state['role']}

    ### Key Responsibilities:
    {jd.strip()}

    ### Required Skills & Qualifications:
    {required_skills.strip()}

    ### Preferred Qualifications:
    {preferred_qualifications.strip()}

    ### Company Culture & Benefits:
    {culture_benefits.strip()}
    """.strip()

    state["jd_text"] = full_jd
    return state


# -------------------------
# Step 2: Decide if JD Needs Skill Expansion
# -------------------------

llm_client = GroqLLMClient()

def decide_expansion(state: JDState) -> JDState:
    """
    Uses Groq LLaMA 3.1 to decide if the JD needs skill expansion.
    Retries if the LLM output is not strictly 'YES' or 'NO'.
    """
    prompt_base = f"""
    You are an expert HR professional evaluating a job description.

    JOB DESCRIPTION:
    {state['jd_text']}

    QUESTION:
    Does this JD clearly connect each listed skill to specific, concrete responsibilities? 
    If not, it may need a 'Skill Expansion' step.

    ANSWER FORMAT:
    Respond with ONLY one word, either:
    YES
    NO
    """

    system_prompt = (
        "You are an experienced recruiter and hiring consultant. "
        "Be concise and decisive."
    )

    allowed_answers = {"YES", "NO"}
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        # Build prompt
        prompt = prompt_base
        if attempt > 1:
            print(f"[Retry {attempt}] Invalid LLM response, trying again...")
            prompt += "\nRemember: ONLY answer YES or NO."

        # Call LLM
        decision = llm_client.generate(prompt, system_prompt=system_prompt).strip().upper()

        # If valid → return decision
        if decision in allowed_answers:
            state["needs_expansion"] = decision == "YES"
            return state

    # If we exhausted retries → return an error message in state
    state["needs_expansion"] = None
    state["error"] = "Please retry the step for me."
    return state


# -------------------------
# Step 3: Skill Expansion
# -------------------------
def expand_skills(state: JDState) -> JDState:
    if not state.get("needs_expansion", False):
        return state

    expander = SkillExpander()
    expanded_jd = expander.expand_from_jd(
        role=state["role"],
        skills=state["skills"],
        existing_jd=state["jd_text"]
    )
    state["jd_text"] = expanded_jd
    return state


# -------------------------
# Step 4: Bias & Clarity Check
# -------------------------
def run_bias_clarity(state: JDState) -> JDState:
    checker = BiasClarityChecker()
    report = checker.analyze_jd(state["jd_text"])
    state["bias_clarity_report"] = report
    return state


# -------------------------
# Step 5: Template RAG Comparison (Optional)
# -------------------------
def run_template_rag(state: JDState) -> JDState:
    if not state.get("uploaded_template"):
        state["template_comparison"] = None
        return state

    rag_checker = TemplateRAG()
    comparison = rag_checker.compare_to_template(
        generated_jd=state["jd_text"],
        uploaded_file_path=state["uploaded_template"]
    )
    state["template_comparison"] = comparison
    return state


# -------------------------
# Build LangGraph Workflow
# -------------------------
def run_agentic_jd_builder(
    role: str,
    skills: str,
    context: Optional[str] = "",
    uploaded_template: Optional[str] = None
) -> JDState:

    # Initial state
    initial_state: JDState = {
        "role": role,
        "skills": skills,
        "context": context,
        "uploaded_template": uploaded_template,
        "jd_text": None,
        "bias_clarity_report": None,
        "template_comparison": None,
        "needs_expansion": None
    }

    # Define LangGraph
    workflow = StateGraph(JDState)

    workflow.add_node("generate_modular_jd", generate_modular_jd)
    workflow.add_node("decide_expansion", decide_expansion)
    workflow.add_node("expand_skills", expand_skills)
    workflow.add_node("bias_clarity", run_bias_clarity)
    workflow.add_node("template_rag", run_template_rag)

    # Add edges
    workflow.add_edge("generate_modular_jd", "decide_expansion")
    workflow.add_conditional_edges(
        "decide_expansion",
        lambda s: "expand_skills" if s["needs_expansion"] else "bias_clarity",
        {"expand_skills": "expand_skills", "bias_clarity": "bias_clarity"}
    )
    workflow.add_edge("expand_skills", "bias_clarity")
    workflow.add_edge("bias_clarity", "template_rag")
    workflow.add_edge("template_rag", END)

    # Compile and run
    app = workflow.compile()
    final_state = app.invoke(initial_state)

    return final_state