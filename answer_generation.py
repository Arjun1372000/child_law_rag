# answer_generation.py
"""
Domain validation guardrails, prompt structuring, and grounded legal answer generation
for Indian Child Protection and Rights Awareness.
"""

from typing import List, Dict, Tuple, Optional
from config import call_llm, CHILD_LAW_KEYWORDS, CITATION_MAP


def validate_child_law_domain(question: str) -> Tuple[bool, str, Optional[str]]:
    """
    Dual-Stage Guardrail to ensure query focus remains strictly on child protection and child rights.
    
    Stage 1: Fast lexical match for high-confidence child protection markers.
    Stage 2: Few-shot semantic intent classifier via Gemini.
    
    Returns:
        Tuple of (is_valid: bool, classification: str, explanation_or_redirection: Optional[str])
    """
    q_clean = question.strip().lower()
    
    # Fast-path check: Explicit high-signal statutory child terms
    strong_child_markers = [
        "pocso", "child", "children", "minor", "juvenile", "cwc", "jjb",
        "child abuse", "child labour", "child marriage", "tender age",
        "penetrative sexual assault", "right to education", "rte act",
        "juvenile justice", "foster care", "adoption of orphan"
    ]
    if any(m in q_clean for m in strong_child_markers):
        return True, "IN_DOMAIN_CHILD_LAW", None

    # Stage 2: Semantic Intent Guardrail using LLM classifier
    guardrail_prompt = f"""You are a specialized legal triage classifier for an Indian Child Rights and Protection Legal System.
Classify the following user input into EXACTLY ONE of three categories:
1. IN_DOMAIN: Questions regarding child rights, child protection, juvenile justice, offences against minors (POCSO), foster care, child labour, or guardianship in India.
2. OUT_OF_DOMAIN_LEGAL: Legal questions that do NOT involve children or minors (e.g., adult criminal law, murder/theft involving adults, income tax, corporate law, divorce/alimony without minor custody, property disputes).
3. OFF_TOPIC: General conversation, greetings, coding, math, general non-legal inquiries, or creative writing.

User Input: "{question}"

Respond in the format:
CATEGORY: <IN_DOMAIN | OUT_OF_DOMAIN_LEGAL | OFF_TOPIC>
REASON: <One brief sentence>"""

    try:
        response = call_llm(guardrail_prompt)
        first_line = response.split("\n")[0].upper()
        
        if "IN_DOMAIN" in first_line:
            return True, "IN_DOMAIN_CHILD_LAW", None
            
        elif "OUT_OF_DOMAIN_LEGAL" in first_line:
            redirection = (
                "⚠️ **Domain Notice:** This legal assistant specializes exclusively in **Indian Child Protection and Child Rights Laws** "
                "(such as the POCSO Act 2012, Juvenile Justice Act 2015, and Constitutional protections for children).\n\n"
                "Your query appears to relate to general adult criminal, civil, or commercial law. "
                "Please consult the relevant statutory codes (e.g., BNS, BNSS, CPC) or a certified legal advocate for non-child legal matters."
            )
            return False, "OUT_OF_DOMAIN_LEGAL", redirection
            
        else:
            redirection = (
                "ℹ️ **Focus Notice:** I am an AI Legal Assistant specialized strictly in **Child Protection Laws and Rights Awareness in India**.\n\n"
                "Please ask questions related to:\n"
                "• Protection of Children from Sexual Offences (POCSO Act)\n"
                "• Juvenile Justice (Care and Protection of Children) Act\n"
                "• Child Labour, Right to Education (RTE), and Fundamental Child Rights\n"
                "• Procedures of Child Welfare Committees (CWC) and Juvenile Justice Boards (JJB)."
            )
            return False, "OFF_TOPIC", redirection

    except Exception as e:
        # Fallback to keyword presence if LLM call fails
        is_valid = any(k in q_clean for k in CHILD_LAW_KEYWORDS)
        if is_valid:
            return True, "IN_DOMAIN_CHILD_LAW", None
        return False, "OFF_TOPIC", "Please enter a question regarding Indian child protection or child rights laws."


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict],
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Generate a grounded legal response using retrieved statutory chunks.
    
    Args:
        question: User's legal query
        retrieved_chunks: List of retrieved relevant legal provisions
        conversation_history: Optional list of previous chat messages
        
    Returns:
        Structured legal answer with statutory citations and disclaimers.
    """
    CITATION_MAP.clear()

    if not retrieved_chunks:
        return (
            "Insufficient statutory legal information found in the available Central Indian Child Law documents.\n"
            "Please consult official legal gazettes or legal aid authorities."
        )

    # Build context with clear act and section attribution
    context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks):
        cite_key = f"CITE_{idx+1}"
        file_name = chunk.get("file", "Unknown")
        act_title = chunk.get("act_title", file_name)
        section_hint = chunk.get("section_hint", f"Chunk {chunk.get('chunk_id', idx)}")
        
        citation_str = f"[{act_title}, {section_hint}]"
        CITATION_MAP[cite_key] = citation_str
        
        context_blocks.append(
            f"--- Document Source {cite_key} ---\n"
            f"Statute: {act_title}\n"
            f"Provision: {section_hint}\n"
            f"Text:\n{chunk['text'].strip()}\n"
        )

    context_str = "\n".join(context_blocks)

    system_instruction = """You are an AI Legal Assistant specialized in Indian Child Protection and Child Rights Laws.
Your objective is to provide accurate, grounded, and accessible legal information based strictly on Central Indian legislation (including POCSO Act 2012, Juvenile Justice Act 2015, and Constitutional provisions).

Strict Guidelines:
1. Ground every substantive statement strictly on the provided Legal Context.
2. Quote or reference specific sections (e.g., Section 4, Section 19, Article 21A) whenever present in the context.
3. Use statutory citations in the format: [Act Name, Section/Provision] right after the relevant statement.
4. If the context does not contain sufficient information to answer the question, clearly state that the available central statutory documents do not contain the answer.
5. If the query mentions active child sexual abuse, highlight the mandatory reporting duty under Section 19 of the POCSO Act.
6. Provide answers in clear, structured paragraphs suitable for both legal comprehension and layperson readability.
7. Always conclude with a brief legal disclaimer."""

    prompt = f"""Legal Context:
{context_str}

User Question:
{question}

Provide a comprehensive, accurately cited legal response grounded exclusively in the context above:"""

    try:
        response_text = call_llm(prompt, system_instruction=system_instruction)
        
        # Append statutory disclaimer if not already present
        disclaimer = (
            "\n\n---\n*Disclaimer: This response is provided for legal awareness and informational purposes "
            "under Indian law and does not constitute formal legal advice. In emergencies or child rights violations, "
            "contact Childline (1098) or the nearest Child Welfare Committee (CWC).*"
        )
        if "Disclaimer" not in response_text:
            response_text += disclaimer
            
        return response_text

    except Exception as e:
        return f"Error generating grounded legal answer: {e}"
