import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.common.config import settings
from src.common.logger import get_logger
from src.agents.state import AgentState
from src.agents.guardrails import enforce_biochemical_guardrail

logger = get_logger("CitationSynthesizerNode")

def citation_synthesizer_node(state: AgentState) -> dict:
    """
    Synthesizes the final scientifically grounded response with inline citations
    and runs it through the biochemical guardrail before returning.
    """
    logger.info("Executing Citation Synthesizer: Building final verified response...")

    user_query = state.get("user_query", "")
    evidence = state.get("retrieved_evidence", [])

    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.1
    )

    synthesis_prompt = f"""
    You are an expert computational biochemist synthesizer.
    Answer the user query using ONLY the verified laboratory evidence provided below.

    Mandatory Rules:
    1. Every factual assertion, % inhibition number, and target relationship MUST have an inline citation.
       Format: [Assay: <assay_id>], [UniProt: <uniprot_id>], or [ChEMBL: <chembl_id>].
    2. Do NOT extrapolate or introduce external unverified drugs/targets.
    3. Conclude your answer with an itemized 'References & Citations' list.

    User Inquiry: {user_query}
    Verified Evidence:
    {json.dumps(evidence, indent=2, default=str)}
    """

    response = llm.invoke([
        SystemMessage(content="You write peer-reviewed quality biochemical summaries with strict inline citations."),
        HumanMessage(content=synthesis_prompt)
    ])

    raw_answer = response.content

    # Validate through biochemical guardrail
    guardrail_result = enforce_biochemical_guardrail(raw_answer, evidence)

    final_text = (
        guardrail_result["data"]["summary"]
        if guardrail_result.get("is_valid")
        else guardrail_result.get("sanitized_response", raw_answer)
    )

    # Format structured citations for Streamlit collapsible drawer
    structured_citations = []
    for item in evidence:
        structured_citations.append({
            "source": f"{item.get('source_type')}: {item.get('identifier')}",
            "details": str(item.get("data", {}))[:250] + "..."
        })

    logger.info("Synthesis complete and verified against guardrails.")

    return {
        "final_response": final_text,
        "citations": structured_citations
    }