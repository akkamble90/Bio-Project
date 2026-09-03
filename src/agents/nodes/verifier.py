import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.common.config import settings
from src.common.logger import get_logger
from src.agents.state import AgentState

logger = get_logger("VerifierNode")

def verifier_node(state: AgentState) -> dict:
    """
    Strict fact-verification node. Cross-checks claims against retrieved evidence.
    Triggers re-retrieval loop if essential ground-truth points are missing.
    """
    logger.info("Executing Verifier: Cross-referencing claims against evidence...")

    user_query = state.get("user_query", "")
    evidence = state.get("retrieved_evidence", [])
    iteration = state.get("iteration_count", 1)

    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.0
    )

    verification_prompt = f"""
    You are a strict bio-cheminformatics scientific verifier.
    Evaluate whether the gathered evidence provides sufficient factual proof to answer the user query.
    Every numerical inhibition score, compound efficacy claim, or protein target assertion MUST be supported by evidence.

    User Inquiry: {user_query}
    Retrieved Evidence:
    {json.dumps(evidence, indent=2, default=str)}

    Respond ONLY with a valid JSON object matching this schema:
    {{
      "is_verified": true | false,
      "critique": "Brief description of verification status or missing data",
      "missing_evidence_queries": ["specific missing piece of information if false"]
    }}
    """

    response = llm.invoke([
        SystemMessage(content="You verify biochemical claims strictly against provided data."),
        HumanMessage(content=verification_prompt)
    ])

    try:
        clean_text = response.content.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:-3].strip()
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:-3].strip()
        result = json.loads(clean_text)
    except Exception as exc:
        logger.error(f"Verifier output parse failed: {exc}. Defaulting based on evidence availability.")
        result = {
            "is_verified": len(evidence) > 0,
            "critique": "Fallback parsing pass based on evidence presence.",
            "missing_evidence_queries": []
        }

    # Hard cap circuit breaker: terminate loop if threshold hit
    if iteration >= settings.max_agent_cycles:
        logger.warning(f"Cycle threshold ({settings.max_agent_cycles}) reached. Forcing verification to complete.")
        result["is_verified"] = True

    logger.info(f"Verification outcome: {result.get('is_verified')} | Critique: {result.get('critique')}")

    return {
        "is_verified": result.get("is_verified", False),
        "verification_critique": result.get("critique"),
        "missing_evidence_queries": result.get("missing_evidence_queries", [])
    }