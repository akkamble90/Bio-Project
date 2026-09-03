from typing import Literal
from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.nodes.researcher import researcher_node
from src.agents.nodes.verifier import verifier_node
from src.agents.nodes.citation_synthesizer import citation_synthesizer_node
from src.common.config import settings
from src.common.logger import get_logger

logger = get_logger("AgentGraph")

def verification_gate(state: AgentState) -> Literal["researcher", "citation_synthesizer"]:
    """
    Cyclic decision router:
    - If facts remain unverified AND cycle count is below MAX_AGENT_CYCLES: route back to 'researcher'.
    - If verified OR cycle threshold is reached: proceed to 'citation_synthesizer'.
    """
    is_verified = state.get("is_verified", False)
    iteration = state.get("iteration_count", 0)
    max_cycles = settings.max_agent_cycles

    logger.info(f"Gate check: verified={is_verified}, iteration={iteration}/{max_cycles}")

    if not is_verified and iteration < max_cycles:
        logger.info("Routing back: claims unverified, executing targeted retrieval cycle...")
        return "researcher"
    
    logger.info("Routing forward to citation synthesis.")
    return "citation_synthesizer"

# Instantiate StateGraph using our typed schema
workflow = StateGraph(AgentState)

# Register execution nodes
workflow.add_node("researcher", researcher_node)
workflow.add_node("verifier", verifier_node)
workflow.add_node("citation_synthesizer", citation_synthesizer_node)

# Wire deterministic and conditional transitions
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "verifier")

workflow.add_conditional_edges(
    "verifier",
    verification_gate,
    {
        "researcher": "researcher",
        "citation_synthesizer": "citation_synthesizer"
    }
)

workflow.add_edge("citation_synthesizer", END)

# Compile into executable runnable
cyclic_agent_app = workflow.compile()