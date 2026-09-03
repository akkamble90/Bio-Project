from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """
    Shared execution state across the LangGraph multi-agent loop.
    """
    user_query: str
    iteration_count: int
    retrieved_evidence: List[Dict[str, Any]]
    missing_evidence_queries: List[str]
    claims: List[Dict[str, Any]]
    is_verified: bool
    verification_critique: Optional[str]
    final_response: Optional[str]
    citations: List[Dict[str, str]]