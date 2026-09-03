import streamlit as st
from typing import Dict, Any, List
from src.agents.graph import cyclic_agent_app

def render_chat_view():
    """
    Renders the LangGraph multi-agent chat interface with explicit citations,
    cycle feedback tracking, and expandable source evidence inspection.
    """
    st.subheader("💬 Verified Scientific Assistant")
    st.caption("Cyclic multi-agent reasoning: Every claim is cross-checked against lab assays and biological databases.")

    # Initialize chat session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am your bio-cheminformatics research assistant. "
                    "You can ask me to evaluate aggregation inhibition across specific proteins "
                    "(e.g., Amyloid-beta P05067, Tau P10636) and small molecules (e.g., Resveratrol CHEMBL112, Curcumin CHEMBL148). "
                    "Every finding includes inline citations verified against our Lakehouse."
                ),
                "citations": []
            }
        ]

    # Render persistent conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("📚 Verified Citations & Assay Evidence"):
                    for cit in msg["citations"]:
                        st.markdown(f"**{cit.get('source', 'Reference')}**")
                        st.caption(cit.get("details", ""))

    # Chat prompt handler
    user_prompt = st.chat_input("Ask about compound efficacy, assay results, or aggregation kinetics...")

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Executing cyclic research, verification, and citation synthesis..."):
                initial_state = {
                    "user_query": user_prompt,
                    "iteration_count": 0,
                    "retrieved_evidence": [],
                    "missing_evidence_queries": [],
                    "claims": [],
                    "is_verified": False,
                    "verification_critique": None,
                    "final_response": None,
                    "citations": []
                }

                try:
                    graph_output = cyclic_agent_app.invoke(initial_state)
                    answer_text = graph_output.get("final_response", "No verified conclusion could be synthesized.")
                    citations_data = graph_output.get("citations", [])

                    st.markdown(answer_text)

                    if citations_data:
                        with st.expander("📚 Verified Citations & Assay Evidence"):
                            for cit in citations_data:
                                st.markdown(f"**{cit.get('source', 'Reference')}**")
                                st.caption(cit.get("details", ""))

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "citations": citations_data
                    })

                except Exception as exc:
                    error_msg = f"Agent workflow encountered an error: {str(exc)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "citations": []
                    })