import streamlit as st
import json
import requests
from src.common.config import settings
from src.agents.graph import cyclic_agent_app
from src.agents.tools.kafka_stream_tool import fetch_latest_kafka_predictions

st.set_page_config(
    page_title="Protein Aggregation Platform & Verified AI Agent",
    page_icon="🧬",
    layout="wide"
)

st.title(" Protein Aggregation Prediction Platform")
st.caption("Real-Time Streaming Inference, Parquet Feature Store & Cyclic Multi-Agent Fact-Verification")

tabs = st.tabs([" Research Chatbot & Citation Agent", " Real-Time Kafka Monitor", "🧪 Single Assay Scorer"])


# TAB 1: CYCLIC MULTI-AGENT CHATBOT
with tabs[0]:
    st.subheader("Cyclic Verified Bio-Cheminformatics Assistant")
    st.markdown(
        "Ask questions regarding protein targets (e.g., Amyloid-beta, Tau), compound efficacy, "
        "or stream predictions. Every claim is cross-checked against MinIO Parquet assays & bioinformatics APIs."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                with st.expander("📚 Verified Citations & Assay Evidence"):
                    for cit in msg["citations"]:
                        st.markdown(f"**{cit['source']}**")
                        st.caption(cit["details"])

    user_query = st.chat_input("e.g., What is the inhibition efficacy of Resveratrol (CHEMBL112) against Amyloid-beta (P05067)?")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Executing cyclic research and strict fact-verification..."):
                initial_state = {
                    "user_query": user_query,
                    "iteration_count": 0,
                    "retrieved_evidence": [],
                    "missing_evidence_queries": [],
                    "claims": [],
                    "is_verified": False,
                    "verification_critique": None,
                    "final_response": None,
                    "citations": []
                }

                graph_result = cyclic_agent_app.invoke(initial_state)
                final_text = graph_result.get("final_response", "Error generating grounded response.")
                citations = graph_result.get("citations", [])

                st.markdown(final_text)

                if citations:
                    with st.expander("📚 Verified Citations & Assay Evidence"):
                        for cit in citations:
                            st.markdown(f"**{cit['source']}**")
                            st.caption(cit["details"])

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": final_text,
                    "citations": citations
                })

# TAB 2: REAL-TIME KAFKA STREAM MONITOR
with tabs[1]:
    st.subheader("Live Prediction Stream (`predictions-out`)")
    col1, col2 = st.columns([1, 4])
    with col1:
        poll_btn = st.button(" Poll Kafka Stream")
    
    events = fetch_latest_kafka_predictions(max_records=10)
    if events:
        st.dataframe(events, use_container_width=True)
    else:
        st.info("No recent predictions found in Kafka. Ensure kafka_producer and streaming consumer are running.")

# TAB 3: SINGLE ASSAY MANUAL SCORING
with tabs[2]:
    st.subheader("Manual Assay Parameter Scoring")
    with st.form("manual_assay_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            smiles_in = st.text_input("Canonical SMILES", value="Oc1ccc(cc1)/C=C/c2ccc(O)cc2")
            seq_in = st.text_area("Protein Sequence (Peptide)", value="DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA")
            mw_in = st.number_input("Molecular Weight", value=228.24)
        with col_b:
            temp_in = st.slider("Temperature (°C)", min_value=20.0, max_value=60.0, value=37.0)
            ph_in = st.slider("Solution pH", min_value=3.0, max_value=11.0, value=7.4)
            conc_in = st.slider("Drug Concentration (µM)", min_value=1.0, max_value=100.0, value=25.0)

        submitted = st.form_submit_button("Run Real-Time Prediction")

    if submitted:
        req_payload = {
            "canonical_smiles": smiles_in,
            "protein_sequence": seq_in,
            "molecular_weight": mw_in,
            "temperature_celsius": temp_in,
            "ph": ph_in,
            "drug_concentration_umol": conc_in
        }
        try:
            resp = requests.post(settings.inference_api_url, json=req_payload, timeout=5)
            if resp.status_code == 200:
                res_data = resp.json()
                pred = res_data["prediction"]
                st.success(f"Outcome: **{pred['predicted_class']}** | Inhibition: **{pred['predicted_inhibition_pct']}%** (Confidence: {pred['confidence_score']:.2f})")
                st.caption(f"Inference Latency: {res_data['latency_ms']} ms")
            else:
                st.error(f"Inference service returned status {resp.status_code}")
        except Exception as err:
            st.error(f"Failed to connect to FastAPI serving at {settings.inference_api_url}: {err}")