import os
import streamlit as st
import requests
from src.common.config import settings
from src.ui.components.chat_view import render_chat_view
from src.ui.components.assay_inspector import render_assay_inspector
from src.ui.components.molecule_viewer import render_molecule_viewer

# 1. Page Configuration
st.set_page_config(
    page_title="Protein Aggregation AI Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom CSS
css_path = os.path.join(os.path.dirname(__file__), "style", "custom.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 3. Sidebar Infrastructure Status Health Checks
st.sidebar.title("🧬 System Telemetry")
st.sidebar.caption("Distributed Biochemical Infrastructure")

def check_service_health(url: str) -> bool:
    try:
        res = requests.get(url, timeout=1.5)
        return res.status_code in [200, 403, 401]
    except Exception:
        return False

minio_online = check_service_health(f"{settings.minio_endpoint}/minio/health/live")
serving_online = check_service_health(f"http://localhost:8000/health")

col_sb1, col_sb2 = st.sidebar.columns(2)
col_sb1.metric("MinIO S3", "ONLINE" if minio_online else "OFFLINE")
col_sb2.metric("Inference API", "ONLINE" if serving_online else "OFFLINE")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Kafka Broker:** `{settings.kafka_bootstrap_servers}`")
st.sidebar.markdown(f"**Input Stream:** `{settings.kafka_input_topic}`")
st.sidebar.markdown(f"**Output Stream:** `{settings.kafka_output_topic}`")
st.sidebar.markdown(f"**LLM Reasoner:** `{settings.groq_model}`")
st.sidebar.markdown(f"**Max Agent Cycles:** `{settings.max_agent_cycles}`")

# 4. Main Application Header
st.title("Protein Aggregation Prediction & Research Platform")
st.markdown(
    "Real-time streaming featurization, multi-modal neural screening, "
    "and cyclic LangGraph multi-agent verification over experimental lakehouse data."
)

# 5. Core Navigation Tabs
tab_chat, tab_inspector, tab_mol = st.tabs([
    " Verified Research Assistant",
    " Assay & Stream Inspector",
    " Molecular Structure & Drug-Likeness"
])

with tab_chat:
    render_chat_view()

with tab_inspector:
    render_assay_inspector()

with tab_mol:
    render_molecule_viewer()