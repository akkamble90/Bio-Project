import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.common.config import settings
from src.common.logger import get_logger
from src.agents.state import AgentState
from src.agents.tools.feature_store_tool import query_feature_store
from src.agents.tools.kafka_stream_tool import fetch_latest_kafka_predictions
from src.agents.tools.chem_bio_tools import lookup_compound_info, lookup_uniprot_info

logger = get_logger("ResearcherNode")

def researcher_node(state: AgentState) -> dict:
    """
    Bio-cheminformatics researcher node. Extracts identifiers, queries
    the Lakehouse via DuckDB, inspects Kafka streams, and fetches external bio-data.
    """
    iteration = state.get("iteration_count", 0) + 1
    logger.info(f"Executing Researcher - Cycle #{iteration}")

    user_query = state.get("user_query", "")
    missing_queries = state.get("missing_evidence_queries", [])
    current_evidence = list(state.get("retrieved_evidence", []))

    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.0
    )

    extraction_prompt = f"""
    You are a computational biochemist research assistant.
    Analyze the user inquiry and any requests for missing evidence.
    Extract key identifiers: UniProt accession (e.g., P05067), ChEMBL ID (e.g., CHEMBL112),
    or relevant SQL WHERE clauses to filter experimental lakehouse records.

    User Inquiry: {user_query}
    Targeted Missing Evidence: {missing_queries if missing_queries else 'Initial sweep'}

    Respond ONLY with a JSON object matching this schema:
    {{
      "uniprot_id": "string or null",
      "chembl_id": "string or null",
      "lakehouse_sql_filter": "string or null"
    }}
    """

    response = llm.invoke([
        SystemMessage(content="You extract biochemical query parameters into strict JSON."),
        HumanMessage(content=extraction_prompt)
    ])

    try:
        clean_text = response.content.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:-3].strip()
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:-3].strip()
        params = json.loads(clean_text)
    except Exception as exc:
        logger.warning(f"JSON parsing fallback triggered: {exc}")
        params = {"uniprot_id": None, "chembl_id": None, "lakehouse_sql_filter": None}

    # 1. Query Parquet Feature Store on MinIO
    sql_filter = params.get("lakehouse_sql_filter")
    lakehouse_records = query_feature_store(where_clause=sql_filter, limit=5)
    for rec in lakehouse_records:
        current_evidence.append({
            "source_type": "ASSAY_LAKEHOUSE",
            "identifier": rec.get("assay_id", "UNKNOWN"),
            "data": rec
        })

    # 2. Inspect Live Kafka Stream for real-time predictions
    live_stream_events = fetch_latest_kafka_predictions(max_records=3)
    for ev in live_stream_events:
        current_evidence.append({
            "source_type": "KAFKA_STREAM",
            "identifier": ev.get("assay_id", "LIVE_STREAM"),
            "data": ev
        })

    # 3. Fetch ChEMBL compound details if targeted
    chembl_id = params.get("chembl_id")
    if chembl_id:
        comp_info = lookup_compound_info(chembl_id)
        if comp_info:
            current_evidence.append({
                "source_type": "CHEMBL",
                "identifier": chembl_id,
                "data": comp_info
            })

    # 4. Fetch UniProt target sequence details if targeted
    uniprot_id = params.get("uniprot_id")
    if uniprot_id:
        prot_info = lookup_uniprot_info(uniprot_id)
        if prot_info:
            current_evidence.append({
                "source_type": "UNIPROT",
                "identifier": uniprot_id,
                "data": prot_info
            })

    logger.info(f"Researcher finished cycle #{iteration}. Total evidence gathered: {len(current_evidence)}")

    return {
        "retrieved_evidence": current_evidence,
        "iteration_count": iteration,
        "missing_evidence_queries": []
    }