import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from src.agents.tools.kafka_stream_tool import fetch_latest_kafka_predictions
from src.agents.tools.feature_store_tool import query_feature_store

def render_assay_inspector():
    """
    Renders the biophysical assay inspector view, including:
    1. Live Kafka streaming predictions monitor.
    2. Historical Lakehouse Parquet feature browser via DuckDB.
    """
    st.subheader("🔬 Real-Time Assay & Lakehouse Inspector")
    st.caption("Inspect live streaming inferences from Kafka or historical feature distributions in MinIO.")

    mode = st.radio(
        "Data Source Selection",
        ["Live Kafka Stream (`predictions-out`)", "Historical Lakehouse (`feature-store`)"],
        horizontal=True
    )

    if mode == "Live Kafka Stream (`predictions-out`)":
        col1, col2 = st.columns([1, 5])
        with col1:
            poll_clicked = st.button("🔄 Poll Kafka", use_container_width=True)
        with col2:
            st.caption("Fetches the latest scored micro-batches processed by the inference daemon.")

        events: List[Dict[str, Any]] = fetch_latest_kafka_predictions(max_records=15)

        if not events:
            st.info("No active inference records detected on Kafka topic. Verify the producer and serving daemon are running.")
            return

        # Flatten records for tabular rendering
        flattened = []
        for ev in events:
            pred = ev.get("prediction", {})
            flattened.append({
                "Assay ID": ev.get("assay_id"),
                "Protein": ev.get("protein_id"),
                "Drug": ev.get("drug_id"),
                "Class": pred.get("predicted_class"),
                "Inhibition (%)": pred.get("predicted_inhibition_pct"),
                "Confidence": pred.get("confidence_score"),
                "Latency (ms)": ev.get("latency_ms"),
                "Timestamp": ev.get("timestamp")
            })

        df = pd.DataFrame(flattened)

        # High-level KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Recent Assays", len(df))
        avg_inhib = round(df["Inhibition (%)"].mean(), 2) if not df.empty else 0.0
        kpi2.metric("Mean Inhibition", f"{avg_inhib}%")
        avg_lat = round(df["Latency (ms)"].mean(), 1) if not df.empty else 0.0
        kpi3.metric("Avg Latency", f"{avg_lat} ms")
        inhibitor_ratio = round((df["Class"] == "INHIBITOR").mean() * 100, 1) if not df.empty else 0.0
        kpi4.metric("Inhibitor Ratio", f"{inhibitor_ratio}%")

        st.dataframe(df, use_container_width=True)

    else:
        st.markdown("**Query Lakehouse Parquet via DuckDB**")
        filter_input = st.text_input(
            "SQL WHERE Filter (optional)",
            value="is_potent_inhibitor = 1",
            help="Filter expressions executed directly over S3 Parquet via DuckDB"
        )
        limit_val = st.slider("Record Limit", min_value=5, max_value=50, value=15)

        records = query_feature_store(where_clause=filter_input, limit=limit_val)

        if records:
            lakehouse_df = pd.DataFrame(records)
            st.dataframe(lakehouse_df, use_container_width=True)

            # Distribution visualizations
            if "target_inhibition_pct" in lakehouse_df.columns:
                st.bar_chart(lakehouse_df.set_index("assay_id")["target_inhibition_pct"])
        else:
            st.warning("No records matched the filter or MinIO bucket is empty.")