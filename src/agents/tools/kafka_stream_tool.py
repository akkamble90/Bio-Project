import json
from typing import List, Dict, Any
from kafka import KafkaConsumer
from src.common.config import settings
from src.common.logger import get_logger

logger = get_logger("KafkaStreamTool")

def fetch_latest_kafka_predictions(max_records: int = 3, timeout_ms: int = 1500) -> List[Dict[str, Any]]:
    """
    Polls the latest scored assay inferences directly from the Kafka predictions topic.
    """
    logger.info(f"Inspecting active Kafka topic '{settings.kafka_output_topic}'...")
    events: List[Dict[str, Any]] = []

    try:
        consumer = KafkaConsumer(
            settings.kafka_output_topic,
            bootstrap_servers=settings.kafka_bootstrap_servers.split(","),
            auto_offset_reset="latest",
            enable_auto_commit=False,
            consumer_timeout_ms=timeout_ms,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            group_id="agent-ephemeral-inspector"
        )

        for message in consumer:
            if message.value:
                events.append(message.value)
                if len(events) >= max_records:
                    break

        consumer.close()
        logger.info(f"Retrieved {len(events)} real-time inference messages from Kafka.")
    except Exception as exc:
        logger.warning(f"Could not read from Kafka topic '{settings.kafka_output_topic}': {exc}. Stream may be idle.")

    return events