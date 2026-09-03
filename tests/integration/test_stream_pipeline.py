import json
import time
import pytest
from kafka import KafkaProducer, KafkaConsumer
from src.common.config import settings

@pytest.mark.integration
def test_kafka_producer_consumer_roundtrip():
    """
    Validates end-to-end event serialization and consumption 
    over the primary Kafka ingestion topic.
    """
    test_topic = settings.kafka_input_topic
    bootstrap_servers = settings.kafka_bootstrap_servers.split(",")

    # 1. Initialize Producer
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8"),
            request_timeout_ms=5000
        )
    except Exception as exc:
        pytest.skip(f"Kafka broker is not available at {bootstrap_servers}: {exc}")

    # 2. Build test payload
    unique_assay_id = f"TEST-INTEG-{int(time.time())}"
    sample_payload = {
        "assay_id": unique_assay_id,
        "timestamp": int(time.time()),
        "protein": {
            "uniprot_id": "P05067",
            "name": "Amyloid-beta",
            "sequence": "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
        },
        "drug": {
            "chembl_id": "CHEMBL112",
            "name": "Resveratrol",
            "canonical_smiles": "Oc1ccc(cc1)/C=C/c2ccc(O)cc2",
            "molecular_weight": 228.24
        },
        "conditions": {
            "temperature_celsius": 37.0,
            "ph": 7.4,
            "drug_concentration_umol": 25.0
        },
        "label": {
            "inhibition_percentage": 78.5,
            "outcome": "INHIBITOR"
        }
    }

    # 3. Publish to Kafka
    future = producer.send(test_topic, key=unique_assay_id, value=sample_payload)
    record_metadata = future.get(timeout=10)
    producer.flush()
    producer.close()

    assert record_metadata.topic == test_topic

    # 4. Initialize Consumer & Poll
    consumer = KafkaConsumer(
        test_topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=7000,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id=f"test-group-{unique_assay_id}"
    )

    found_event = None
    for message in consumer:
        if message.value and message.value.get("assay_id") == unique_assay_id:
            found_event = message.value
            break

    consumer.close()

    assert found_event is not None
    assert found_event["assay_id"] == unique_assay_id
    assert found_event["drug"]["chembl_id"] == "CHEMBL112"