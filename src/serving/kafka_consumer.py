import json
import time
from kafka import KafkaConsumer, KafkaProducer
from src.common.config import settings
from src.common.logger import get_logger
from src.serving.predictor import AggregationPredictor

logger = get_logger("StreamingConsumerService")

def run_consumer_service():
    logger.info("Initializing Real-Time Stream Scoring Daemon...")
    predictor = AggregationPredictor()

    consumer = KafkaConsumer(
        settings.kafka_inference_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers.split(","),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="model-inference-workers",
        value_deserializer=lambda v: json.loads(v.decode("utf-8"))
    )

    producer = KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers.split(","),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8")
    )

    logger.info(f"Listening on '{settings.kafka_inference_topic}', emitting to '{settings.kafka_output_topic}'")

    try:
        for message in consumer:
            payload = message.value
            if not payload:
                continue

            assay_id = payload.get("assay_id", "UNKNOWN")
            smiles = payload.get("drug_smiles", "")
            seq = payload.get("protein_sequence", "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA")

            # Run PyTorch inference
            result = predictor.predict(
                smiles=smiles,
                sequence=seq,
                molecular_weight=float(payload.get("drug_mw", 300.0)),
                temp_c=float(payload.get("temp_c", 37.0)),
                ph=float(payload.get("solution_ph", 7.4)),
                drug_conc_um=float(payload.get("drug_conc_um", 20.0))
            )

            # Package formatted response matching data/schemas/inference_response.json
            out_event = {
                "assay_id": assay_id,
                "protein_id": payload.get("protein_id", "P05067"),
                "drug_id": payload.get("drug_id", "CHEMBL000"),
                "model_version": result["model_version"],
                "prediction": {
                    "predicted_inhibition_pct": result["predicted_inhibition_pct"],
                    "predicted_class": result["predicted_class"],
                    "confidence_score": result["confidence_score"]
                },
                "latency_ms": result["latency_ms"],
                "timestamp": int(time.time())
            }

            producer.send(settings.kafka_output_topic, key=assay_id, value=out_event)
            producer.flush()

            logger.info(
                f"[SCORED] {assay_id} -> {result['predicted_class']} "
                f"({result['predicted_inhibition_pct']}%) in {result['latency_ms']}ms"
            )

    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Closing consumer...")
    finally:
        consumer.close()
        producer.close()

if __name__ == "__main__":
    run_consumer_service()