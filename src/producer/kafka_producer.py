import sys
import time
import json
from kafka import KafkaProducer
from src.common.config import settings
from src.common.logger import get_logger
from src.producer.assay_simulator import AssaySimulator

logger = get_logger("KafkaProducerService")

def create_producer(bootstrap_servers: str, max_retries: int = 10, retry_delay: int = 3) -> KafkaProducer:
    """Connects to the Kafka broker with retry backoff."""
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Connecting to Kafka at {bootstrap_servers} (Attempt {attempt}/{max_retries})...")
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8"),
                acks="all",
                retries=3
            )
            logger.info("Successfully connected to Kafka.")
            return producer
        except Exception as e:
            logger.warning(f"Kafka unavailable: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    logger.error("Failed to connect to Kafka after maximum attempts.")
    sys.exit(1)

def run_stream(rate_seconds: float = 1.0, max_messages: int = 0):
    """
    Publishes streaming assay events at a controlled rate.
    Set max_messages=0 for infinite streaming.
    """
    producer = create_producer(settings.kafka_bootstrap_servers)
    simulator = AssaySimulator()
    topic = settings.kafka_input_topic
    
    logger.info(f"Streaming live assays to topic: '{topic}' at 1 event per {rate_seconds}s")
    sent = 0

    try:
        while True:
            event = simulator.generate_event()
            key = event["assay_id"]

            producer.send(topic, key=key, value=event)
            producer.flush()

            sent += 1
            logger.info(
                f"[{sent:04d}] Emitted {event['assay_id']} | "
                f"Protein: {event['protein']['uniprot_id']} | "
                f"Drug: {event['drug']['chembl_id']} | "
                f"Inhib: {event['label']['inhibition_percentage']}%"
            )

            if 0 < max_messages <= sent:
                logger.info(f"Reached message limit ({max_messages}). Stopping.")
                break

            time.sleep(rate_seconds)

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Closing producer gracefully...")
    finally:
        producer.close()
        logger.info("Kafka producer closed.")

if __name__ == "__main__":
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
    run_stream(rate_seconds=interval)