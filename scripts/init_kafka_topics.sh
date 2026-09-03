#!/usr/bin/env bash
set -e

CONTAINER_NAME="bio_kafka"
BOOTSTRAP_SERVER="localhost:9092"

echo "[INFO] Waiting for Kafka broker ($CONTAINER_NAME) to become ready..."
until docker exec "$CONTAINER_NAME" kafka-topics --bootstrap-server "$BOOTSTRAP_SERVER" --list > /dev/null 2>&1; do
  echo "[WAIT] Kafka is not ready yet. Retrying in 3 seconds..."
  sleep 3
done

echo "[INFO] Kafka is online. Creating required streaming topics..."

TOPICS=(
  "raw-aggregation-events"
  "inference-requests"
  "predictions-out"
)

for TOPIC in "${TOPICS[@]}"; do
  docker exec "$CONTAINER_NAME" kafka-topics \
    --bootstrap-server "$BOOTSTRAP_SERVER" \
    --create \
    --if-not-exists \
    --topic "$TOPIC" \
    --partitions 3 \
    --replication-factor 1
  echo "[CREATED] Topic: $TOPIC"
done

echo "[SUCCESS] Active Kafka topics:"
docker exec "$CONTAINER_NAME" kafka-topics --bootstrap-server "$BOOTSTRAP_SERVER" --list