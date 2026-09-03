#!/usr/bin/env bash
set -e

MINIO_HOST="http://localhost:9000"
ACCESS_KEY="minioadmin"
SECRET_KEY="minioadmin"
ALIAS="myminio"

echo "[INFO] Configuring MinIO client alias..."
docker run --rm --network host minio/mc:RELEASE.2024-01-16T16-06-34Z \
  alias set "$ALIAS" "$MINIO_HOST" "$ACCESS_KEY" "$SECRET_KEY"

BUCKETS=(
  "feature-store"
  "spark-checkpoints"
  "mlflow-artifacts"
)

for BUCKET in "${BUCKETS[@]}"; do
  echo "[INFO] Ensuring bucket exists: $BUCKET"
  docker run --rm --network host minio/mc:RELEASE.2024-01-16T16-06-34Z \
    mb --ignore-existing "$ALIAS/$BUCKET"
done

echo "[INFO] Setting anonymous download policy for feature-store..."
docker run --rm --network host minio/mc:RELEASE.2024-01-16T16-06-34Z \
  anonymous set download "$ALIAS/feature-store"

echo "[SUCCESS] Buckets configured successfully:"
docker run --rm --network host minio/mc:RELEASE.2024-01-16T16-06-34Z \
  ls "$ALIAS"