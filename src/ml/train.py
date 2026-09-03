import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import mlflow
import mlflow.pytorch
from src.common.config import settings
from src.common.logger import get_logger
from src.ml.model import AggregationMultimodalFusionNet
from src.ml.dataset import AggregationDataset
from src.ml.evaluate import evaluate_model

logger = get_logger("ModelTrainer")

def train_pipeline(epochs: int = 15, batch_size: int = 8, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Beginning ML training pipeline on device: {device}")

    # Set up MLflow tracking
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("protein-aggregation-inhibitor-screen")

    # Load dataset
    dataset = AggregationDataset()
    if len(dataset) < 4:
        logger.warn("Dataset too small for multi-split validation. Re-run fetch_real_data.py for more records.")

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = AggregationMultimodalFusionNet(fp_dim=1024, bio_dim=6, latent_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Loss Functions: Huber loss for continuous % inhibition, BCE for classification
    reg_criterion = nn.HuberLoss()
    cls_criterion = nn.BCEWithLogitsLoss()

    with mlflow.start_run(run_name="multimodal-fusion-v1") as run:
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "model_architecture": "MultimodalFusionNet",
            "fingerprint_bits": 1024
        })

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0

            for fps, bios, y_reg, y_cls in train_loader:
                fps = fps.to(device)
                bios = bios.to(device)
                y_reg = y_reg.to(device)
                y_cls = y_cls.to(device)

                optimizer.zero_grad()
                pred_reg, pred_logits = model(fps, bios)

                loss_reg = reg_criterion(pred_reg, y_reg)
                loss_cls = cls_criterion(pred_logits, y_cls)
                total_loss = loss_reg + (10.0 * loss_cls)

                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()

            train_loss = running_loss / len(train_loader)
            val_metrics = evaluate_model(model, val_loader, device)

            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
                f"Val RMSE: {val_metrics['val_rmse']:.2f}% | Val Acc: {val_metrics['val_accuracy']:.2f}"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(k, v, step=epoch)

        # Save local weights artifact
        os.makedirs("models", exist_ok=True)
        model_save_path = os.path.join("models", "fusion_net.pt")
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Model saved locally at: {model_save_path}")

        # Register artifact with MLflow
        mlflow.pytorch.log_model(model, "model")
        logger.info(f"Logged run to MLflow [Run ID: {run.info.run_id}]")

if __name__ == "__main__":
    train_pipeline()