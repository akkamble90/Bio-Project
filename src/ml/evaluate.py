import torch
import numpy as np
from typing import Dict
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score

def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Runs zero-grad evaluation over validation/test splits."""
    model.eval()
    
    all_y_true_reg = []
    all_y_pred_reg = []
    all_y_true_cls = []
    all_y_pred_logits = []

    with torch.no_grad():
        for fps, bios, y_reg, y_cls in dataloader:
            fps = fps.to(device)
            bios = bios.to(device)

            pred_reg, pred_logits = model(fps, bios)

            all_y_true_reg.extend(y_reg.numpy())
            all_y_pred_reg.extend(pred_reg.cpu().numpy())
            all_y_true_cls.extend(y_cls.numpy())
            all_y_pred_logits.extend(pred_logits.cpu().numpy())

    y_true_reg = np.array(all_y_true_reg)
    y_pred_reg = np.array(all_y_pred_reg)
    y_true_cls = np.array(all_y_true_cls)
    y_pred_probs = 1.0 / (1.0 + np.exp(-np.array(all_y_pred_logits)))
    y_pred_labels = (y_pred_probs >= 0.5).astype(int)

    # Metrics computation
    rmse = float(np.sqrt(mean_squared_error(y_true_reg, y_pred_reg)))
    mae = float(mean_absolute_error(y_true_reg, y_pred_reg))
    acc = float(accuracy_score(y_true_cls, y_pred_labels))

    # Guard ROC-AUC against single-class mini-batches
    try:
        auc = float(roc_auc_score(y_true_cls, y_pred_probs))
    except Exception:
        auc = 0.5

    return {
        "val_rmse": round(rmse, 4),
        "val_mae": round(mae, 4),
        "val_accuracy": round(acc, 4),
        "val_roc_auc": round(auc, 4)
    }