import pytest
import torch
from src.ml.model import AggregationMultimodalFusionNet

def test_fusion_net_forward_shapes():
    """Verifies output dimensions for a multi-sample batch."""
    batch_size = 4
    fp_dim = 1024
    bio_dim = 6

    model = AggregationMultimodalFusionNet(fp_dim=fp_dim, bio_dim=bio_dim, latent_dim=128)
    model.eval()

    dummy_fp = torch.randint(0, 2, (batch_size, fp_dim)).float()
    dummy_bio = torch.randn(batch_size, bio_dim).float()

    pred_inhibition, class_logits = model(dummy_fp, dummy_bio)

    # Validate output tensor shapes
    assert pred_inhibition.shape == (batch_size,)
    assert class_logits.shape == (batch_size,)

    # Validate regression bounds (scaled by 100 via Sigmoid)
    assert torch.all(pred_inhibition >= 0.0)
    assert torch.all(pred_inhibition <= 100.0)

def test_fusion_net_backward_gradients():
    """Ensures gradients propagate back through both the chemical and biophysical trunks."""
    batch_size = 2
    model = AggregationMultimodalFusionNet(fp_dim=1024, bio_dim=6, latent_dim=128)
    model.train()

    dummy_fp = torch.randint(0, 2, (batch_size, 1024)).float()
    dummy_bio = torch.randn(batch_size, 6).float()

    target_reg = torch.tensor([85.0, 15.0], dtype=torch.float32)
    target_cls = torch.tensor([1.0, 0.0], dtype=torch.float32)

    pred_reg, pred_logits = model(dummy_fp, dummy_bio)

    loss_fn_reg = torch.nn.HuberLoss()
    loss_fn_cls = torch.nn.BCEWithLogitsLoss()

    loss = loss_fn_reg(pred_reg, target_reg) + loss_fn_cls(pred_logits, target_cls)
    loss.backward()

    # Check that weights across both encoders received valid gradients
    for param in model.chem_encoder.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    for param in model.bio_encoder.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()