from types import SimpleNamespace

import pytest
import torch

from core.registry import get_registry, build_model

_SMALL_ARGS = SimpleNamespace(
    vocab_size=32, num_steps=2,
    n_channels=8, n_fourier_basis=4,
    logit_softcap=30.0, decay_init=3.0, activation="gelu",
    num_heads=2, num_kv_heads=2, rope_base=10000.0, qk_gain_init=1.5,
    kernel_size=4,
    unique_steps=2, invocations_per_step=1, n_heads=2, transform_rank=4,
    band_split="1,1,2", slow_decay_init=4.0, fast_decay_init=2.0,
    n_ops=4, interaction_rank=8,
    state_dim=8, inner_dim=16,
    k_active=16, inner_mul=2, parallel_waves=True, grad_checkpoint=False,
    gumbel_tau=1.0, halt_threshold=0.5, ponder_lambda=0.01,
)

BATCH, SEQ = 2, 4


@pytest.mark.parametrize("version", list(get_registry().keys()))
def test_forward_returns_scalar_loss(version):
    """Each model forward(input_ids, target_ids) returns a scalar loss."""
    model = build_model(version, _SMALL_ARGS).float()
    input_ids = torch.randint(0, _SMALL_ARGS.vocab_size, (BATCH, SEQ))
    target_ids = torch.randint(0, _SMALL_ARGS.vocab_size, (BATCH, SEQ))

    loss = model(input_ids, target_ids)

    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert loss.item() > 0, "Loss should be positive"
    assert torch.isfinite(loss), "Loss should be finite"


@pytest.mark.parametrize("version", list(get_registry().keys()))
def test_backward(version):
    """Each model supports backward pass."""
    model = build_model(version, _SMALL_ARGS).float()
    input_ids = torch.randint(0, _SMALL_ARGS.vocab_size, (BATCH, SEQ))
    target_ids = torch.randint(0, _SMALL_ARGS.vocab_size, (BATCH, SEQ))

    loss = model(input_ids, target_ids)
    loss.backward()

    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "At least one parameter should have a gradient"
