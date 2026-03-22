from types import SimpleNamespace

import pytest
import torch.nn as nn

from core.registry import get_registry, build_model

# Small args for fast instantiation on CPU
_SMALL_ARGS = SimpleNamespace(
    vocab_size=32, num_steps=2,
    n_channels=8, n_fourier_basis=4,
    logit_softcap=30.0, decay_init=3.0, activation="gelu",
    # v1
    num_heads=2, num_kv_heads=2, rope_base=10000.0, qk_gain_init=1.5,
    # v2
    kernel_size=4,
    # v4
    unique_steps=2, invocations_per_step=1, n_heads=2, transform_rank=4,
    # wave
    band_split="1,1,2", slow_decay_init=4.0, fast_decay_init=2.0,
    # lgp / policy
    n_ops=4,
    # graph
    interaction_rank=8,
    # meta / brainwave / tpg
    state_dim=8, inner_dim=16,
    # sparse
    k_active=16, inner_mul=2, parallel_waves=True, grad_checkpoint=False,
    # tpg
    gumbel_tau=1.0, halt_threshold=0.5, ponder_lambda=0.01,
)


@pytest.mark.parametrize("version", list(get_registry().keys()))
def test_registry_import(version):
    """Every REGISTRY key imports and instantiates successfully."""
    model = build_model(version, _SMALL_ARGS)
    assert isinstance(model, nn.Module)


def test_build_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model version"):
        build_model("nonexistent_v99", _SMALL_ARGS)
