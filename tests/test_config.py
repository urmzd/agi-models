import json

import pytest

from core.config import Hyperparameters


def test_defaults():
    hp = Hyperparameters()
    assert hp.model_version == "v8_graph"
    assert hp.vocab_size == 1024
    assert hp.lr == 0.03
    assert hp.seed == 1337


def test_to_dict_json_serializable():
    hp = Hyperparameters()
    d = hp.to_dict()
    assert isinstance(d, dict)
    assert "ModelCommonConfig" in d
    assert "OptimizerConfig" in d
    # Must be JSON-serializable
    s = json.dumps(d, default=str)
    assert len(s) > 0


def test_getattr_raises_on_unknown():
    hp = Hyperparameters()
    with pytest.raises(AttributeError, match="no field"):
        _ = hp.nonexistent_field_xyz


def test_env_override(monkeypatch):
    monkeypatch.setenv("VOCAB_SIZE", "512")
    monkeypatch.setenv("LR", "0.01")
    hp = Hyperparameters()
    assert hp.vocab_size == 512
    assert hp.lr == 0.01
