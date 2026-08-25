from __future__ import annotations

import os

import torch

from instrument_agnostic_amt.cli.train import (
    compile_backbone_transformers,
    drop_unsearchable_path_entries,
)
from instrument_agnostic_amt.modeling.model import (
    AudioSemiCRFTransformer,
    SemiCRFModelConfig,
)


def _tiny_model() -> AudioSemiCRFTransformer:
    return AudioSemiCRFTransformer(
        SemiCRFModelConfig(
            sample_rate=8_000,
            hop_length=128,
            n_fft=256,
            cqt_n_bins=48,
            cqt_bins_per_octave=12,
            harmonics=(1.0,),
            hidden_size=16,
            base_ch=4,
            encoder_num_layers=1,
            encoder_num_heads=1,
            dropout=0.0,
            semi_crf_head_dim=5,
            num_instrument_classes=3,
            use_gradient_checkpoint=False,
        )
    )


def test_drop_unsearchable_path_entries_keeps_real_directories(
    tmp_path, monkeypatch
) -> None:
    real = tmp_path / "bin"
    real.mkdir()
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv(
        "PATH", os.pathsep.join([str(real), str(missing), ""])
    )

    dropped = drop_unsearchable_path_entries()

    assert str(missing) in dropped
    assert os.environ["PATH"].split(os.pathsep) == [str(real)]


def test_drop_unsearchable_path_entries_is_a_noop_when_all_are_valid(
    tmp_path, monkeypatch
) -> None:
    real = tmp_path / "bin"
    real.mkdir()
    monkeypatch.setenv("PATH", str(real))

    assert drop_unsearchable_path_entries() == []
    assert os.environ["PATH"] == str(real)


def test_compiling_transformers_keeps_state_dict_keys() -> None:
    """torch.compile 後もチェックポイントのキーが変わらないことを固定する。"""
    model = _tiny_model()
    before = list(model.state_dict())

    compile_backbone_transformers(model, "default")

    assert list(model.state_dict()) == before
    assert isinstance(model.backbone.layers[0][0], torch.nn.Module)
