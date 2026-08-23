from __future__ import annotations

import pytest
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils._python_dispatch import TorchDispatchMode

from instrument_agnostic_amt.inference.types import InferenceSettings
from instrument_agnostic_amt.inference.windowed import _select_pair_candidates


def _settings() -> InferenceSettings:
    return InferenceSettings(
        window_ms=8_000,
        stride_ms=4_000,
        track_batch_size=128,
        window_batch_size=1,
        merge_gap_ms=None,
        merge_onset_ms=50.0,
        silence_gate_rms_dbfs=None,
        note_bias=0.0,
        disable_tqdm=True,
        instrument_pair_infer_topk=10,
        instrument_pair_gate_threshold=1.0,
        instrument_pair_max_pairs=0,
        allowed_instrument_ids=(0, 1),
    )


def _candidate_logits(device: torch.device | str) -> torch.Tensor:
    logits = torch.full((3, 88), float("-inf"))
    logits[0, 0] = 3.0
    logits[0, 1] = 3.0
    logits[0, 2] = 2.0
    logits[0, 3] = float("nan")
    logits[0, 4] = float("inf")
    logits[1, 0] = 1.0
    logits[1, 1] = 0.5
    # 許可対象外の楽器は、最大scoreでも候補に含めない。
    logits[2, 0] = 100.0
    return logits.to(device)


class _DeviceReadRecorder(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.scalar_reads = 0
        self.host_copy_numels: list[int] = []

    def __torch_dispatch__(
        self,
        func: object,
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        call_kwargs = {} if kwargs is None else kwargs
        if "_local_scalar_dense" in str(func):
            self.scalar_reads += sum(
                int(isinstance(value, torch.Tensor) and value.device.type != "cpu")
                for value in args
            )
        if "_to_copy" in str(func) and args and isinstance(args[0], torch.Tensor):
            source = args[0]
            target_device = call_kwargs.get("device")
            if (
                source.device.type != "cpu"
                and target_device is not None
                and torch.device(target_device).type == "cpu"
            ):
                self.host_copy_numels.append(int(source.numel()))
        return func(*args, **call_kwargs)  # type: ignore[operator]


def test_pair_candidates_keep_threshold_topk_tie_and_nonfinite_semantics() -> None:
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        selected = _select_pair_candidates(
            _candidate_logits("cpu"),
            settings=_settings(),
            instrument_filter_id=None,
        )
    scalar_reads = sum(
        event.count
        for event in profiler.key_averages()
        if event.key == "aten::_local_scalar_dense"
    )

    assert (selected, scalar_reads) == ([0, 1, 2, 88, 89], 0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS上の候補選択転送を検査するテストです",
)
def test_pair_candidates_use_one_masked_score_host_transfer() -> None:
    logits = _candidate_logits("mps")
    recorder = _DeviceReadRecorder()

    with recorder:
        selected = _select_pair_candidates(
            logits,
            settings=_settings(),
            instrument_filter_id=None,
        )

    assert (
        selected,
        recorder.scalar_reads,
        recorder.host_copy_numels,
    ) == ([0, 1, 2, 88, 89], 0, [logits.numel()])
