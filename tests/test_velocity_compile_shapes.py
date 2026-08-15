from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from tests.test_mps_inference import _small_velocity_model, _velocity_inputs


def test_velocity_compile_stabilizes_after_note_count_becomes_dynamic() -> None:
    compile_count = 0

    def counting_backend(
        graph_module: torch.fx.GraphModule,
        _example_inputs: Sequence[object],
    ) -> Callable[..., object]:
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    torch.compiler.reset()
    try:
        model = _small_velocity_model(predict_stem_gain=False)
        compiled_forward = torch.compile(model, backend=counting_backend)
        counts_after_call: list[int] = []

        with torch.inference_mode():
            # 実MIDIで観測した疎密の異なるノート数を同じ固定長窓へ流す。
            for note_count in (48, 184, 545, 96, 2):
                outputs = compiled_forward(**_velocity_inputs(note_count))
                assert outputs["velocity_expected"].shape == (1, note_count)
                counts_after_call.append(compile_count)

        # 最初のshape変更で動的Nのグラフへ移行し、その後は再利用する。
        assert counts_after_call[1] > counts_after_call[0]
        assert counts_after_call[2:] == [counts_after_call[1]] * 3
    finally:
        torch.compiler.reset()
