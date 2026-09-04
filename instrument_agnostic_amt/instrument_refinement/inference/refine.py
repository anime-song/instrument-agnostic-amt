"""ステム音声と AMT の MIDI から、各ノートの楽器を推論し直す推論本体。

大まかな流れ:

1. MIDI をノート表に変換し、音声を窓（既定 8 秒 / 4 秒ストライド）に切る。
2. 窓ごとにモデルを流し、ノート単位のロジットと音色埋め込みを得る。
   窓は重なるので、同じノートが複数の窓で観測される -> 平均して安定させる。
3. 音色埋め込みでノートをクラスタリングし、クラスタ単位で楽器を 1 つ決める。
   ノート単位で決めると同じ楽器の中で判定が揺れるため、まとめて多数決する。
4. 決めた楽器で MIDI を書き直す（時刻・音高はそのまま、program と名前だけ変更）。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi
import torch
from tqdm.auto import tqdm

from ...amt.inference.audio import load_audio
from ...amt.inference.instruments import resolve_stem_instrument_class_ids
from ...runtime import (
    copy_tensors_to_cpu_once,
    is_amp_supported,
    resolve_amp_dtype,
    resolve_device,
)
from ...taxonomy.instrument_classes import (
    INSTRUMENT_CLASSES,
    get_instrument_class_id_by_name,
    get_program_number_from_class_id,
)
from ..data.labels import (
    inference_stem_group,
    refinement_stem_extra_class_ids,
    stem_context_id,
)
from ..data.midi import RefinementNoteTable, load_refinement_note_table
from ..modeling.checkpoints import load_refinement_model
from ..modeling.model import InstrumentRefinementConfig, InstrumentRefinementModel
from .aggregation import (
    cluster_note_embeddings,
    top_candidates,
    viterbi_smooth_classes,
)


def _mask_logits(values: np.ndarray, allowed: tuple[int, ...]) -> np.ndarray:
    """候補外のクラスを -1e9 で潰し、allowed の中だけで比較できるようにする。"""
    output = np.full_like(values, -1e9, dtype=np.float64)
    output[..., list(allowed)] = values[..., list(allowed)]
    return output


def _window_starts(total_frames: int, window_frames: int, stride_frames: int) -> list[int]:
    """窓の開始サンプル位置を並べる。末尾が余る場合は末尾ぴったりの窓を足す。"""
    if total_frames <= window_frames:
        return [0]
    starts = list(range(0, total_frames - window_frames + 1, stride_frames))
    last_start = total_frames - window_frames
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _allowed_class_ids(stem_name: str | None, *, use_stem_mask: bool) -> tuple[int, ...]:
    """このステムで許す楽器クラスを決める。ドラムは常に除外する。

    例: stem_name="other" -> strings や organ など + 分離器が other に混ぜがちな
    acoustic_guitar（refinement_stem_extra_class_ids で補う）。
    """
    drum_id = get_instrument_class_id_by_name("drums")
    stem_class_ids = (
        resolve_stem_instrument_class_ids(inference_stem_group(stem_name)) if use_stem_mask and stem_name else None
    )
    candidate_ids = stem_class_ids or tuple(range(len(INSTRUMENT_CLASSES)))
    if stem_class_ids is not None:
        candidate_ids = tuple(dict.fromkeys((*candidate_ids, *refinement_stem_extra_class_ids(stem_name))))
    allowed = tuple(class_id for class_id in candidate_ids if int(class_id) != int(drum_id))
    if not allowed:
        # ドラムステムを渡すとここに来る（候補がドラムしかないため）。
        raise ValueError("No non-drum instrument candidates remain")
    return allowed


def _resolve_model(
    checkpoint_path: str | Path | None,
    preloaded_model: InstrumentRefinementModel | None,
    preloaded_config: InstrumentRefinementConfig | None,
    *,
    device: torch.device,
) -> tuple[InstrumentRefinementModel, InstrumentRefinementConfig]:
    """チェックポイントから読むか、渡された読み込み済みモデルを使う。

    ステムごとに何度も呼ぶ用途では、毎回読み直さずに使い回せるようにしてある。
    """
    if (preloaded_model is None) != (preloaded_config is None):
        raise ValueError("preloaded_model and preloaded_config must be provided together")
    if preloaded_model is None or preloaded_config is None:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required when no model is preloaded")
        model, config, _ = load_refinement_model(checkpoint_path, device=device)
        return model, config
    preloaded_model.to(device)
    preloaded_model.eval()
    return preloaded_model, preloaded_config


@dataclass(frozen=True)
class _WindowScan:
    """窓ごとの推論結果と、それをノート単位へ集約したもの。"""

    starts: list[int]
    window_logits: np.ndarray  # [窓数, クラス数]
    window_note_counts: list[int]
    global_logits: np.ndarray  # [クラス数] 曲全体の傾向
    note_logits: np.ndarray  # [ノート数, クラス数]
    note_embeddings: np.ndarray  # [ノート数, 埋め込み次元]


def _scan_windows(
    model: InstrumentRefinementModel,
    config: InstrumentRefinementConfig,
    waveform: torch.Tensor,
    notes: RefinementNoteTable,
    *,
    stem_name: str | None,
    allowed: tuple[int, ...],
    window_seconds: float,
    stride_seconds: float,
    window_batch_size: int,
    device: torch.device,
    disable_tqdm: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> _WindowScan:
    """音声を窓に切ってモデルを流し、窓をまたいで平均する。

    窓は重なるので同じノートが複数回観測される。合計と観測回数を持っておき、
    最後に割って平均にする。どの窓にも入らなかったノートは曲全体の傾向で代用する。
    """
    total_frames = int(waveform.shape[-1])
    window_frames = int(round(window_seconds * config.sample_rate))
    stride_frames = int(round(stride_seconds * config.sample_rate))
    starts = _window_starts(total_frames, window_frames, stride_frames)

    scanned_starts: list[int] = []
    window_logits: list[np.ndarray] = []
    window_embeddings: list[np.ndarray] = []
    window_note_counts: list[int] = []
    note_logit_sums = np.zeros((notes.note_count, config.num_instrument_classes), dtype=np.float64)
    note_embedding_sums = np.zeros((notes.note_count, config.embedding_size), dtype=np.float64)
    note_observations = np.zeros(notes.note_count, dtype=np.int64)
    context_id = stem_context_id(inference_stem_group(stem_name))
    if window_batch_size <= 0:
        raise ValueError("window_batch_size must be positive")

    start_batches = [
        starts[offset : offset + int(window_batch_size)] for offset in range(0, len(starts), int(window_batch_size))
    ]
    for batch_starts in tqdm(
        start_batches,
        desc="refine instruments",
        disable=disable_tqdm,
    ):
        windows: list[torch.Tensor] = []
        valid_frames_batch: list[int] = []
        indices_batch: list[np.ndarray] = []
        relative_starts: list[torch.Tensor] = []
        relative_ends: list[torch.Tensor] = []
        pitches: list[torch.Tensor] = []
        for sample_start in batch_starts:
            valid_frames = min(window_frames, total_frames - sample_start)
            window = waveform[:, sample_start : sample_start + window_frames]
            if int(window.shape[-1]) < window_frames:
                # 末尾の窓は 0 埋めし、有効長はモデルへ伝える。
                window = torch.nn.functional.pad(
                    window,
                    (0, window_frames - int(window.shape[-1])),
                )
            start_seconds = float(sample_start) / float(config.sample_rate)
            end_seconds = start_seconds + window_seconds
            active = (notes.start_seconds >= start_seconds) & (notes.start_seconds < end_seconds)
            indices = np.flatnonzero(active)
            windows.append(window)
            valid_frames_batch.append(valid_frames)
            indices_batch.append(indices)
            relative_starts.append(torch.from_numpy(notes.start_seconds[indices] - start_seconds))
            relative_ends.append(torch.from_numpy(notes.end_seconds[indices] - start_seconds))
            pitches.append(torch.from_numpy(notes.pitch[indices]))

        if not any(len(indices) for indices in indices_batch):
            continue

        note_start_batch = torch.nn.utils.rnn.pad_sequence(
            relative_starts,
            batch_first=True,
        )
        note_end_batch = torch.nn.utils.rnn.pad_sequence(
            relative_ends,
            batch_first=True,
        )
        note_pitch_batch = torch.nn.utils.rnn.pad_sequence(
            pitches,
            batch_first=True,
        )
        max_notes = int(note_pitch_batch.shape[1])
        note_mask = torch.zeros(
            len(batch_starts),
            max_notes,
            dtype=torch.bool,
        )
        for batch_index, indices in enumerate(indices_batch):
            note_mask[batch_index, : len(indices)] = True

        # backboneのCQT/convとTransformerが窓ごとの支配項なので、AMTと同じ
        # autocastをかける。集約はこの直後にfloat32へ戻してから行う。
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            outputs = model(
                torch.stack(windows).to(device),
                valid_audio_frames=torch.tensor(
                    valid_frames_batch,
                    device=device,
                ),
                note_start_seconds=note_start_batch.to(device),
                note_end_seconds=note_end_batch.to(device),
                note_pitch=note_pitch_batch.to(device),
                # AMT が付けた楽器は意図的に渡さない（-1）。音声から判断させるため。
                note_prior_class=torch.full(
                    (len(batch_starts), max_notes),
                    -1,
                    dtype=torch.long,
                    device=device,
                ),
                note_confidence=torch.ones(
                    len(batch_starts),
                    max_notes,
                    device=device,
                ),
                note_mask=note_mask.to(device),
                stem_context_id=torch.full(
                    (len(batch_starts),),
                    context_id,
                    dtype=torch.long,
                    device=device,
                ),
                window_seconds=window_seconds,
            )
        (
            window_logits_tensor,
            window_embeddings_tensor,
            note_logits_tensor,
            note_embeddings_tensor,
        ) = copy_tensors_to_cpu_once(
            (
                outputs["window_logits"].float(),
                outputs["window_embedding"].float(),
                outputs["note_logits"].float(),
                outputs["note_embedding"].float(),
            )
        )
        del outputs
        window_logits_batch = window_logits_tensor.numpy()
        window_embeddings_batch = window_embeddings_tensor.numpy()
        note_logits_batch = note_logits_tensor.numpy()
        note_embeddings_batch = note_embeddings_tensor.numpy()
        for batch_index, indices in enumerate(indices_batch):
            scanned_starts.append(batch_starts[batch_index])
            window_logits.append(_mask_logits(window_logits_batch[batch_index], allowed))
            window_embeddings.append(window_embeddings_batch[batch_index])
            window_note_counts.append(len(indices))
            if len(indices):
                note_logit_sums[indices] += _mask_logits(
                    note_logits_batch[batch_index, : len(indices)],
                    allowed,
                )
                note_embedding_sums[indices] += note_embeddings_batch[
                    batch_index,
                    : len(indices),
                ]
                note_observations[indices] += 1

    # 曲全体の傾向は、ノートが多い窓ほど重く見る。
    if not window_logits:
        return _WindowScan(
            starts=[],
            window_logits=np.zeros(
                (0, config.num_instrument_classes),
                dtype=np.float64,
            ),
            window_note_counts=[],
            global_logits=np.zeros(config.num_instrument_classes, dtype=np.float64),
            note_logits=note_logit_sums,
            note_embeddings=note_embedding_sums,
        )

    stacked = np.stack(window_logits)
    weights = np.asarray(window_note_counts, dtype=np.float64) + 1.0
    global_logits = np.average(stacked, axis=0, weights=weights)
    counts = note_observations.clip(min=1)[:, None]
    note_logits = note_logit_sums / counts
    note_embeddings = note_embedding_sums / counts
    missing = note_observations == 0
    if np.any(missing):
        note_logits[missing] = global_logits
        note_embeddings[missing] = np.average(np.stack(window_embeddings), axis=0, weights=weights)
    return _WindowScan(
        starts=scanned_starts,
        window_logits=stacked,
        window_note_counts=window_note_counts,
        global_logits=global_logits,
        note_logits=note_logits,
        note_embeddings=note_embeddings,
    )


def _cluster_reports(
    notes: RefinementNoteTable,
    scan: _WindowScan,
    cluster_ids: np.ndarray,
    *,
    allowed: tuple[int, ...],
    global_logit_weight: float,
    top_k: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """クラスタごとに楽器を 1 つ決め、所属ノート全部に同じ楽器を割り当てる。"""
    predicted_class_ids = np.zeros(notes.note_count, dtype=np.int64)
    clusters: list[dict[str, Any]] = []
    for cluster_id in sorted(set(cluster_ids.tolist())):
        members = np.flatnonzero(cluster_ids == cluster_id)
        cluster_logits = scan.note_logits[members].mean(axis=0) + float(global_logit_weight) * scan.global_logits
        candidates = top_candidates(cluster_logits, allowed_class_ids=allowed, top_k=top_k)
        predicted_class_ids[members] = candidates[0][0]
        # レポート用にクラスタの代表音色（正規化した重心）を残す。
        centroid = scan.note_embeddings[members].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 1e-8:
            centroid /= norm
        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "note_count": int(len(members)),
                "start_seconds": float(notes.start_seconds[members].min()),
                "end_seconds": float(notes.end_seconds[members].max()),
                "candidates": _candidate_dicts(candidates),
                "embedding": centroid.astype(float).tolist(),
                "note_indices": members.astype(int).tolist(),
            }
        )
    return predicted_class_ids, clusters


def _candidate_dicts(candidates: list[tuple[int, float]]) -> list[dict[str, Any]]:
    return [
        {"class_id": class_id, "class_name": INSTRUMENT_CLASSES[class_id], "probability": probability}
        for class_id, probability in candidates
    ]


def _build_report(
    audio_file: Path,
    midi_file: Path,
    notes: RefinementNoteTable,
    scan: _WindowScan,
    clusters: list[dict[str, Any]],
    cluster_ids: np.ndarray,
    predicted_class_ids: np.ndarray,
    *,
    config: InstrumentRefinementConfig,
    stem_name: str | None,
    mode: str,
    allowed: tuple[int, ...],
    global_logit_weight: float,
    switch_penalty: float,
    top_k: int,
) -> dict[str, Any]:
    """予測結果とデバッグ情報を 1 つの dict にまとめる。

    window_flip_count_* は平滑化の前後で窓の楽器が何回入れ替わったかで、値が大きいほど
    判定が不安定（音が薄い / 候補が紛らわしい）ことを示す。平滑化した系列はここでしか
    使わず、ノートの楽器決定には影響しない。
    """
    smoothed = viterbi_smooth_classes(scan.window_logits, switch_penalty=switch_penalty)
    return {
        "audio_path": str(audio_file),
        "midi_path": str(midi_file),
        "stem_name": stem_name,
        "mode": mode,
        "global_logit_weight": float(global_logit_weight),
        "note_count": notes.note_count,
        "window_count": len(scan.starts),
        "cluster_count": len(clusters),
        "window_flip_count_before_smoothing": int(
            np.count_nonzero(np.diff(np.asarray(scan.window_logits.argmax(axis=1), dtype=np.int64)))
        ),
        "window_flip_count_after_smoothing": int(np.count_nonzero(np.diff(smoothed))),
        "global_candidates": _candidate_dicts(
            top_candidates(scan.global_logits, allowed_class_ids=allowed, top_k=top_k)
        ),
        "windows": [
            {
                "start_seconds": float(start) / float(config.sample_rate),
                "note_count": int(note_count),
                "smoothed_class_id": int(class_id),
                "smoothed_class_name": INSTRUMENT_CLASSES[int(class_id)],
            }
            for start, note_count, class_id in zip(scan.starts, scan.window_note_counts, smoothed)
        ],
        "clusters": clusters,
        "notes": [
            {
                "note_index": index,
                "start_seconds": float(notes.start_seconds[index]),
                "end_seconds": float(notes.end_seconds[index]),
                "pitch": int(notes.pitch[index]),
                "cluster_id": int(cluster_ids[index]),
                "class_id": int(predicted_class_ids[index]),
                "class_name": INSTRUMENT_CLASSES[int(predicted_class_ids[index])],
            }
            for index in range(notes.note_count)
        ],
    }


def _rewrite_midi(
    input_path: Path,
    output_path: Path,
    *,
    table: RefinementNoteTable,
    predicted_class_ids: np.ndarray,
) -> None:
    """元 MIDI を読み直し、ノートを予測クラスごとのトラックに割り振って保存する。

    ノート自体（時刻・音高・velocity）は触らず、所属トラックの program と名前だけが
    変わる。1 つのトラックから複数の楽器が出た場合はトラックが分割される。
    """
    midi = pretty_midi.PrettyMIDI(str(input_path))
    # ノート表は (トラック番号, トラック内のノート番号) を保持しているので、
    # それをキーに「このノートは何の楽器か」を引けるようにする。
    assignments = {
        (int(instrument_index), int(note_index)): int(class_id)
        for instrument_index, note_index, class_id in zip(table.instrument_index, table.note_index, predicted_class_ids)
    }
    rewritten: list[pretty_midi.Instrument] = []
    for instrument_index, instrument in enumerate(midi.instruments):
        # ドラムはこのモデルの対象外なのでそのまま残す。
        if instrument.is_drum or not instrument.notes:
            rewritten.append(instrument)
            continue
        grouped: dict[int, list[pretty_midi.Note]] = {}
        # ノート表に載らなかったノート（想定外だが念のため）は piano に寄せる。
        fallback = get_instrument_class_id_by_name("piano")
        for note_index, note in enumerate(instrument.notes):
            class_id = assignments.get((instrument_index, note_index), fallback)
            grouped.setdefault(class_id, []).append(note)
        for class_id, notes in sorted(grouped.items()):
            refined = copy.deepcopy(instrument)
            refined.notes = notes
            refined.program = get_program_number_from_class_id(class_id)
            refined.name = INSTRUMENT_CLASSES[class_id]
            rewritten.append(refined)
    midi.instruments = rewritten
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))


def _write_outputs(
    report: dict[str, Any],
    midi_file: Path,
    notes: RefinementNoteTable,
    predicted_class_ids: np.ndarray,
    *,
    output_midi_path: str | Path | None,
    output_json_path: str | Path | None,
) -> None:
    """指定されていれば書き出し、書いた先を report にも記録する。"""
    if output_midi_path is not None:
        destination = Path(output_midi_path)
        _rewrite_midi(midi_file, destination, table=notes, predicted_class_ids=predicted_class_ids)
        report["output_midi_path"] = str(destination.resolve())
    if output_json_path is not None:
        json_path = Path(output_json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        report["output_json_path"] = str(json_path.resolve())
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)


@torch.inference_mode()
def refine_midi_instruments(
    audio_path: str | Path,
    midi_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    output_midi_path: str | Path | None = None,
    output_json_path: str | Path | None = None,
    stem_name: str | None = None,
    device: torch.device | str | None = None,
    window_seconds: float = 8.0,
    stride_seconds: float = 4.0,
    window_batch_size: int = 1,
    mode: str = "cluster",
    cluster_distance: float = 0.25,
    min_cluster_notes: int = 2,
    switch_penalty: float = 2.0,
    global_logit_weight: float = 0.0,
    top_k: int = 3,
    use_stem_mask: bool = True,
    disable_tqdm: bool = False,
    preloaded_model: InstrumentRefinementModel | None = None,
    preloaded_config: InstrumentRefinementConfig | None = None,
    preloaded_waveform: torch.Tensor | None = None,
    amp: bool = False,
    amp_dtype: str | torch.dtype | None = None,
) -> dict[str, Any]:
    """MIDI の楽器を、対応するステム音声から推論し直す。

    Args:
        audio_path: 分離済みステムの音声。midi_path はこの音声を採譜したものであること。
        midi_path: AMT が出力した MIDI。ノートの時刻・音高はそのまま使う。
        stem_name: "vocals" / "bass" / "guitar" / "other" など。楽器候補の絞り込みに使う。
        mode: "cluster" は音色でノートをまとめてから楽器を決める（既定）。
            "single" はステム全体を 1 つの楽器とみなす。
        cluster_distance: 音色クラスタリングのしきい値。小さいほど細かく分かれる。
        switch_penalty: 窓ごとの楽器がころころ変わるのを抑える Viterbi のペナルティ。
            結果のレポート用途のみで、ノートの楽器決定には使わない。
        global_logit_weight: 曲全体の傾向をクラスタ判定に足す量。0 なら足さない。
        use_stem_mask: stem_name に基づく候補の絞り込みを行うか。
        preloaded_model/preloaded_config: 読み込み済みモデルを使い回す場合に両方渡す。
        preloaded_waveform: config.sample_rateへresample済みのCPU float32ステレオ波形。
        amp: 窓ごとの backbone を autocast で走らせるか。CUDA / MPS でのみ有効。
        amp_dtype: "fp16" / "bf16" / torch.dtype。省略時はデバイス既定を使う。

    Returns:
        予測結果とデバッグ情報を含む dict。output_json_path を渡すと同じ内容を保存する。
    """
    if mode not in {"single", "cluster"}:
        raise ValueError("mode must be 'single' or 'cluster'")
    if window_seconds <= 0.0 or stride_seconds <= 0.0:
        raise ValueError("window and stride seconds must be positive")
    if window_batch_size <= 0:
        raise ValueError("window_batch_size must be positive")
    if global_logit_weight < 0.0:
        raise ValueError("global_logit_weight must be nonnegative")
    audio_file = Path(audio_path).resolve()
    midi_file = Path(midi_path).resolve()
    if not audio_file.is_file() or not midi_file.is_file():
        raise FileNotFoundError("audio and MIDI paths must both exist")
    target_device = resolve_device(device)
    amp_enabled = bool(amp and is_amp_supported(target_device))
    resolved_amp_dtype = (
        amp_dtype
        if isinstance(amp_dtype, torch.dtype)
        else resolve_amp_dtype(target_device, amp_dtype)
    )

    # 1. モデルを用意する。ステムごとに呼ばれる用途では読み込み済みモデルを受け取る。
    model, config = _resolve_model(checkpoint_path, preloaded_model, preloaded_config, device=target_device)

    # 2. 音声と MIDI を読み、楽器の候補集合を決める。
    if preloaded_waveform is None:
        waveform = load_audio(audio_file, target_sample_rate=config.sample_rate)
    else:
        waveform = preloaded_waveform
        if waveform.device.type != "cpu":
            raise ValueError("preloaded_waveform must be on CPU")
        if waveform.dtype != torch.float32:
            raise ValueError("preloaded_waveform must have dtype float32")
        if waveform.ndim != 2 or int(waveform.shape[0]) != 2:
            raise ValueError("preloaded_waveform must have shape [2, T]")
    notes = load_refinement_note_table(midi_file)
    allowed = _allowed_class_ids(stem_name, use_stem_mask=use_stem_mask)

    # 3. 窓ごとに推論し、窓をまたいで集約する。
    scan = _scan_windows(
        model,
        config,
        waveform,
        notes,
        stem_name=stem_name,
        allowed=allowed,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        window_batch_size=int(window_batch_size),
        device=target_device,
        disable_tqdm=disable_tqdm,
        amp_enabled=amp_enabled,
        amp_dtype=resolved_amp_dtype,
    )

    # 4. 音色埋め込みでノートをまとめる。single モードでは全ノートを 1 クラスタ扱い。
    if mode == "single" or notes.note_count == 0:
        cluster_ids = np.zeros(notes.note_count, dtype=np.int64)
    else:
        cluster_ids = cluster_note_embeddings(
            scan.note_embeddings,
            distance_threshold=cluster_distance,
            min_cluster_notes=min_cluster_notes,
        )

    # 5. クラスタごとに楽器を決める。
    predicted_class_ids, clusters = _cluster_reports(
        notes,
        scan,
        cluster_ids,
        allowed=allowed,
        global_logit_weight=global_logit_weight,
        top_k=top_k,
    )

    # 6. 結果をまとめる。
    report = _build_report(
        audio_file,
        midi_file,
        notes,
        scan,
        clusters,
        cluster_ids,
        predicted_class_ids,
        config=config,
        stem_name=stem_name,
        mode=mode,
        allowed=allowed,
        global_logit_weight=global_logit_weight,
        switch_penalty=switch_penalty,
        top_k=top_k,
    )

    # 7. 出力先が指定されていれば MIDI / JSON を書き出す。
    _write_outputs(
        report,
        midi_file,
        notes,
        predicted_class_ids,
        output_midi_path=output_midi_path,
        output_json_path=output_json_path,
    )
    return report
