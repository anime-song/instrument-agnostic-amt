import argparse
import contextlib
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union  # noqa: UP035

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch import nn
from torch.utils.data import Dataset

from ...data.audio import inspect_audio, read_audio_frames
from .meter_grouping import major_grouping_loss


@dataclass(frozen=True)
class BeatConfig:
    downbeat_pos_weight: float = 20.0
    beat_pos_weight: float = 5.0
    meter_loss_weight: float = 0.05
    meter_grid_ranking_loss_weight: float = 0.0
    meter_grid_ranking_margin: float = 0.1
    meter_grid_kl_loss_weight: float = 0.0
    meter_grid_kl_temperature: float = 0.2
    loss_tolerance: int = 1
    beat_phase_loss_weight: float = 1.0
    bar_phase_loss_weight: float = 1.0
    major_grouping_loss_weight: float = 1.0
    major_grouping_accent_loss_weight: float = 0.25
    major_grouping_accent_temperature: float = 0.5

    def __post_init__(self) -> None:
        if self.downbeat_pos_weight <= 0.0:
            raise ValueError("downbeat_pos_weight must be positive")
        if self.beat_pos_weight <= 0.0:
            raise ValueError("beat_pos_weight must be positive")
        if self.meter_loss_weight < 0.0:
            raise ValueError("meter_loss_weight must be non-negative")
        if self.meter_grid_ranking_loss_weight < 0.0:
            raise ValueError("meter_grid_ranking_loss_weight must be non-negative")
        if self.meter_grid_ranking_margin < 0.0:
            raise ValueError("meter_grid_ranking_margin must be non-negative")
        if self.meter_grid_kl_loss_weight < 0.0:
            raise ValueError("meter_grid_kl_loss_weight must be non-negative")
        if self.meter_grid_kl_temperature <= 0.0:
            raise ValueError("meter_grid_kl_temperature must be positive")
        if self.loss_tolerance < 0:
            raise ValueError("loss_tolerance must be non-negative")
        if self.beat_phase_loss_weight < 0.0:
            raise ValueError("beat_phase_loss_weight must be non-negative")
        if self.bar_phase_loss_weight < 0.0:
            raise ValueError("bar_phase_loss_weight must be non-negative")
        if self.major_grouping_loss_weight < 0.0:
            raise ValueError("major_grouping_loss_weight must be non-negative")
        if self.major_grouping_accent_loss_weight < 0.0:
            raise ValueError("major_grouping_accent_loss_weight must be non-negative")
        if self.major_grouping_accent_temperature <= 0.0:
            raise ValueError("major_grouping_accent_temperature must be positive")


def add_beat_training_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Beat training")
    group.add_argument(
        "--beat_dataset_path",
        type=str,
        default="beat_chord_dataset/beat_dataset",
        help="Path to beat dataset root containing audio/ and label/.",
    )
    group.add_argument(
        "--enable_beat_training",
        action="store_true",
        help="Enable auxiliary beat/downbeat/meter training.",
    )
    group.add_argument(
        "--beat_batch_size",
        type=int,
        default=0,
        help="Beat batch size. Uses --batch_size when set to 0.",
    )
    group.add_argument(
        "--beat_num_workers",
        type=int,
        default=-1,
        help="Beat DataLoader workers. Uses --num_workers when set to -1.",
    )
    group.add_argument(
        "--beat_loss_scale",
        type=float,
        default=0.1,
        help="Scale for beat loss to control its impact on the backbone.",
    )
    group.add_argument(
        "--beat_update_interval",
        type=int,
        default=1,
        help="Run beat auxiliary updates every N AMT steps.",
    )
    group.add_argument(
        "--downbeat_pos_weight",
        "--beat_downbeat_pos_weight",
        dest="beat_downbeat_pos_weight",
        type=float,
        default=20.0,
        help="Positive class weight for downbeat BCE loss.",
    )
    group.add_argument(
        "--beat_pos_weight",
        "--beat_beat_pos_weight",
        dest="beat_beat_pos_weight",
        type=float,
        default=5.0,
        help="Positive class weight for beat BCE loss.",
    )
    group.add_argument(
        "--meter_loss_weight",
        "--beat_meter_loss_weight",
        dest="beat_meter_loss_weight",
        type=float,
        default=0.05,
        help="Weight for meter classification loss.",
    )
    group.add_argument(
        "--meter_grid_ranking_loss_weight",
        "--grid_consistency_loss_weight",
        "--beat_grid_consistency_loss_weight",
        dest="beat_meter_grid_ranking_loss_weight",
        type=float,
        default=0.0,
        help="Weight for bar-level ranking between meter-derived beat grids.",
    )
    group.add_argument(
        "--meter_grid_ranking_margin",
        "--beat_meter_grid_ranking_margin",
        dest="beat_meter_grid_ranking_margin",
        type=float,
        default=0.1,
        help="Margin for bar-level meter-grid ranking loss.",
    )
    group.add_argument(
        "--meter_grid_kl_loss_weight",
        "--beat_meter_grid_kl_loss_weight",
        dest="beat_meter_grid_kl_loss_weight",
        type=float,
        default=0.0,
        help="Weight for KL consistency from detached beat-grid distribution to meter logits.",
    )
    group.add_argument(
        "--meter_grid_kl_temperature",
        "--beat_meter_grid_kl_temperature",
        dest="beat_meter_grid_kl_temperature",
        type=float,
        default=0.2,
        help="Softmax temperature for beat-grid distribution used by meter-grid KL loss.",
    )
    group.add_argument(
        "--loss_tolerance",
        "--beat_loss_tolerance",
        dest="beat_loss_tolerance",
        type=int,
        default=1,
        help="Shift tolerance in frames for beat/downbeat BCE loss.",
    )
    group.add_argument(
        "--major_grouping_loss_weight",
        "--beat_major_grouping_loss_weight",
        dest="beat_major_grouping_loss_weight",
        type=float,
        default=1.0,
        help="Weight for fixed/latent major beat-group boundary loss.",
    )
    group.add_argument(
        "--major_grouping_accent_loss_weight",
        "--beat_major_grouping_accent_loss_weight",
        dest="beat_major_grouping_accent_loss_weight",
        type=float,
        default=0.25,
        help="Weight for MIDI-onset accent alignment within latent groupings.",
    )
    group.add_argument(
        "--major_grouping_accent_temperature",
        "--beat_major_grouping_accent_temperature",
        dest="beat_major_grouping_accent_temperature",
        type=float,
        default=0.5,
        help="Soft-target temperature for latent major-group selection.",
    )


def beat_config_from_args(args: Any) -> BeatConfig:
    return BeatConfig(
        downbeat_pos_weight=float(getattr(args, "beat_downbeat_pos_weight", 20.0)),
        beat_pos_weight=float(getattr(args, "beat_beat_pos_weight", 5.0)),
        meter_loss_weight=float(getattr(args, "beat_meter_loss_weight", 0.05)),
        meter_grid_ranking_loss_weight=float(
            getattr(args, "beat_meter_grid_ranking_loss_weight", 0.0)
        ),
        meter_grid_ranking_margin=float(
            getattr(args, "beat_meter_grid_ranking_margin", 0.1)
        ),
        meter_grid_kl_loss_weight=float(
            getattr(args, "beat_meter_grid_kl_loss_weight", 0.0)
        ),
        meter_grid_kl_temperature=float(
            getattr(args, "beat_meter_grid_kl_temperature", 0.2)
        ),
        loss_tolerance=int(getattr(args, "beat_loss_tolerance", 1)),
        major_grouping_loss_weight=float(
            getattr(args, "beat_major_grouping_loss_weight", 1.0)
        ),
        major_grouping_accent_loss_weight=float(
            getattr(args, "beat_major_grouping_accent_loss_weight", 0.25)
        ),
        major_grouping_accent_temperature=float(
            getattr(args, "beat_major_grouping_accent_temperature", 0.5)
        ),
    )


def beat_dataset_has_wav_audio(root: str | Path) -> bool:
    audio_dir = Path(root) / "audio"
    return audio_dir.exists() and any(audio_dir.glob("*.wav"))


class BeatDataset(Dataset):
    """
    beat_chord_dataset/beat_dataset を読む Dataset。

    1. wav と JSON ラベルを対応付ける。
    2. downbeat と拍子から beat/downbeat/meter の時刻表現を作る。
    3. 学習時にランダム窓を切り出してフレーム単位ターゲットへ変換する。
    """

    def __init__(
        self,
        root: str | Path,
        *,
        window_ms: int,
        sample_rate: int,
        hop_length: int,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.audio_dir = self.root / "audio"
        self.label_dir = self.root / "label"
        self.window_ms = int(window_ms)
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.seed = int(seed)
        self.epoch = 0

        if self.window_ms <= 0:
            raise ValueError("window_ms must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if not self.audio_dir.exists() or not self.label_dir.exists():
            raise FileNotFoundError(
                f"Beat dataset must contain audio/ and label/: {self.root}"
            )

        # 1. wav と JSON ラベルを stem 名で対応付ける。
        #    label/song.beat.beats.json に対して audio/song.wav を探す。
        label_suffix = ".beat.beats.json"
        audio_by_stem = {
            path.stem: path for path in self.audio_dir.glob("*.wav") if path.is_file()
        }

        raw_items: List[Dict[str, Any]] = []
        meter_keys: set[Tuple[int, int]] = set()
        for label_path in sorted(self.label_dir.glob(f"*{label_suffix}")):
            stem = label_path.name[: -len(label_suffix)]
            audio_path = audio_by_stem.get(stem)
            if audio_path is None:
                continue

            with open(label_path, "r", encoding="utf-8") as f:
                label_data = json.load(f)

            measures: List[Dict[str, float | int]] = []
            for raw_measure in label_data.get("measures", []):
                meter_num = int(raw_measure["time_sig_num"])
                meter_den = int(raw_measure["time_sig_den"])
                if meter_num <= 0 or meter_den <= 0:
                    continue
                measures.append(
                    {
                        "downbeat_sec": float(raw_measure["downbeat_sec"]),
                        "meter_num": meter_num,
                        "meter_den": meter_den,
                        "tempo_bpm": float(raw_measure.get("tempo_bpm", 0.0)),
                    }
                )
                meter_keys.add((meter_num, meter_den))

            measures.sort(key=lambda measure: float(measure["downbeat_sec"]))
            if not measures:
                continue

            raw_items.append(
                {
                    "song_name": stem,
                    "audio_path": audio_path,
                    "measures": measures,
                }
            )

        # 2. データセット全体に出てくる拍子を class index に変換する。
        #    meter は分類問題として扱うので、(4, 4), (6, 8) などを固定順に並べる。
        self.meter_classes: Tuple[Tuple[int, int], ...] = tuple(sorted(meter_keys))
        self.meter_to_index: Dict[Tuple[int, int], int] = {
            meter: index for index, meter in enumerate(self.meter_classes)
        }
        self.num_meter_classes = len(self.meter_classes)
        if self.num_meter_classes == 0 or not raw_items:
            raise ValueError(f"No usable wav beat samples found in {self.root}")

        self.window_frames = int(round(self.window_ms * self.sample_rate / 1000.0))
        self.window_sec = float(self.window_frames) / float(self.sample_rate)
        self.model_frames = math.ceil(self.window_frames / self.hop_length)

        # 3. downbeat だけのラベルから beat 時刻と meter 区間を作る。
        #    同時に BalancedSoftmaxLoss 用の meter 出現回数をフレーム単位で数える。
        items: List[Dict[str, Any]] = []
        meter_counts = torch.zeros(self.num_meter_classes, dtype=torch.float32)
        for raw_item in raw_items:
            beat_times: List[float] = []
            downbeat_times: List[float] = []
            meter_intervals: List[Tuple[float, float, int]] = []
            measures = raw_item["measures"]
            for index, measure in enumerate(measures):
                start_sec = float(measure["downbeat_sec"])
                meter_num = int(measure["meter_num"])
                meter_den = int(measure["meter_den"])
                tempo_bpm = float(measure["tempo_bpm"])

                if index + 1 < len(measures):
                    end_sec = float(measures[index + 1]["downbeat_sec"])
                elif tempo_bpm > 0.0:
                    measure_sec = meter_num * (4.0 / meter_den) * 60.0 / tempo_bpm
                    end_sec = start_sec + measure_sec
                else:
                    end_sec = start_sec + 4.0

                if end_sec <= start_sec:
                    continue

                meter_index = self.meter_to_index[(meter_num, meter_den)]
                meter_intervals.append((start_sec, end_sec, meter_index))
                downbeat_times.append(start_sec)

                # ラベルには downbeat しかないので、小節内を拍子の分子で等分して beat を補間する。
                # 6/8 は 2 拍ではなく 6 拍として扱う。
                measure_duration = end_sec - start_sec
                for beat_index in range(meter_num):
                    beat_times.append(
                        start_sec + measure_duration * beat_index / meter_num
                    )

                # BalancedSoftmaxLoss 用に meter の出現回数をフレーム単位で数える。
                start_frame = max(
                    0,
                    math.floor(start_sec * self.sample_rate / self.hop_length),
                )
                end_frame = math.ceil(end_sec * self.sample_rate / self.hop_length)
                if end_frame > start_frame:
                    meter_counts[meter_index] += float(end_frame - start_frame)

            if not meter_intervals:
                continue

            item = dict(raw_item)
            item.pop("measures")
            item.update(
                {
                    "beat_times": tuple(beat_times),
                    "downbeat_times": tuple(downbeat_times),
                    "meter_intervals": tuple(meter_intervals),
                }
            )
            items.append(item)

        if not items:
            raise ValueError(f"No usable beat labels found in {self.root}")

        self.items = items
        self.meter_class_counts = meter_counts

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        rng = random.Random(self.seed + self.epoch * len(self.items) + idx)
        info = inspect_audio(item["audio_path"])
        source_sample_rate = int(info.sample_rate)
        duration_sec = float(info.num_frames) / float(source_sample_rate)

        # 1. 曲全体から学習用の窓を選ぶ。
        #    完全ランダムではなく、なるべくアノテーション区間を含む範囲から選ぶ。
        max_start = max(0.0, duration_sec - self.window_sec)
        if max_start <= 0.0:
            window_start_sec = 0.0
        else:
            first_label = item["meter_intervals"][0][0]
            last_label = item["meter_intervals"][-1][1]
            min_start = max(0.0, first_label - self.window_sec)
            max_labeled_start = min(max_start, last_label)
            if max_labeled_start > min_start:
                window_start_sec = rng.uniform(min_start, max_labeled_start)
            else:
                window_start_sec = rng.uniform(0.0, max_start)

        # 2. wav を窓単位で読み、モデルの sample_rate と 2ch 入力にそろえる。
        source_offset = int(round(window_start_sec * source_sample_rate))
        source_frames = int(math.ceil(self.window_sec * source_sample_rate))
        audio_array, loaded_sample_rate = read_audio_frames(
            item["audio_path"],
            frame_offset=source_offset,
            num_frames=source_frames,
        )
        audio = torch.from_numpy(audio_array)
        source_sample_rate = loaded_sample_rate

        if audio.shape[0] > 2:
            audio = audio[:2]
        elif audio.shape[0] == 1:
            audio = audio.repeat(2, 1)

        if source_sample_rate != self.sample_rate:
            audio = AF.resample(
                audio,
                orig_freq=source_sample_rate,
                new_freq=self.sample_rate,
            )

        valid_audio_frames = min(int(audio.shape[-1]), self.window_frames)
        if audio.shape[-1] < self.window_frames:
            audio = F.pad(audio, (0, self.window_frames - audio.shape[-1]))
        elif audio.shape[-1] > self.window_frames:
            audio = audio[:, : self.window_frames]
        audio = audio.contiguous()

        # 3. 出力フレーム数に合わせて beat/downbeat/meter ターゲットを初期化する。
        #    meter_targets は未アノテーション区間を -100 にして loss から外す。
        beat_targets = torch.zeros(self.model_frames, dtype=torch.float32)
        downbeat_targets = torch.zeros(self.model_frames, dtype=torch.float32)
        meter_targets = torch.full((self.model_frames,), -100, dtype=torch.long)
        beat_mask = torch.zeros(self.model_frames, dtype=torch.float32)

        window_end_sec = window_start_sec + self.window_sec
        valid_model_frames = math.ceil(valid_audio_frames / self.hop_length)
        valid_model_frames = min(valid_model_frames, self.model_frames)

        # 4. meter 区間をフレームへ展開し、beat/downbeat loss 用の mask も作る。
        for start_sec, end_sec, meter_index in item["meter_intervals"]:
            overlap_start = max(start_sec, window_start_sec)
            overlap_end = min(end_sec, window_end_sec)
            if overlap_end <= overlap_start:
                continue

            start_frame = max(
                0,
                math.floor(
                    (overlap_start - window_start_sec)
                    * self.sample_rate
                    / self.hop_length
                ),
            )
            end_frame = min(
                valid_model_frames,
                math.ceil(
                    (overlap_end - window_start_sec)
                    * self.sample_rate
                    / self.hop_length
                ),
            )
            if end_frame > start_frame:
                meter_targets[start_frame:end_frame] = int(meter_index)
                beat_mask[start_frame:end_frame] = 1.0

        # 5. beat/downbeat のイベント時刻を最近傍フレームへ立てる。
        for target, times in (
            (beat_targets, item["beat_times"]),
            (downbeat_targets, item["downbeat_times"]),
        ):
            for event_sec in times:
                if event_sec < window_start_sec or event_sec >= window_end_sec:
                    continue
                frame_index = int(
                    round(
                        (event_sec - window_start_sec)
                        * self.sample_rate
                        / self.hop_length
                    )
                )
                if 0 <= frame_index < valid_model_frames:
                    target[frame_index] = 1.0

        # 6. wav が窓長より短かった padding 部分は loss から外す。
        if valid_model_frames < self.model_frames:
            beat_mask[valid_model_frames:] = 0.0
            meter_targets[valid_model_frames:] = -100

        return {
            "audio": audio,
            "valid_audio_frames": valid_audio_frames,
            "song_name": item["song_name"],
            "window_start_sec": window_start_sec,
            "beat_targets": beat_targets,
            "downbeat_targets": downbeat_targets,
            "meter_targets": meter_targets,
            "beat_mask": beat_mask,
        }


def beat_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "audio": torch.stack([item["audio"] for item in batch]),
        "valid_audio_frames": torch.tensor(
            [item["valid_audio_frames"] for item in batch],
            dtype=torch.long,
        ),
        "beat_targets": torch.stack([item["beat_targets"] for item in batch]),
        "downbeat_targets": torch.stack([item["downbeat_targets"] for item in batch]),
        "meter_targets": torch.stack([item["meter_targets"] for item in batch]),
        "beat_mask": torch.stack([item["beat_mask"] for item in batch]),
        "song_name": [item["song_name"] for item in batch],
        "window_start_sec": torch.tensor(
            [item["window_start_sec"] for item in batch],
            dtype=torch.float32,
        ),
    }


class BalancedSoftmaxLoss(nn.Module):
    def __init__(
        self,
        class_counts: Union[List[int], torch.Tensor],
        tau: float = 1.0,
        ignore_index: int = -100,
    ):
        """
        Args:
            class_counts (Union[List[int], torch.Tensor]):
                各クラスの出現回数のリストまたはテンソル。
                事前に Laplace 平滑化（全カウントに+1するなど）を推奨します。
            tau (float, optional): 補正のスケール係数. Defaults to 1.0.
        """
        super().__init__()

        class_counts = torch.as_tensor(class_counts, dtype=torch.float32)

        # log_prior を計算し、バッファとして登録
        # 希少クラスの出現回数が極小のときに log_prior が負の無限大に爆発するのを防ぐため、最小値を 1.0 に制限します
        log_prior = torch.log(torch.clamp(class_counts, min=1.0))

        self.register_buffer("log_prior", log_prior)
        self.tau = tau
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): モデルの出力ロジット (B, T, C)
            labels (torch.Tensor): 正解ラベル (B, T)

        Returns:
            torch.Tensor: 計算された損失値 (スカラー)
        """
        # 形状を合わせる
        if logits.dim() > 2:
            logits = logits.reshape(-1, logits.size(-1))  # (B*T, C)
            labels = labels.reshape(-1)  # (B*T,)

        # meter が未定義のフレームは ignore_index にして、そのまま落とす。
        valid = labels != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0

        logits = logits[valid]
        labels = labels[valid]

        # ロジット補正: z_k <- z_k + τ * log(n_k)
        adjusted_logits = logits + self.tau * self.log_prior
        loss = F.cross_entropy(adjusted_logits, labels)
        return loss


def masked_l1_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = (predictions - targets).abs()
    if mask is None:
        return diff.mean()

    weighted = diff * mask.to(diff.dtype)
    normalizer = mask.sum().clamp_min(1.0).to(diff.dtype)
    return weighted.sum() / normalizer


def masked_circular_phase_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    phase_error = 1.0 - torch.cos((predictions - targets) * (2.0 * math.pi))
    if mask is None:
        return phase_error.mean()

    weighted = phase_error * mask.to(phase_error.dtype)
    normalizer = mask.sum().clamp_min(1.0).to(phase_error.dtype)
    return weighted.sum() / normalizer


def make_bar_grid_mask(
    *,
    length: int,
    beat_count: int,
    tolerance: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """小節長と beat 数から、許容幅付きの等間隔 grid mask を作る。"""

    if length <= 0:
        raise ValueError("length must be positive")
    if beat_count <= 0:
        raise ValueError("beat_count must be positive")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    grid_mask = torch.zeros(length, device=device, dtype=dtype)
    for beat_index in range(beat_count):
        grid_position = int(
            round(float(beat_index) * float(length) / float(beat_count))
        )
        grid_position = max(0, min(length - 1, grid_position))
        grid_mask[grid_position] = 1.0

    if tolerance > 0:
        grid_mask = F.max_pool1d(
            grid_mask.view(1, 1, length),
            kernel_size=1 + 2 * tolerance,
            stride=1,
            padding=tolerance,
        ).view(length)
    return grid_mask.clamp(max=1.0)


def bar_meter_grid_ranking_loss(
    beat_logits: torch.Tensor,
    downbeat_targets: torch.Tensor,
    meter_targets: torch.Tensor,
    meter_class_beat_counts: Sequence[int],
    mask: torch.Tensor | None = None,
    *,
    tolerance: int = 1,
    margin: float = 0.1,
) -> torch.Tensor:
    """正解 meter の beat grid が他候補より高スコアになるようにする小節単位 loss。"""

    if beat_logits.shape != downbeat_targets.shape:
        raise ValueError("beat_logits and downbeat_targets must have the same shape")
    if beat_logits.shape != meter_targets.shape:
        raise ValueError("beat_logits and meter_targets must have the same shape")
    if mask is not None and beat_logits.shape != mask.shape:
        raise ValueError("beat_logits and mask must have the same shape")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    if len(meter_class_beat_counts) == 0:
        raise ValueError("meter_class_beat_counts must not be empty")

    unique_beat_counts = tuple(
        sorted({int(count) for count in meter_class_beat_counts})
    )
    if len(unique_beat_counts) <= 1:
        return beat_logits.sum() * 0.0
    beat_count_to_grid_index = {
        beat_count: index for index, beat_count in enumerate(unique_beat_counts)
    }

    beat_prob = torch.sigmoid(beat_logits)
    bar_losses: list[torch.Tensor] = []

    # 1. downbeat のペアから、窓内で完結している小節だけを取り出す。
    for batch_index in range(beat_logits.shape[0]):
        if mask is None:
            valid_mask = torch.ones_like(
                downbeat_targets[batch_index], dtype=torch.bool
            )
        else:
            valid_mask = mask[batch_index] > 0.0
        downbeat_frames = torch.nonzero(
            (downbeat_targets[batch_index] > 0.5) & valid_mask,
            as_tuple=False,
        ).flatten()
        if downbeat_frames.numel() < 2:
            continue

        for start_tensor, end_tensor in zip(downbeat_frames[:-1], downbeat_frames[1:]):
            start_frame = int(start_tensor.item())
            end_frame = int(end_tensor.item())
            bar_length = end_frame - start_frame
            if bar_length <= 1:
                continue

            bar_valid_mask = valid_mask[start_frame:end_frame]
            bar_meter_targets = meter_targets[batch_index, start_frame:end_frame]
            meter_valid = (bar_meter_targets >= 0) & bar_valid_mask
            if not torch.any(meter_valid):
                continue

            # 2. 小節内で最も多い meter class を、この小節の正解 meter とみなす。
            target_meter_class = int(
                torch.mode(bar_meter_targets[meter_valid]).values.item()
            )
            if target_meter_class < 0 or target_meter_class >= len(
                meter_class_beat_counts
            ):
                continue
            target_beat_count = int(meter_class_beat_counts[target_meter_class])
            target_grid_index = beat_count_to_grid_index.get(target_beat_count)
            if target_grid_index is None:
                continue

            bar_prob = beat_prob[batch_index, start_frame:end_frame]
            bar_weight = bar_valid_mask.to(dtype=bar_prob.dtype)
            grid_scores: list[torch.Tensor] = []

            # 3. 各 meter 候補の等間隔 grid で beat 分布をスコア化する。
            for beat_count in unique_beat_counts:
                grid_mask = make_bar_grid_mask(
                    length=bar_length,
                    beat_count=beat_count,
                    tolerance=tolerance,
                    device=bar_prob.device,
                    dtype=bar_prob.dtype,
                )
                on_grid_weight = grid_mask * bar_weight
                off_grid_weight = (1.0 - grid_mask) * bar_weight
                on_grid_score = (bar_prob * on_grid_weight).sum()
                on_grid_score = on_grid_score / on_grid_weight.sum().clamp_min(1.0)
                off_grid_score = (bar_prob * off_grid_weight).sum()
                off_grid_score = off_grid_score / off_grid_weight.sum().clamp_min(1.0)
                grid_scores.append(on_grid_score - off_grid_score)

            # 4. 正解 grid が他の grid より margin 以上高くなるように ranking loss を取る。
            score_tensor = torch.stack(grid_scores)
            target_score = score_tensor[target_grid_index]
            other_mask = torch.ones(
                len(unique_beat_counts),
                device=score_tensor.device,
                dtype=torch.bool,
            )
            other_mask[target_grid_index] = False
            other_scores = score_tensor[other_mask]
            bar_losses.append(F.relu(margin + other_scores - target_score).mean())

    if not bar_losses:
        return beat_logits.sum() * 0.0
    return torch.stack(bar_losses).mean()


def estimate_bar_phase_step(
    bar_phase_targets: torch.Tensor,
    phase_mask: torch.Tensor,
) -> torch.Tensor:
    """bar phase の隣接差分から、各フレームのおおよその1フレーム進行量を推定する。"""

    phase_delta = bar_phase_targets[:, 1:] - bar_phase_targets[:, :-1]
    valid_delta = (
        (phase_delta > 0.0) & (phase_mask[:, 1:] > 0.0) & (phase_mask[:, :-1] > 0.0)
    )
    next_step = F.pad(
        torch.where(valid_delta, phase_delta, torch.zeros_like(phase_delta)),
        (0, 1),
    )
    prev_step = F.pad(
        torch.where(valid_delta, phase_delta, torch.zeros_like(phase_delta)),
        (1, 0),
    )
    return torch.where(next_step > 0.0, next_step, prev_step)


def phase_grid_masks_from_targets(
    *,
    unique_beat_counts: torch.Tensor,
    bar_phase_targets: torch.Tensor,
    phase_mask: torch.Tensor,
    tolerance: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """bar phase から各 meter 候補の grid mask と有効 frame mask を作る。"""

    phase_step = estimate_bar_phase_step(bar_phase_targets, phase_mask)
    counts = unique_beat_counts.to(
        device=bar_phase_targets.device,
        dtype=dtype,
    ).view(-1, 1, 1)
    phase = bar_phase_targets.clamp(0.0, 1.0).to(dtype).unsqueeze(0)
    phase_step_expanded = phase_step.to(dtype).unsqueeze(0).clamp_min(1e-6)
    cycle_position = phase * counts
    distance_in_cycles = (cycle_position - torch.round(cycle_position)).abs()
    distance_in_frames = (distance_in_cycles / counts) / phase_step_expanded
    grid_mask = (distance_in_frames <= float(tolerance)).to(dtype)
    return grid_mask, phase_step > 0.0


def score_phase_grids(
    *,
    beat_prob: torch.Tensor,
    grid_mask: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """候補 grid ごとの on-grid/off-grid 差分 score を計算する。"""

    candidate_weight = weight.unsqueeze(0)
    on_grid_weight = grid_mask * candidate_weight
    off_grid_weight = (1.0 - grid_mask) * candidate_weight
    beat_prob_expanded = beat_prob.unsqueeze(0)

    on_grid_score = (beat_prob_expanded * on_grid_weight).sum(dim=(1, 2))
    on_grid_score = on_grid_score / on_grid_weight.sum(dim=(1, 2)).clamp_min(1.0)
    off_grid_score = (beat_prob_expanded * off_grid_weight).sum(dim=(1, 2))
    off_grid_score = off_grid_score / off_grid_weight.sum(dim=(1, 2)).clamp_min(1.0)
    return on_grid_score - off_grid_score


def meter_logits_to_grid_probs(
    *,
    meter_logits: torch.Tensor,
    meter_class_grid_indices: torch.Tensor,
    weight: torch.Tensor,
    num_grids: int,
) -> torch.Tensor:
    """meter class 確率を、同じ beat 数を持つ grid group ごとの確率に集約する。"""

    meter_probs = torch.softmax(meter_logits, dim=-1)
    grid_probs: list[torch.Tensor] = []
    normalizer = weight.sum().clamp_min(1.0)
    class_grid_indices = meter_class_grid_indices.to(meter_logits.device)
    for grid_index in range(num_grids):
        class_mask = class_grid_indices == grid_index
        frame_grid_prob = meter_probs[..., class_mask].sum(dim=-1)
        grid_probs.append((frame_grid_prob * weight).sum() / normalizer)

    probs = torch.stack(grid_probs)
    probs = probs.clamp_min(1e-8)
    return probs / probs.sum().clamp_min(1e-8)


def phase_meter_grid_ranking_loss(
    beat_logits: torch.Tensor,
    meter_targets: torch.Tensor,
    unique_beat_counts: torch.Tensor,
    meter_class_grid_indices: torch.Tensor,
    bar_phase_targets: torch.Tensor,
    phase_mask: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    tolerance: int = 1,
    margin: float = 0.1,
) -> torch.Tensor:
    """bar phase target を使って、meter grid ranking をベクトル化して計算する。"""

    if beat_logits.shape != meter_targets.shape:
        raise ValueError("beat_logits and meter_targets must have the same shape")
    if beat_logits.shape != bar_phase_targets.shape:
        raise ValueError("beat_logits and bar_phase_targets must have the same shape")
    if beat_logits.shape != phase_mask.shape:
        raise ValueError("beat_logits and phase_mask must have the same shape")
    if mask is not None and beat_logits.shape != mask.shape:
        raise ValueError("beat_logits and mask must have the same shape")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    if unique_beat_counts.numel() <= 1:
        return beat_logits.sum() * 0.0

    dtype = beat_logits.dtype
    beat_prob = torch.sigmoid(beat_logits)
    valid_mask = (meter_targets >= 0) & (phase_mask > 0.0)
    if mask is not None:
        valid_mask = valid_mask & (mask > 0.0)

    # 1. bar phase から「grid まで何フレーム離れているか」を候補 meter ごとに計算する。
    grid_mask, valid_phase_step = phase_grid_masks_from_targets(
        unique_beat_counts=unique_beat_counts,
        bar_phase_targets=bar_phase_targets,
        phase_mask=phase_mask,
        tolerance=tolerance,
        dtype=dtype,
    )
    valid_mask = valid_mask & valid_phase_step

    # 2. meter class を「候補 beat 数の index」に変換し、同じ正解 grid ごとに集計する。
    safe_meter_targets = meter_targets.clamp(
        min=0,
        max=int(meter_class_grid_indices.numel()) - 1,
    )
    target_grid_indices = meter_class_grid_indices.to(beat_logits.device)[
        safe_meter_targets
    ]
    base_weight = valid_mask.to(dtype)
    weighted_losses: list[torch.Tensor] = []
    group_weights: list[torch.Tensor] = []

    for target_grid_index in range(int(unique_beat_counts.numel())):
        target_weight = base_weight * (target_grid_indices == target_grid_index).to(
            dtype
        )
        target_weight_sum = target_weight.sum()
        group_present = (target_weight_sum > 0.0).to(dtype)

        scores = score_phase_grids(
            beat_prob=beat_prob,
            grid_mask=grid_mask,
            weight=target_weight,
        )

        target_score = scores[target_grid_index]
        other_loss = F.relu(margin + scores - target_score)
        other_loss = other_loss.clone()
        other_loss[target_grid_index] = 0.0
        weighted_losses.append(
            other_loss.sum() / max(1, int(unique_beat_counts.numel()) - 1)
        )
        group_weights.append(group_present)

    loss_tensor = torch.stack(weighted_losses)
    weight_tensor = torch.stack(group_weights)
    return (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(1.0)


def phase_meter_grid_kl_loss(
    beat_logits: torch.Tensor,
    meter_logits: torch.Tensor,
    meter_targets: torch.Tensor,
    unique_beat_counts: torch.Tensor,
    meter_class_grid_indices: torch.Tensor,
    bar_phase_targets: torch.Tensor,
    phase_mask: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    tolerance: int = 1,
    temperature: float = 0.2,
) -> torch.Tensor:
    """detach した beat-grid 分布に meter logits の grid 分布を合わせる KL loss。"""

    if beat_logits.shape != meter_targets.shape:
        raise ValueError("beat_logits and meter_targets must have the same shape")
    if beat_logits.shape != bar_phase_targets.shape:
        raise ValueError("beat_logits and bar_phase_targets must have the same shape")
    if beat_logits.shape != phase_mask.shape:
        raise ValueError("beat_logits and phase_mask must have the same shape")
    if meter_logits.shape[:2] != beat_logits.shape:
        raise ValueError("meter_logits must have shape [B, T, C]")
    if mask is not None and beat_logits.shape != mask.shape:
        raise ValueError("beat_logits and mask must have the same shape")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if unique_beat_counts.numel() <= 1:
        return beat_logits.sum() * 0.0

    dtype = beat_logits.dtype
    valid_mask = (meter_targets >= 0) & (phase_mask > 0.0)
    if mask is not None:
        valid_mask = valid_mask & (mask > 0.0)

    grid_mask, valid_phase_step = phase_grid_masks_from_targets(
        unique_beat_counts=unique_beat_counts,
        bar_phase_targets=bar_phase_targets,
        phase_mask=phase_mask,
        tolerance=tolerance,
        dtype=dtype,
    )
    valid_mask = valid_mask & valid_phase_step

    safe_meter_targets = meter_targets.clamp(
        min=0,
        max=int(meter_class_grid_indices.numel()) - 1,
    )
    target_grid_indices = meter_class_grid_indices.to(beat_logits.device)[
        safe_meter_targets
    ]
    base_weight = valid_mask.to(dtype)
    beat_prob = torch.sigmoid(beat_logits.detach())
    losses: list[torch.Tensor] = []
    group_weights: list[torch.Tensor] = []

    # 1. 正解 grid group ごとに分け、複数拍子が混ざる batch でも分布を混ぜない。
    for target_grid_index in range(int(unique_beat_counts.numel())):
        target_weight = base_weight * (target_grid_indices == target_grid_index).to(
            dtype
        )
        group_present = (target_weight.sum() > 0.0).to(dtype)

        # 2. beat pattern から q_grid を作る。detach 済みなので beat 側へは勾配を返さない。
        with torch.no_grad():
            grid_scores = score_phase_grids(
                beat_prob=beat_prob,
                grid_mask=grid_mask,
                weight=target_weight,
            )
            q_grid = torch.softmax(grid_scores / float(temperature), dim=0)

        # 3. meter_logits を同じ grid group 空間に集約し、KL(q_grid || p_meter_grid) を取る。
        p_grid = meter_logits_to_grid_probs(
            meter_logits=meter_logits,
            meter_class_grid_indices=meter_class_grid_indices,
            weight=target_weight,
            num_grids=int(unique_beat_counts.numel()),
        )
        losses.append((q_grid * (q_grid.clamp_min(1e-8).log() - p_grid.log())).sum())
        group_weights.append(group_present)

    loss_tensor = torch.stack(losses)
    weight_tensor = torch.stack(group_weights)
    return (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(1.0)


# https://github.com/CPJKU/beat_this/blob/main/beat_this/model/loss.py
class ShiftTolerantBCELoss(torch.nn.Module):
    """
    少しずれた beat/downbeat ラベルを許容する BCE loss。
    予測側を max-pooling し、正解フレーム周辺で最も強い予測に勾配を流す。
    """

    def __init__(self, pos_weight: float = 1, tolerance: int = 1):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight, dtype=torch.get_default_dtype()),
            persistent=False,
        )
        self.tolerance = tolerance

    def spread(self, x: torch.Tensor, factor: int = 1):
        if self.tolerance == 0:
            return x
        return F.max_pool1d(x, 1 + 2 * factor * self.tolerance, 1)

    def crop(self, x: torch.Tensor, factor: int = 1):
        return x[..., factor * self.tolerance : -factor * self.tolerance or None]

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        # 予測側を tolerance 分だけ広げ、端の不確かなフレームを落とす。
        spreaded_preds = self.crop(self.spread(preds))
        cropped_targets = self.crop(targets, factor=2)
        # 正解 beat 周辺の負例は見ない。padding や未アノテーション区間も mask で落とす。
        look_at = cropped_targets + (1 - self.spread(targets, factor=2))
        if mask is not None:
            look_at = look_at * self.crop(mask, factor=2)
        return F.binary_cross_entropy_with_logits(
            spreaded_preds,
            cropped_targets,
            weight=look_at,
            pos_weight=self.pos_weight,
        )


class BeatHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_meter_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        phase_peak_sharpness: float = 12.0,
        min_phase_velocity: float = 1e-4,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_meter_classes <= 0:
            raise ValueError("num_meter_classes must be positive")
        if hidden_dim is None:
            hidden_dim = input_dim
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if phase_peak_sharpness <= 0.0:
            raise ValueError("phase_peak_sharpness must be positive")
        if min_phase_velocity <= 0.0:
            raise ValueError("min_phase_velocity must be positive")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_meter_classes = int(num_meter_classes)
        self.phase_peak_sharpness = float(phase_peak_sharpness)
        self.min_phase_velocity = float(min_phase_velocity)

        self.shared = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.frame_proj = nn.Linear(self.hidden_dim, self.num_meter_classes + 2)
        self.group_boundary_proj = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.group_boundary_proj.weight)
        nn.init.zeros_(self.group_boundary_proj.bias)

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        # Legacy checkpoints predate the independent major-boundary head. A
        # zero projection is neutral for the decoder (p=0.5) and preserves
        # strict inference loading for those checkpoints.
        for parameter_name in ("weight", "bias"):
            key = f"{prefix}group_boundary_proj.{parameter_name}"
            if key not in state_dict:
                parameter = getattr(self.group_boundary_proj, parameter_name)
                state_dict[key] = parameter.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if features.dim() != 3:
            raise ValueError("features must have shape [B, T, D]")
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"features last dim must be {self.input_dim}, got {features.shape[-1]}"
            )

        shared_features = self.shared(features)
        frame_outputs = self.frame_proj(shared_features)
        group_boundary_logits = self.group_boundary_proj(shared_features).squeeze(-1)
        (
            beat_logits,
            downbeat_logits,
            meter_logits,
        ) = torch.split(
            frame_outputs,
            [1, 1, self.num_meter_classes],
            dim=-1,
        )

        beat_logits = beat_logits.squeeze(-1)
        downbeat_logits = downbeat_logits.squeeze(-1)

        # Beat This! の SumHead と同じく、downbeat logit を beat logit に加える。
        # downbeat は必ず beat でもある、という包含関係を head 自体に持たせる。
        # AMP 中の低精度加算による NaN を避けるため、この加算だけ float32 で行う。
        if hasattr(
            torch.amp, "is_autocast_available"
        ) and not torch.amp.is_autocast_available(beat_logits.device.type):
            autocast_context = contextlib.nullcontext()
        else:
            autocast_context = torch.autocast(
                beat_logits.device.type,
                enabled=False,
            )
        with autocast_context:
            beat_logits = beat_logits.float() + downbeat_logits.float()

        return {
            "beat_logits": beat_logits,
            "downbeat_logits": downbeat_logits,
            "meter_logits": meter_logits,
            "group_boundary_logits": group_boundary_logits,
        }


class BeatLoss(nn.Module):
    def __init__(
        self,
        config: BeatConfig,
        meter_class_counts: Union[List[int], torch.Tensor],
        meter_classes: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.meter_classes = (
            tuple(
                (int(meter_num), int(meter_den))
                for meter_num, meter_den in meter_classes
            )
            if meter_classes is not None
            else ()
        )
        self.meter_class_beat_counts = (
            tuple(int(meter_num) for meter_num, _meter_den in self.meter_classes)
            if self.meter_classes
            else ()
        )
        unique_beat_counts = tuple(sorted(set(self.meter_class_beat_counts)))
        beat_count_to_grid_index = {
            beat_count: index for index, beat_count in enumerate(unique_beat_counts)
        }
        meter_class_grid_indices = tuple(
            beat_count_to_grid_index[beat_count]
            for beat_count in self.meter_class_beat_counts
        )
        self.register_buffer(
            "meter_grid_unique_beat_counts",
            torch.tensor(unique_beat_counts, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "meter_class_grid_indices",
            torch.tensor(meter_class_grid_indices, dtype=torch.long),
            persistent=False,
        )
        if self.meter_class_beat_counts and len(self.meter_class_beat_counts) != int(
            torch.as_tensor(meter_class_counts).numel()
        ):
            raise ValueError("meter_classes must match meter_class_counts length")
        if (
            config.meter_grid_ranking_loss_weight > 0.0
            and not self.meter_class_beat_counts
        ):
            raise ValueError(
                "meter_classes is required when meter_grid_ranking_loss_weight is positive"
            )
        if config.meter_grid_kl_loss_weight > 0.0 and not self.meter_class_beat_counts:
            raise ValueError(
                "meter_classes is required when meter_grid_kl_loss_weight is positive"
            )
        self.beat_loss = ShiftTolerantBCELoss(
            pos_weight=config.beat_pos_weight,
            tolerance=config.loss_tolerance,
        )
        self.downbeat_loss = ShiftTolerantBCELoss(
            pos_weight=config.downbeat_pos_weight,
            tolerance=config.loss_tolerance,
        )
        self.meter_loss = BalancedSoftmaxLoss(meter_class_counts, tau=0.3)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        beat_logits = outputs.get("beat_logits")
        downbeat_logits = outputs.get("downbeat_logits")
        meter_logits = outputs.get("meter_logits")
        if beat_logits is None or downbeat_logits is None or meter_logits is None:
            raise ValueError("Beat training requires beat/downbeat/meter logits")

        beat_targets = batch["beat_targets"].to(beat_logits.device)
        downbeat_targets = batch["downbeat_targets"].to(downbeat_logits.device)
        meter_targets = batch["meter_targets"].to(meter_logits.device)
        beat_mask = batch.get("beat_mask")
        if beat_mask is not None:
            beat_mask = beat_mask.to(beat_logits.device)
        bar_phase_targets = batch.get("bar_phase_targets")
        if bar_phase_targets is not None:
            bar_phase_targets = bar_phase_targets.to(beat_logits.device)
        phase_mask = batch.get("phase_mask")
        if phase_mask is not None:
            phase_mask = phase_mask.to(beat_logits.device)

        beat_loss = self.beat_loss(
            beat_logits.unsqueeze(1),
            beat_targets.unsqueeze(1),
            None if beat_mask is None else beat_mask.unsqueeze(1),
        )
        downbeat_loss = self.downbeat_loss(
            downbeat_logits.unsqueeze(1),
            downbeat_targets.unsqueeze(1),
            None if beat_mask is None else beat_mask.unsqueeze(1),
        )
        meter_loss = self.meter_loss(meter_logits, meter_targets)
        meter_grid_ranking_loss = beat_logits.sum() * 0.0
        if self.config.meter_grid_ranking_loss_weight > 0.0:
            if bar_phase_targets is not None and phase_mask is not None:
                meter_grid_ranking_loss = phase_meter_grid_ranking_loss(
                    beat_logits,
                    meter_targets,
                    self.meter_grid_unique_beat_counts,
                    self.meter_class_grid_indices,
                    bar_phase_targets,
                    phase_mask,
                    beat_mask,
                    tolerance=self.config.loss_tolerance,
                    margin=self.config.meter_grid_ranking_margin,
                )
            else:
                meter_grid_ranking_loss = bar_meter_grid_ranking_loss(
                    beat_logits,
                    downbeat_targets,
                    meter_targets,
                    self.meter_class_beat_counts,
                    beat_mask,
                    tolerance=self.config.loss_tolerance,
                    margin=self.config.meter_grid_ranking_margin,
                )
        meter_grid_kl_loss = beat_logits.sum() * 0.0
        if self.config.meter_grid_kl_loss_weight > 0.0:
            if bar_phase_targets is None or phase_mask is None:
                raise ValueError(
                    "meter_grid_kl_loss requires bar_phase_targets and phase_mask"
                )
            meter_grid_kl_loss = phase_meter_grid_kl_loss(
                beat_logits,
                meter_logits,
                meter_targets,
                self.meter_grid_unique_beat_counts,
                self.meter_class_grid_indices,
                bar_phase_targets,
                phase_mask,
                beat_mask,
                tolerance=self.config.loss_tolerance,
                temperature=self.config.meter_grid_kl_temperature,
            )

        major_grouping_total_loss = beat_logits.sum() * 0.0
        major_grouping_valid_loss = beat_logits.sum() * 0.0
        major_grouping_accent_loss = beat_logits.sum() * 0.0
        major_grouping_bar_count = beat_logits.sum() * 0.0
        meter_aware_crop_rate = beat_logits.sum() * 0.0
        meter_aware_crop = batch.get("meter_aware_crop")
        if meter_aware_crop is not None:
            meter_aware_crop_rate = meter_aware_crop.to(
                device=beat_logits.device,
                dtype=beat_logits.dtype,
            ).mean()
        group_boundary_logits = outputs.get("group_boundary_logits")
        midi_frames = batch.get("midi_frames")
        if (
            self.config.major_grouping_loss_weight > 0.0
            and group_boundary_logits is not None
            and midi_frames is not None
            and self.meter_classes
        ):
            grouping_result = major_grouping_loss(
                group_boundary_logits=group_boundary_logits,
                beat_targets=beat_targets,
                downbeat_targets=downbeat_targets,
                meter_targets=meter_targets,
                beat_mask=beat_mask,
                midi_frames=midi_frames.to(group_boundary_logits.device),
                meter_classes=self.meter_classes,
                tolerance=self.config.loss_tolerance,
                accent_loss_weight=(self.config.major_grouping_accent_loss_weight),
                accent_temperature=(self.config.major_grouping_accent_temperature),
            )
            major_grouping_total_loss = grouping_result.loss
            major_grouping_valid_loss = grouping_result.valid_pattern_loss
            major_grouping_accent_loss = grouping_result.accent_alignment_loss
            major_grouping_bar_count = torch.tensor(
                float(grouping_result.supervised_bar_count),
                device=beat_logits.device,
                dtype=beat_logits.dtype,
            )

        total_loss = (
            beat_loss
            + downbeat_loss
            + meter_loss * float(self.config.meter_loss_weight)
            + meter_grid_ranking_loss
            * float(self.config.meter_grid_ranking_loss_weight)
            + meter_grid_kl_loss * float(self.config.meter_grid_kl_loss_weight)
            + major_grouping_total_loss * float(self.config.major_grouping_loss_weight)
        )
        return total_loss, {
            "beat_total_loss": total_loss,
            "beat_loss": beat_loss,
            "downbeat_loss": downbeat_loss,
            "meter_loss": meter_loss,
            "meter_grid_ranking_loss": meter_grid_ranking_loss,
            "meter_grid_kl_loss": meter_grid_kl_loss,
            "major_grouping_loss": major_grouping_total_loss,
            "major_grouping_valid_loss": major_grouping_valid_loss,
            "major_grouping_accent_loss": major_grouping_accent_loss,
            "major_grouping_bar_count": major_grouping_bar_count,
            "meter_aware_crop_rate": meter_aware_crop_rate,
        }


def train_beat_batch(
    *,
    model: nn.Module,
    batch: Dict[str, Any],
    loss_fn: BeatLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
    device: torch.device,
    ema_model: Any = None,
    grad_clip_norm: float = 1.0,
    loss_scale: float = 1.0,
) -> Tuple[float, Dict[str, torch.Tensor], bool]:
    total_loss, loss_dict = compute_beat_batch_loss(
        model=model,
        batch=batch,
        loss_fn=loss_fn,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        device=device,
        loss_scale=loss_scale,
    )

    optimizer.zero_grad(set_to_none=True)

    if scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_step_was_skipped = scaler.get_scale() < scale_before_step
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        optimizer_step_was_skipped = False

    if not optimizer_step_was_skipped:
        scheduler.step()
        if ema_model is not None:
            ema_model.update(model)

    return float(total_loss.item()), loss_dict, optimizer_step_was_skipped


def compute_beat_batch_loss(
    *,
    model: nn.Module,
    batch: Dict[str, Any],
    loss_fn: BeatLoss,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
    device: torch.device,
    loss_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    audio = batch["audio"]
    valid_audio_frames = batch["valid_audio_frames"]

    with torch.amp.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=use_amp,
    ):
        outputs = model(
            audio,
            valid_audio_frames=valid_audio_frames,
            include_amt=False,
            include_beat=True,
            include_chord=False,
        )
        total_loss, loss_dict = loss_fn(outputs, batch)
        total_loss = total_loss * loss_scale

    return total_loss, loss_dict
