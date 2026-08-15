import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple  # noqa: UP035

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from ...data.audio import inspect_audio, read_audio_frames
from ..data.modulation import (
    ModulationAugmentConfig,
    inject_synthetic_boundaries,
)


@dataclass(frozen=True)
class ChordConfig:
    # 損失の重み
    chord_boundary_weight: float = 3.0
    root_chord_weight: float = 1.0
    bass_weight: float = 1.0
    key_boundary_weight: float = 3.0
    key_weight: float = 1.0
    chord_pitch_weight: float = 10.0

    # 各損失の設定
    boundary_pos_weight: float = 5.0
    key_boundary_pos_weight: float = 150.0
    chord_boundary_loss_tolerance: int = 1
    key_boundary_loss_tolerance: int = 8
    focal_tversky_alpha: float = 0.3
    focal_tversky_gamma: float = 1.5

    def __post_init__(self) -> None:
        # 非負チェック
        for field in self.__dataclass_fields__:
            val = getattr(self, field)
            if isinstance(val, (int, float)) and val < 0:
                raise ValueError(f"{field} must be non-negative")


def add_chord_training_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Chord training")
    group.add_argument(
        "--chord_dataset_path",
        type=str,
        default="beat_chord_dataset/chord_dataset",
        help="Path to chord dataset root containing audio/, chord_label/, and key_label/.",
    )
    group.add_argument(
        "--enable_chord_training",
        action="store_true",
        help="Enable auxiliary chord training.",
    )
    group.add_argument(
        "--chord_batch_size",
        type=int,
        default=0,
        help="Chord batch size. Uses --batch_size when set to 0.",
    )
    group.add_argument(
        "--chord_num_workers",
        type=int,
        default=-1,
        help="Chord DataLoader workers. Uses --num_workers when set to -1.",
    )
    group.add_argument(
        "--chord_loss_scale",
        type=float,
        default=0.1,
        help="Scale for chord loss to control its impact on the backbone.",
    )
    group.add_argument(
        "--chord_update_interval",
        type=int,
        default=1,
        help="Run chord auxiliary updates every N AMT steps.",
    )
    group.add_argument(
        "--chord_boundary_loss_tolerance",
        type=int,
        default=1,
        help="Frame tolerance for chord-boundary BCE.",
    )
    group.add_argument(
        "--key_boundary_loss_tolerance",
        type=int,
        default=8,
        help="Frame tolerance for key-boundary BCE.",
    )


def chord_config_from_args(args: Any) -> ChordConfig:
    # 必要に応じて引数から Config を作成する（現在はデフォルト値を使用）
    return ChordConfig(
        chord_boundary_loss_tolerance=int(
            getattr(args, "chord_boundary_loss_tolerance", 1)
        ),
        key_boundary_loss_tolerance=int(
            getattr(args, "key_boundary_loss_tolerance", 8)
        ),
    )


from dlchordx import Tone
from dlchordx.const import CHORD_MAP


def _require_torchaudio():
    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            "ChordDataset requires torchaudio to resample audio files"
        ) from exc
    return torchaudio


# 音名からインデックスへのマッピング
ROOT_TO_INDEX = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "N": 12,
}


def normalize_chord_quality(quality: str) -> str:
    """コード品質ラベルの空白表記ゆれを、quality.json と同じキーへそろえる。"""
    return "".join(str(quality).split())


def parse_key_to_relative_major(key_str: str) -> int:
    """キー文字列を解析し、相対長調のルートインデックス (0-11) または N (12) を返す。"""
    if key_str == "N":
        return 12

    # 'm' で終わるか、あるいは 'Am' のような形式を判定
    is_minor = False
    root_part = key_str
    if key_str.endswith("m"):
        is_minor = True
        root_part = key_str[:-1]

    try:
        root_idx = int(Tone(root_part).get_interval())
    except Exception:
        root_idx = ROOT_TO_INDEX.get(root_part, 12)

    if root_idx >= 12:
        return 12

    if is_minor:
        # 短調の場合は3半音上げて相対長調にする
        return (root_idx + 3) % 12
    return root_idx


class ChordDataset(Dataset):
    """
    chord_dataset を読み込む Dataset。
    root_chord (Root*Quality+N), bass, key (相対長調), chord_pitch (25次元) を扱う。
    """

    def __init__(
        self,
        root: str | Path,
        *,
        window_ms: int,
        sample_rate: int,
        hop_length: int,
        seed: int = 42,
        modulation_config: ModulationAugmentConfig | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.audio_dir = self.root / "audio"
        self.chord_label_dir = self.root / "chord_label"
        self.key_label_dir = self.root / "key_label"
        self.window_ms = int(window_ms)
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.seed = int(seed)
        self.epoch = 0
        self.modulation_config = modulation_config or ModulationAugmentConfig()

        # Quality 情報の読み込み
        with open(self.root / "quality.json", "r", encoding="utf-8") as f:
            self.quality_map = json.load(f)  # index -> quality_str
        with open(self.root / "quality_freq_count.json", "r", encoding="utf-8") as f:
            self.quality_freqs = json.load(f)

        # dlchordx の CHORD_MAP を前処理
        self.dl_chord_map = {
            normalize_chord_quality(key): val for key, val in CHORD_MAP.items()
        }

        self.num_qualities = len(self.quality_map)  # 通常 63 ('N' 含む)
        # Root (12) * Quality (62; 'N'除外) + N (1) = 745
        # 'N' は quality_map の最後 (62) と仮定
        self.n_quality_idx = 62
        self.num_root_chord_classes = 12 * (self.num_qualities - 1) + 1
        self.quality_to_idx = {
            normalize_chord_quality(v): int(k) for k, v in self.quality_map.items()
        }

        # BalancedSoftmax 用の頻度計算
        # quality_freqs (64?) を 12 等分して各ルートに割り当てる
        self.root_chord_counts = torch.zeros(self.num_root_chord_classes)
        for q_idx in range(self.num_qualities):
            count = float(self.quality_freqs[q_idx])
            if q_idx == self.n_quality_idx:
                self.root_chord_counts[-1] = count
            else:
                for r in range(12):
                    self.root_chord_counts[r * (self.num_qualities - 1) + q_idx] = (
                        count / 12.0
                    )

        torchaudio = _require_torchaudio()
        audio_files = {p.stem: p for p in self.audio_dir.glob("*.wav")}

        self.items = []
        for label_path in sorted(self.chord_label_dir.glob("*.jsonl")):
            stem = label_path.stem
            audio_path = audio_files.get(stem)
            key_path = self.key_label_dir / f"{stem}.txt"
            if not audio_path or not key_path.exists():
                continue

            # コードラベルの読み込み
            chords = []
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chords.append(json.loads(line))

            # キーラベルの読み込み
            keys = []
            with open(key_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(":")
                    if len(parts) == 3:
                        keys.append(
                            {
                                "start_time": float(parts[0]),
                                "end_time": float(parts[1]),
                                "key": parts[2],
                            }
                        )

            if not chords:
                continue

            self.items.append(
                {
                    "song_name": stem,
                    "audio_path": audio_path,
                    "chords": chords,
                    "keys": keys,
                }
            )

        self.window_frames = int(round(self.window_ms * self.sample_rate / 1000.0))
        self.window_sec = self.window_frames / self.sample_rate
        self.model_frames = math.ceil(self.window_frames / self.hop_length)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def _get_root_chord_index(self, root: str, quality_idx: int) -> int:
        if root == "N" or quality_idx == self.n_quality_idx:
            return self.num_root_chord_classes - 1
        r_idx = ROOT_TO_INDEX.get(root, 12)
        if r_idx >= 12:
            return self.num_root_chord_classes - 1
        return r_idx * (self.num_qualities - 1) + quality_idx

    def _create_chord_vector(self, root: str, quality: str, bass: str) -> torch.Tensor:
        """25次元のコードベクトル (12次元ピッチ + 13次元ベース) を生成する。"""
        vec = torch.zeros(25)
        # "N" (No Chord) の場合はゼロベクトルを返す (LabelProcessor に準拠)
        if root == "N" or quality == "N":
            return vec

        try:
            r_idx = int(Tone(root).get_interval())
            # ピッチクラス (0-11)
            q_clean = normalize_chord_quality(quality)
            if q_clean in self.dl_chord_map:
                for interval in self.dl_chord_map[q_clean]:
                    # ルート音からの相対音程を足して12で割った余りを計算
                    vec[(r_idx + int(interval)) % 12] = 1.0

            # 13次元のベース音ベクトル (12-24)
            # "N"は0、Cは1、... Bは12
            # 連結後は 12+0=12, 12+1=13...
            bass_idx = int(Tone(bass).get_interval()) + 1 if bass != "N" else 0
            vec[12 + bass_idx] = 1.0

        except Exception:
            # 解析できない場合はゼロベクトルのまま
            pass

        return vec

    def _load_audio_window(
        self,
        *,
        torchaudio: Any,
        audio_path: Path,
        frame_offset: int,
        num_frames: int,
    ) -> torch.Tensor:
        """同じ窓位置の音声を読み、学習入力の長さへ正規化する。"""
        audio_array, source_sample_rate = read_audio_frames(
            audio_path,
            frame_offset=int(frame_offset),
            num_frames=int(num_frames),
        )
        audio = torch.from_numpy(audio_array)

        # 1. チャンネル数を 2ch にそろえる。
        if audio.shape[0] > 2:
            audio = audio[:2]
        elif audio.shape[0] == 1:
            audio = audio.repeat(2, 1)

        # 2. モデル側の sample_rate へそろえる。
        if int(source_sample_rate) != self.sample_rate:
            audio = torchaudio.functional.resample(
                audio,
                int(source_sample_rate),
                self.sample_rate,
            )

        # 3. 最終的な窓長へ pad / trim する。
        if audio.shape[-1] < self.window_frames:
            audio = F.pad(audio, (0, self.window_frames - audio.shape[-1]))
        else:
            audio = audio[:, : self.window_frames]
        return audio

    def _maybe_apply_modulation(
        self,
        *,
        item: dict[str, Any],
        audio: torch.Tensor,
        window_start_sec: float,
        window_end_sec: float,
        frame_offset: int,
        num_frames: int,
        rng: random.Random,
        torchaudio: Any,
    ) -> tuple[torch.Tensor, List[Dict[str, Any]], List[Dict[str, Any]], int | None]:
        """条件を満たすときだけ既存 `_pitch_±n` 素材で人工転調を合成する。"""
        # Synthetic modulation is defined on MIDI rolls.  This legacy audio-only
        # dataset intentionally leaves the waveform and labels unchanged.
        return audio, item["chords"], item["keys"], None

    def _render_chord_targets(
        self,
        *,
        chord_segments: List[Dict[str, Any]],
        window_start_sec: float,
        window_end_sec: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """コードラベル列を frame 単位ターゲットへ展開する。"""
        chord_boundary = torch.zeros(self.model_frames)
        root_chord_targets = torch.full((self.model_frames,), -100, dtype=torch.long)
        bass_targets = torch.full((self.model_frames,), -100, dtype=torch.long)
        chord_pitch_targets = torch.zeros(self.model_frames, 25)

        for chord_segment in chord_segments:
            start_time = float(chord_segment["start_time"])
            end_time = float(chord_segment["end_time"])
            overlap_start = max(start_time, window_start_sec)
            overlap_end = min(end_time, window_end_sec)
            if overlap_end <= overlap_start:
                continue

            frame_start = int(
                math.floor(
                    (overlap_start - window_start_sec)
                    * self.sample_rate
                    / self.hop_length
                )
            )
            frame_end = int(
                math.ceil(
                    (overlap_end - window_start_sec)
                    * self.sample_rate
                    / self.hop_length
                )
            )
            frame_start = max(0, frame_start)
            frame_end = min(self.model_frames, frame_end)

            if frame_end > frame_start:
                quality_idx = self.quality_to_idx.get(
                    normalize_chord_quality(str(chord_segment["quality"])),
                    self.n_quality_idx,
                )
                root_chord_index = self._get_root_chord_index(
                    str(chord_segment["root"]),
                    quality_idx,
                )
                root_chord_targets[frame_start:frame_end] = root_chord_index
                bass_targets[frame_start:frame_end] = ROOT_TO_INDEX.get(
                    str(chord_segment["bass"]),
                    12,
                )
                chord_pitch_targets[frame_start:frame_end] = self._create_chord_vector(
                    str(chord_segment["root"]),
                    str(chord_segment["quality"]),
                    str(chord_segment["bass"]),
                )

            if window_start_sec <= start_time < window_end_sec:
                frame_index = int(
                    round(
                        (start_time - window_start_sec)
                        * self.sample_rate
                        / self.hop_length
                    )
                )
                if 0 <= frame_index < self.model_frames:
                    chord_boundary[frame_index] = 1.0

        return chord_boundary, root_chord_targets, bass_targets, chord_pitch_targets

    def _render_key_targets(
        self,
        *,
        key_segments: List[Dict[str, Any]],
        window_start_sec: float,
        window_end_sec: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """キーラベル列を frame 単位ターゲットへ展開する。"""
        key_boundary = torch.zeros(self.model_frames)
        key_targets = torch.full((self.model_frames,), -100, dtype=torch.long)

        for key_segment in key_segments:
            start_time = float(key_segment["start_time"])
            end_time = float(key_segment["end_time"])
            overlap_start = max(start_time, window_start_sec)
            overlap_end = min(end_time, window_end_sec)
            if overlap_end <= overlap_start:
                continue

            frame_start = int(
                math.floor(
                    (overlap_start - window_start_sec)
                    * self.sample_rate
                    / self.hop_length
                )
            )
            frame_end = int(
                math.ceil(
                    (overlap_end - window_start_sec)
                    * self.sample_rate
                    / self.hop_length
                )
            )
            frame_start = max(0, frame_start)
            frame_end = min(self.model_frames, frame_end)

            if frame_end > frame_start:
                key_targets[frame_start:frame_end] = parse_key_to_relative_major(
                    str(key_segment["key"])
                )

            if window_start_sec <= start_time < window_end_sec:
                frame_index = int(
                    round(
                        (start_time - window_start_sec)
                        * self.sample_rate
                        / self.hop_length
                    )
                )
                if 0 <= frame_index < self.model_frames:
                    key_boundary[frame_index] = 1.0

        return key_boundary, key_targets

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        rng = random.Random(self.seed + self.epoch * len(self.items) + idx)
        torchaudio = _require_torchaudio()

        info = inspect_audio(item["audio_path"])
        source_sample_rate = int(info.sample_rate)
        duration_sec = float(info.num_frames) / source_sample_rate

        # 窓の切り出し
        max_start = max(0.0, duration_sec - self.window_sec)
        window_start_sec = rng.uniform(0.0, max_start)
        window_end_sec = window_start_sec + self.window_sec

        # 1. 元の窓音声を読む。
        offset = int(round(window_start_sec * source_sample_rate))
        num_frames = int(round(self.window_sec * source_sample_rate))
        audio = self._load_audio_window(
            torchaudio=torchaudio,
            audio_path=Path(item["audio_path"]),
            frame_offset=offset,
            num_frames=num_frames,
        )

        # 2. 必要なら既存 `_pitch_±n` 素材で人工転調を合成する。
        (
            audio,
            chord_segments,
            key_segments,
            splice_frame,
        ) = self._maybe_apply_modulation(
            item=item,
            audio=audio,
            window_start_sec=window_start_sec,
            window_end_sec=window_end_sec,
            frame_offset=offset,
            num_frames=num_frames,
            rng=rng,
            torchaudio=torchaudio,
        )

        # 3. 最終的なラベル列を frame 単位ターゲットへ落とす。
        (
            chord_boundary,
            root_chord_targets,
            bass_targets,
            chord_pitch_targets,
        ) = self._render_chord_targets(
            chord_segments=chord_segments,
            window_start_sec=window_start_sec,
            window_end_sec=window_end_sec,
        )
        key_boundary, key_targets = self._render_key_targets(
            key_segments=key_segments,
            window_start_sec=window_start_sec,
            window_end_sec=window_end_sec,
        )
        inject_synthetic_boundaries(
            chord_boundary,
            splice_frame,
            force_boundary=self.modulation_config.force_boundary_for_chord,
        )
        inject_synthetic_boundaries(
            key_boundary,
            splice_frame,
            force_boundary=self.modulation_config.force_boundary_for_key,
        )

        return {
            "audio": audio,
            "chord_boundary": chord_boundary,
            "root_chord_targets": root_chord_targets,
            "bass_targets": bass_targets,
            "key_boundary": key_boundary,
            "key_targets": key_targets,
            "chord_pitch_targets": chord_pitch_targets,
            "song_name": item["song_name"],
        }


def chord_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "audio": torch.stack([item["audio"] for item in batch]),
        "chord_boundary": torch.stack([item["chord_boundary"] for item in batch]),
        "root_chord_targets": torch.stack(
            [item["root_chord_targets"] for item in batch]
        ),
        "bass_targets": torch.stack([item["bass_targets"] for item in batch]),
        "key_boundary": torch.stack([item["key_boundary"] for item in batch]),
        "key_targets": torch.stack([item["key_targets"] for item in batch]),
        "chord_pitch_targets": torch.stack(
            [item["chord_pitch_targets"] for item in batch]
        ),
        "song_name": [item["song_name"] for item in batch],
    }


# --- 損失関数 ---


class BalancedSoftmaxLoss(nn.Module):
    def __init__(
        self, class_counts: torch.Tensor, tau: float = 1.0, ignore_index: int = -100
    ):
        super().__init__()
        # 希少クラスの出現回数が極小のときに log_prior が負の無限大に爆発するのを防ぐため、最小値を 1.0 に制限します
        log_prior = torch.log(torch.clamp(class_counts, min=1.0))
        self.register_buffer("log_prior", log_prior)
        self.tau = tau
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.dim() > 2:
            logits = logits.reshape(-1, logits.size(-1))
            labels = labels.reshape(-1)
        valid = labels != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        logits, labels = logits[valid], labels[valid]
        adjusted_logits = logits + self.tau * self.log_prior
        return F.cross_entropy(adjusted_logits, labels)


class SafeCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid = labels != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        return F.cross_entropy(logits[valid], labels[valid])


class ShiftTolerantBCELoss(nn.Module):
    """
    少しずれたコード境界ラベルを許容する BCE loss。
    予測側を max-pooling し、正解フレーム周辺で最も強い予測に勾配を流す。
    """

    def __init__(self, pos_weight: float = 1.0, tolerance: int = 1):
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
        t = factor * self.tolerance
        if t == 0:
            return x
        return x[..., t : -t or None]

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        # preds, targets: (B, T)
        if preds.dim() == 2:
            preds = preds.unsqueeze(1)
        if targets.dim() == 2:
            targets = targets.unsqueeze(1)

        # 予測側を tolerance 分だけ広げ、端の不確かなフレームを落とす。
        spreaded_preds = self.crop(self.spread(preds))
        cropped_targets = self.crop(targets, factor=2)

        # 正解境界周辺の負例は見ない。
        look_at = cropped_targets + (1 - self.spread(targets, factor=2))
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            look_at = look_at * self.crop(mask, factor=2)

        return F.binary_cross_entropy_with_logits(
            spreaded_preds,
            cropped_targets,
            weight=look_at,
            pos_weight=self.pos_weight,
        )


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, gamma: float = 1.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if mask is not None:
            if mask.dim() == targets.dim() - 1:
                mask = mask.unsqueeze(-1)
            mask = mask.to(device=logits.device, dtype=logits.dtype)
            probs = probs * mask
            targets = targets * mask
        tp = (targets * probs).sum(dim=(0, 1))
        fp = ((1 - targets) * probs).sum(dim=(0, 1))
        fn = (targets * (1 - probs)).sum(dim=(0, 1))
        ti = (tp + self.smooth) / (
            tp + self.alpha * fp + (1 - self.alpha) * fn + self.smooth
        )
        return torch.pow(1 - ti.mean(), self.gamma)


# --- Head & Loss ---


class ChordHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_root_chord_classes: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 256

        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # 各種タスク用
        self.boundary = nn.Linear(hidden_dim, 1)
        self.root_chord = nn.Linear(hidden_dim, num_root_chord_classes)
        self.bass = nn.Linear(hidden_dim, 13)
        self.key_boundary = nn.Linear(hidden_dim, 1)
        self.key = nn.Linear(hidden_dim, 13)
        self.pitch = nn.Linear(hidden_dim, 25)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.shared(x)
        return {
            "chord_boundary_logits": self.boundary(h).squeeze(-1),
            "root_chord_logits": self.root_chord(h),
            "bass_logits": self.bass(h),
            "key_boundary_logits": self.key_boundary(h).squeeze(-1),
            "key_logits": self.key(h),
            "chord_pitch_logits": self.pitch(h),
        }


class ChordLoss(nn.Module):
    def __init__(self, config: ChordConfig, root_chord_counts: torch.Tensor):
        super().__init__()
        self.config = config
        self.chord_bce = ShiftTolerantBCELoss(
            pos_weight=config.boundary_pos_weight,
            tolerance=config.chord_boundary_loss_tolerance,
        )
        self.key_bce = ShiftTolerantBCELoss(
            pos_weight=config.key_boundary_pos_weight,
            tolerance=config.key_boundary_loss_tolerance,
        )
        self.rc_loss = BalancedSoftmaxLoss(root_chord_counts, tau=0.3)
        self.ce_loss = SafeCrossEntropyLoss(ignore_index=-100)
        self.ft_loss = FocalTverskyLoss(
            alpha=config.focal_tversky_alpha, gamma=config.focal_tversky_gamma
        )

    def forward(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        chord_boundary_mask = batch.get("chord_boundary_mask")
        if chord_boundary_mask is not None:
            chord_boundary_mask = chord_boundary_mask.to(
                outputs["chord_boundary_logits"].device
            )
        key_boundary_mask = batch.get("key_boundary_mask")
        if key_boundary_mask is not None:
            key_boundary_mask = key_boundary_mask.to(
                outputs["key_boundary_logits"].device
            )
        chord_pitch_mask = batch.get("chord_pitch_mask")
        if chord_pitch_mask is not None:
            chord_pitch_mask = chord_pitch_mask.to(outputs["chord_pitch_logits"].device)

        l_cb = self.chord_bce(
            outputs["chord_boundary_logits"],
            batch["chord_boundary"].to(outputs["chord_boundary_logits"].device),
            chord_boundary_mask,
        )
        l_rc = self.rc_loss(
            outputs["root_chord_logits"],
            batch["root_chord_targets"].to(outputs["root_chord_logits"].device),
        )
        l_ba = self.ce_loss(
            outputs["bass_logits"],
            batch["bass_targets"].to(outputs["bass_logits"].device),
        )
        l_kb = self.key_bce(
            outputs["key_boundary_logits"],
            batch["key_boundary"].to(outputs["key_boundary_logits"].device),
            key_boundary_mask,
        )
        l_ke = self.ce_loss(
            outputs["key_logits"],
            batch["key_targets"].to(outputs["key_logits"].device),
        )
        l_pi = self.ft_loss(
            outputs["chord_pitch_logits"],
            batch["chord_pitch_targets"].to(outputs["chord_pitch_logits"].device),
            chord_pitch_mask,
        )

        total = (
            l_cb * self.config.chord_boundary_weight
            + l_rc * self.config.root_chord_weight
            + l_ba * self.config.bass_weight
            + l_kb * self.config.key_boundary_weight
            + l_ke * self.config.key_weight
            + l_pi * self.config.chord_pitch_weight
        )
        return total, {
            "chord_total": total,
            "chord_boundary": l_cb,
            "root_chord": l_rc,
            "bass": l_ba,
            "key_boundary": l_kb,
            "key": l_ke,
            "chord_pitch": l_pi,
        }


def train_chord_batch(
    *,
    model: nn.Module,
    batch: Dict[str, Any],
    loss_fn: ChordLoss,
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
    total_loss, loss_dict = compute_chord_batch_loss(
        model=model,
        batch=batch,
        loss_fn=loss_fn,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        device=device,
        loss_scale=loss_scale,
    )

    optimizer.zero_grad(set_to_none=True)

    if scaler:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < scale_before_step
    else:
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        skipped = False

    if not skipped:
        scheduler.step()
        if ema_model is not None:
            ema_model.update(model)

    return total_loss.item(), loss_dict, skipped


def compute_chord_batch_loss(
    *,
    model: nn.Module,
    batch: Dict[str, Any],
    loss_fn: ChordLoss,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
    device: torch.device,
    loss_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }

    with torch.amp.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=use_amp,
    ):
        outputs = model(
            batch["audio"],
            include_amt=False,
            include_beat=False,
            include_chord=True,
        )
        total_loss, loss_dict = loss_fn(outputs, batch)
        total_loss = total_loss * loss_scale

    return total_loss, loss_dict
