"""楽器再判定モデル本体。

AMT と同じ backbone（CQT -> 音高クエリ特徴）を共有し、その上に「ノート単位で音色を
見るヘッド」を載せた構成。backbone は AMT の学習済み重みで初期化できる
（modeling/checkpoints.py の load_amt_backbone を参照）。

1 回の forward で扱うのは音声 1 窓 + その窓に属するノート列で、出力は 4 種類:

- note_logits / note_family_logits   … ノート 1 つずつの楽器・楽器ファミリー
- window_logits / window_family_logits … 窓全体としての楽器・ファミリー
- note_embedding / window_embedding  … 音色の埋め込み（クラスタリングと対照学習に使う）

ポイントは「AMT が付けた楽器 (program) をモデルに渡さないこと」。前段の誤りを引き継が
ないよう、音声とノートの時刻・音高だけから判断させる（_metadata を参照）。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...amt.modeling.backbone import AudioFeatureExtractor, V1Backbone
from ...taxonomy.instrument_classes import NUM_INSTRUMENT_CLASSES
from ..data.labels import FAMILY_NAMES, STEM_CONTEXT_NAMES


@dataclass(frozen=True)
class InstrumentRefinementConfig:
    """モデル構成。前半は AMT backbone と一致させる必要がある項目。

    sample_rate から pitch_query_count までは AMT のチェックポイントを流用する都合上、
    値がずれていると backbone を読み込めない（checkpoints.py が検証する）。
    note_hidden_size 以降がこのモデル固有のヘッド部分。
    """

    sample_rate: int = 22_050
    hop_length: int = 512
    cqt_fmin: float = 27.5
    cqt_n_bins: int = 312
    cqt_bins_per_octave: int = 36
    cqt_filter_scale: float = 0.475
    harmonics: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)
    harmonic_dropout_p: float = 0.0
    cqt_log_scale: bool = False
    input_audio_channels: int = 2
    hidden_size: int = 384
    base_ch: int = 64
    encoder_num_layers: int = 6
    encoder_num_heads: int = 12
    dropout: float = 0.1
    use_gradient_checkpoint: bool = True
    pitch_min: int = 21
    pitch_max: int = 108
    note_hidden_size: int = 256
    embedding_size: int = 128
    num_instrument_classes: int = NUM_INSTRUMENT_CLASSES
    num_family_classes: int = len(FAMILY_NAMES)
    num_stem_contexts: int = len(STEM_CONTEXT_NAMES)
    prior_dropout: float = 1.0
    # 1 ノートから特徴を拾う時刻。アタック（onset 前後のフレーム）、伸ばしている最中
    # （長さに対する割合）、減衰（offset 前後）の 3 種類をサンプリングする。
    # 音色の手がかりはアタックに集中するため onset 側を厚く取っている。
    onset_frame_offsets: tuple[int, ...] = (-2, -1, 0, 1, 2, 4, 8)
    sustain_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    release_frame_offsets: tuple[int, ...] = (-2, 0, 2)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.hop_length <= 0:
            raise ValueError("sample_rate and hop_length must be positive")
        if self.input_audio_channels != 2:
            raise ValueError("the shared V1 backbone requires stereo input")
        if self.pitch_min < 0 or self.pitch_max > 127 or self.pitch_max < self.pitch_min:
            raise ValueError("invalid MIDI pitch range")
        if self.note_hidden_size <= 0 or self.embedding_size <= 0:
            raise ValueError("head dimensions must be positive")
        if self.num_instrument_classes <= 0 or self.num_family_classes <= 0:
            raise ValueError("class counts must be positive")
        if not 0.0 <= self.prior_dropout <= 1.0:
            raise ValueError("prior_dropout must be in [0, 1]")
        if not self.onset_frame_offsets:
            raise ValueError("at least one onset frame offset is required")

    @property
    def pitch_query_count(self) -> int:
        return self.pitch_max - self.pitch_min + 1


def _coerce_config_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """JSON やチェックポイント由来の dict を config へ渡せる形に整える。

    知らないキーは捨て、JSON で list になってしまう項目を tuple へ戻す（config は
    frozen dataclass なので、tuple でないとハッシュや比較で扱いにくい）。
    """
    allowed = {field.name for field in fields(InstrumentRefinementConfig)}
    result = {key: value for key, value in values.items() if key in allowed}
    for key in (
        "harmonics",
        "onset_frame_offsets",
        "sustain_fractions",
        "release_frame_offsets",
    ):
        if key in result:
            result[key] = tuple(result[key])
    return result


def load_refinement_model_config(path: str | Path) -> InstrumentRefinementConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, Mapping):
        raise ValueError("instrument refinement model config must be a JSON object")
    return InstrumentRefinementConfig(**_coerce_config_values(raw))


class InstrumentRefinementModel(nn.Module):
    """Classify and embed AMT note queries against their separated audio.

    ノート 1 つを 1 クエリとして扱い、その発音区間の音響特徴を集めて楽器を判定する。
    backbone を差し替え可能にしてあるのはテスト用（軽量な偽 backbone を注入する）。
    """

    def __init__(
        self,
        config: InstrumentRefinementConfig,
        *,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if backbone is None:
            extractor = AudioFeatureExtractor(
                sampling_rate=config.sample_rate,
                hop_length=config.hop_length,
                cqt_fmin=config.cqt_fmin,
                cqt_n_bins=config.cqt_n_bins,
                cqt_bins_per_octave=config.cqt_bins_per_octave,
                cqt_filter_scale=config.cqt_filter_scale,
                harmonics=config.harmonics,
                harmonic_dropout_p=config.harmonic_dropout_p,
                cqt_log_scale=config.cqt_log_scale,
                input_audio_channels=config.input_audio_channels,
            )
            backbone = V1Backbone(
                feature_extractor=extractor,
                hidden_size=config.hidden_size,
                base_ch=config.base_ch,
                num_layers=config.encoder_num_layers,
                num_heads=config.encoder_num_heads,
                num_pitch_queries=config.pitch_query_count,
                use_global_token=False,
                dropout=config.dropout,
                use_gradient_checkpoint=config.use_gradient_checkpoint,
            )
        if not hasattr(backbone, "query_feature_dim"):
            raise ValueError("backbone must expose query_feature_dim")
        self.backbone = backbone
        backbone_dim = int(getattr(backbone, "query_feature_dim"))
        hidden = int(config.note_hidden_size)

        # --- ノートのメタ情報（音高・ステム種別・長さなど）を 1 本のクエリにする層 ---
        # prior_embedding は「+1」が未知(-1)用の枠。現状は常に未知を渡している。
        self.pitch_embedding = nn.Embedding(128, 32)
        self.prior_embedding = nn.Embedding(config.num_instrument_classes + 1, 24)
        self.stem_embedding = nn.Embedding(config.num_stem_contexts + 1, 16)
        self.scalar_encoder = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 32))
        metadata_dim = 32 + 24 + 16 + 32
        self.event_query = nn.Sequential(
            nn.Linear(metadata_dim, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
        )
        # --- ノート区間から拾った特徴 (backbone_dim + 音量 1 次元) をまとめる層 ---
        # local_attention はサンプリングした複数時点のうち、どこを重く見るかを決める。
        self.local_projection = nn.Linear(backbone_dim + 1, hidden)
        self.local_attention = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.note_fusion = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.timbre_projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.embedding_size),
        )
        # --- 窓全体（ノートに紐づかない音響全体）を見る層 ---
        self.global_class_projection = nn.Sequential(nn.Linear(backbone_dim, hidden), nn.GELU())
        self.global_embedding_projection = nn.Linear(backbone_dim, config.embedding_size)
        self.window_attention = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.window_fusion = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        # ヘッドはノート用と窓用で共有する（同じ意味の分類なので重みを分けない）。
        self.instrument_head = nn.Linear(hidden, config.num_instrument_classes)
        self.family_head = nn.Linear(hidden, config.num_family_classes)
        self.register_buffer(
            "onset_offsets",
            torch.tensor(config.onset_frame_offsets, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sustain_fractions",
            torch.tensor(config.sustain_fractions, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "release_offsets",
            torch.tensor(config.release_frame_offsets, dtype=torch.float32),
            persistent=False,
        )

    def _frame_valid_mask(
        self,
        *,
        batch_size: int,
        num_frames: int,
        valid_audio_frames: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """0 埋めした末尾を除いた、実データのあるフレームだけ True のマスクを作る。"""
        if valid_audio_frames is None:
            return torch.ones(batch_size, num_frames, dtype=torch.bool, device=device)
        lengths = (
            torch.ceil(valid_audio_frames.to(device=device, dtype=torch.float32) / float(self.config.hop_length))
            .long()
            .clamp(0, num_frames)
        )
        return torch.arange(num_frames, device=device)[None, :] < lengths[:, None]

    def _frame_energy(self, audio: torch.Tensor, *, num_frames: int) -> torch.Tensor:
        """フレームごとの音量を dBFS 相当から [-1, 1] へ潰した 1 次元特徴にする。

        アタックの立ち上がり方は楽器を見分ける手がかりになるので、backbone の音高特徴
        とは別に、生波形から直接求めた音量を足して渡す。
        """
        mono_power = audio.float().square().mean(dim=1, keepdim=True)
        energy = F.avg_pool1d(
            mono_power,
            kernel_size=max(1, int(self.config.hop_length)),
            stride=max(1, int(self.config.hop_length)),
            ceil_mode=True,
        )
        energy = F.interpolate(energy, size=num_frames, mode="linear", align_corners=False)
        db = 10.0 * torch.log10(energy.clamp_min(1e-10))
        return ((db + 100.0) / 50.0 - 1.0).clamp(-1.0, 1.0).transpose(1, 2)

    def _sample_positions(
        self,
        note_start_seconds: torch.Tensor,
        note_end_seconds: torch.Tensor,
    ) -> torch.Tensor:
        """各ノートについて、特徴を拾うフレーム位置（小数）を並べて返す。

        戻り値の形状は [B, ノート数, onset+sustain+release のサンプル数]。
        小数のまま返し、_gather_local 側で前後フレームを線形補間する。
        """
        frames_per_second = float(self.config.sample_rate) / float(self.config.hop_length)
        starts = note_start_seconds.float() * frames_per_second
        ends = note_end_seconds.float() * frames_per_second
        durations = (ends - starts).clamp_min(0.0)
        onset = starts[:, :, None] + self.onset_offsets.to(starts.device)[None, None]
        sustain = starts[:, :, None] + durations[:, :, None] * self.sustain_fractions.to(starts.device)[None, None]
        release = ends[:, :, None] + self.release_offsets.to(starts.device)[None, None]
        return torch.cat((onset, sustain, release), dim=-1)

    def _gather_local(
        self,
        pitch_features: torch.Tensor,
        energy_features: torch.Tensor,
        *,
        note_start_seconds: torch.Tensor,
        note_end_seconds: torch.Tensor,
        note_pitch: torch.Tensor,
        note_mask: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ノートごとに、その音高・その時刻の特徴を backbone の出力から抜き出す。

        Returns:
            (特徴 [B, ノート数, サンプル数, 特徴次元+1], 有効サンプルのマスク)。
            マスクは、窓の外や音声末尾を越えたサンプル点を False にする。
        """
        batch_size, num_frames, _, feature_dim = pitch_features.shape
        num_notes = int(note_pitch.shape[1])
        sample_count = (
            len(self.config.onset_frame_offsets)
            + len(self.config.sustain_fractions)
            + len(self.config.release_frame_offsets)
        )
        if num_notes == 0:
            return (
                pitch_features.new_zeros(batch_size, 0, sample_count, feature_dim + 1),
                torch.zeros(
                    batch_size,
                    0,
                    sample_count,
                    dtype=torch.bool,
                    device=pitch_features.device,
                ),
            )
        # サンプル位置の前後フレームを取り、alpha で線形補間する。
        positions = self._sample_positions(note_start_seconds, note_end_seconds)
        lower_raw = torch.floor(positions).long()
        upper_raw = lower_raw + 1
        alpha = (positions - lower_raw.float()).to(dtype=pitch_features.dtype)
        lower = lower_raw.clamp(0, max(0, num_frames - 1))
        upper = upper_raw.clamp(0, max(0, num_frames - 1))
        # backbone の音高クエリは pitch_min 始まりなので、MIDI ノート番号をずらす。
        pitch_index = (note_pitch - self.config.pitch_min).clamp(0, self.config.pitch_query_count - 1)
        batch_index = torch.arange(batch_size, device=pitch_features.device)[:, None, None]
        batch_index = batch_index.expand_as(lower)
        pitch_index = pitch_index[:, :, None].expand_as(lower)
        low_pitch = pitch_features[batch_index, lower, pitch_index]
        high_pitch = pitch_features[batch_index, upper, pitch_index]
        pitch_local = low_pitch + alpha[..., None] * (high_pitch - low_pitch)
        low_energy = energy_features[batch_index, lower]
        high_energy = energy_features[batch_index, upper]
        energy_local = low_energy + alpha[..., None].float() * (high_energy - low_energy)
        # 音高特徴に音量を 1 次元くっつける。
        local = torch.cat((pitch_local, energy_local.to(pitch_local.dtype)), dim=-1)
        # clamp で範囲内に丸めた分、はみ出したサンプル点はここで無効化する。
        valid_lengths = frame_valid_mask.long().sum(dim=1)
        local_mask = (
            note_mask[:, :, None] & (positions >= 0.0) & (positions < valid_lengths[:, None, None].to(positions.dtype))
        )
        return local, local_mask

    def _metadata(
        self,
        *,
        note_start_seconds: torch.Tensor,
        note_end_seconds: torch.Tensor,
        note_pitch: torch.Tensor,
        note_prior_class: torch.Tensor,
        note_confidence: torch.Tensor,
        stem_context_id: torch.Tensor,
        window_seconds: float,
    ) -> torch.Tensor:
        """ノートのメタ情報を 1 本のクエリベクトルへまとめる。"""
        # AMT programs are deliberately ignored: the refinement model must make
        # a fresh timbre decision from audio and note timing/pitch only.
        # 渡された note_prior_class は使わず、常に「未知」の枠へ潰している。
        safe_prior = torch.full_like(note_prior_class, self.config.num_instrument_classes)
        safe_stem = stem_context_id.clamp(0, self.config.num_stem_contexts)
        stem_values = self.stem_embedding(safe_stem)[:, None, :].expand(-1, note_pitch.shape[1], -1)
        # 連続値の特徴 4 つ: 音の長さ・窓内の相対位置・信頼度・音高（いずれも正規化済み）。
        duration = (note_end_seconds - note_start_seconds).clamp_min(0.0)
        relative_start = note_start_seconds / max(1e-6, float(window_seconds))
        scalar = torch.stack(
            (
                torch.log1p(duration),
                relative_start.clamp(-1.0, 2.0),
                note_confidence.clamp(0.0, 1.0),
                torch.log1p(note_pitch.float().clamp_min(0.0)) / math.log(128.0),
            ),
            dim=-1,
        )
        values = torch.cat(
            (
                self.pitch_embedding(note_pitch.clamp(0, 127)),
                self.prior_embedding(safe_prior),
                stem_values,
                self.scalar_encoder(scalar),
            ),
            dim=-1,
        )
        return self.event_query(values)

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """マスク付き softmax。有効要素が 1 つも無い行は NaN にせず 0 を返す。"""
        masked = scores.masked_fill(~mask, -torch.inf)
        has_value = mask.any(dim=-1, keepdim=True)
        safe = torch.where(has_value, masked, torch.zeros_like(scores))
        weights = torch.softmax(safe, dim=-1) * mask.to(scores.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(
        self,
        audio: torch.Tensor,
        *,
        note_start_seconds: torch.Tensor,
        note_end_seconds: torch.Tensor,
        note_pitch: torch.Tensor,
        note_prior_class: torch.Tensor,
        stem_context_id: torch.Tensor,
        note_confidence: torch.Tensor | None = None,
        note_mask: torch.Tensor | None = None,
        valid_audio_frames: torch.Tensor | None = None,
        window_seconds: float = 8.0,
        include_aux_outputs: bool = False,
    ) -> dict[str, torch.Tensor]:
        """音声 1 窓とその窓のノート列から、楽器ロジットと音色埋め込みを求める。

        Args:
            audio: [B, 2, T] のステレオ波形。
            note_*: 窓先頭を 0 秒とした各ノートの時刻・音高。ノート 0 個でも動く。
            note_prior_class: 受け取るが使わない（_metadata のコメントを参照）。
            valid_audio_frames: 0 埋め前の実サンプル数。末尾の窓で使う。
            include_aux_outputs: attention 重みなどデバッグ用の出力も返すか。
        """
        if audio.ndim != 3 or int(audio.shape[1]) != 2:
            raise ValueError("audio must have shape [B, 2, T]")
        if note_mask is None:
            note_mask = torch.ones_like(note_pitch, dtype=torch.bool)
        if note_confidence is None:
            note_confidence = torch.ones_like(note_start_seconds)
        # 1. backbone で音高クエリ特徴 [B, フレーム, 音高, 次元] を得る。
        backbone_output = self.backbone(audio)
        pitch_features = backbone_output.pitch_query_features
        if int(pitch_features.shape[2]) != self.config.pitch_query_count:
            raise ValueError("backbone pitch-query count does not match model config")
        frame_valid_mask = self._frame_valid_mask(
            batch_size=int(audio.shape[0]),
            num_frames=int(pitch_features.shape[1]),
            valid_audio_frames=valid_audio_frames,
            device=audio.device,
        )
        # 2. 各ノートの発音区間から特徴を抜き、メタ情報のクエリを作る。
        energy = self._frame_energy(audio, num_frames=int(pitch_features.shape[1]))
        local, local_mask = self._gather_local(
            pitch_features,
            energy,
            note_start_seconds=note_start_seconds,
            note_end_seconds=note_end_seconds,
            note_pitch=note_pitch,
            note_mask=note_mask,
            frame_valid_mask=frame_valid_mask,
        )
        event = self._metadata(
            note_start_seconds=note_start_seconds,
            note_end_seconds=note_end_seconds,
            note_pitch=note_pitch,
            note_prior_class=note_prior_class,
            note_confidence=note_confidence,
            stem_context_id=stem_context_id,
            window_seconds=window_seconds,
        )
        # 3. ノート内の複数サンプル点を attention で 1 本に集約する。
        #    メタ情報クエリを足してからスコアを出すので、「この音高・この長さの音なら
        #    どのタイミングを見るべきか」をノートごとに変えられる。
        projected = self.local_projection(local)
        attention_scores = self.local_attention(projected + event.unsqueeze(2)).squeeze(-1)
        local_attention = self._masked_softmax(attention_scores, local_mask)
        pooled_local = (projected * local_attention.unsqueeze(-1)).sum(dim=2)
        note_hidden = self.note_fusion(pooled_local + event)
        note_hidden = note_hidden * note_mask.unsqueeze(-1).to(note_hidden.dtype)
        # 音色埋め込みは分類とは別経路。メタ情報を足さず音響だけから作り、
        # コサイン類似度で比べられるよう正規化する。
        note_embedding = F.normalize(self.timbre_projection(pooled_local), dim=-1, eps=1e-8)
        note_embedding = note_embedding * note_mask.unsqueeze(-1).to(note_embedding.dtype)

        # 4. 窓全体の特徴を作る。backbone が band_features を出すならそれを、
        #    無ければ音高クエリ特徴を有効フレームで平均して代用する。
        band_features = getattr(backbone_output, "band_features", None)
        if isinstance(band_features, torch.Tensor) and band_features.ndim == 4:
            global_audio = band_features.mean(dim=(1, 2))
        else:
            per_frame = pitch_features.mean(dim=2)
            frame_weights = frame_valid_mask.unsqueeze(-1).to(per_frame.dtype)
            global_audio = (per_frame * frame_weights).sum(dim=1) / frame_weights.sum(dim=1).clamp_min(1.0)
        global_hidden = self.global_class_projection(global_audio)
        # 5. 窓の代表ベクトルは「ノートを attention で集約したもの + 音響全体」。
        #    ノートが 0 個の窓（無音など）では音響全体だけを使う。
        if note_hidden.shape[1] == 0:
            note_pool = torch.zeros_like(global_hidden)
            embedding_pool = self.global_embedding_projection(global_audio)
        else:
            window_scores = self.window_attention(note_hidden).squeeze(-1)
            window_attention = self._masked_softmax(window_scores, note_mask)
            note_pool = (note_hidden * window_attention.unsqueeze(-1)).sum(dim=1)
            embedding_pool = (note_embedding * window_attention.unsqueeze(-1)).sum(
                dim=1
            ) + self.global_embedding_projection(global_audio)
        window_hidden = self.window_fusion(note_pool + global_hidden)
        window_embedding = F.normalize(embedding_pool, dim=-1, eps=1e-8)
        outputs = {
            "note_logits": self.instrument_head(note_hidden),
            "window_logits": self.instrument_head(window_hidden),
            "note_family_logits": self.family_head(note_hidden),
            "window_family_logits": self.family_head(window_hidden),
            "note_embedding": note_embedding,
            "window_embedding": window_embedding,
            "frame_valid_mask": frame_valid_mask,
        }
        if include_aux_outputs:
            outputs.update(
                {
                    "local_attention": local_attention,
                    "note_hidden": note_hidden,
                    "window_hidden": window_hidden,
                }
            )
        return outputs
