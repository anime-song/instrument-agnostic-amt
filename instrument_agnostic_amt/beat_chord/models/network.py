from __future__ import annotations

from collections.abc import Callable

import einops
import torch
import torch.nn.functional as F
from torch import nn

from ..config import MidiFrameModelConfig
from ..heads.beat import BeatHead
from ..heads.chord import ChordHead
from .backbone import MidiFrameBackbone


class TaskFeatureAdapter(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.net(features)


class MidiFrameBeatChordModel(nn.Module):
    def __init__(self, config: MidiFrameModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = MidiFrameBackbone(config)
        self.beat_adapter = TaskFeatureAdapter(config.hidden_size, config.dropout)
        self.chord_adapter = TaskFeatureAdapter(config.hidden_size, config.dropout)
        self.beat_head = BeatHead(
            input_dim=config.hidden_size,
            num_meter_classes=config.num_meter_classes,
            hidden_dim=config.beat_head_hidden_dim,
            dropout=config.dropout,
        )
        self.chord_head = ChordHead(
            input_dim=config.hidden_size,
            num_root_chord_classes=config.num_root_chord_classes,
            hidden_dim=config.chord_head_hidden_dim,
            dropout=config.dropout,
        )

        # Self-Refinement 用のモジュール・パラメータの定義
        self.inter_refine_layers = list(config.inter_refine_layers)
        if len(self.inter_refine_layers) > 0:
            self.inter_global_up_convs = nn.ModuleDict()
            self.inter_global_token_merges = nn.ModuleDict()

            self.inter_beat_adapters = nn.ModuleDict()
            self.inter_chord_adapters = nn.ModuleDict()
            self.inter_beat_heads = nn.ModuleDict()
            self.inter_chord_heads = nn.ModuleDict()

            self.inter_beat_down_convs = nn.ModuleDict()
            self.inter_chord_down_convs = nn.ModuleDict()

            self.inter_beat_feedback_projs = nn.ModuleDict()
            self.inter_chord_feedback_projs = nn.ModuleDict()

            self.inter_feedback_gates_beat = nn.ParameterDict()
            self.inter_feedback_gates_chord = nn.ParameterDict()

            num_beat_classes = 2 + config.num_meter_classes
            num_chord_classes = config.num_root_chord_classes + 53

            for idx in self.inter_refine_layers:
                layer_str = str(idx)

                self.inter_global_up_convs[layer_str] = nn.ConvTranspose1d(
                    config.hidden_size,
                    config.hidden_size,
                    kernel_size=config.time_downsample_factor,
                    stride=config.time_downsample_factor,
                )
                self.inter_global_token_merges[layer_str] = nn.Linear(
                    config.hidden_size * config.num_global_tokens, config.hidden_size
                )

                # 予測用アダプターとヘッド
                self.inter_beat_adapters[layer_str] = TaskFeatureAdapter(
                    config.hidden_size, config.dropout
                )
                self.inter_chord_adapters[layer_str] = TaskFeatureAdapter(
                    config.hidden_size, config.dropout
                )

                self.inter_beat_heads[layer_str] = BeatHead(
                    input_dim=config.hidden_size,
                    num_meter_classes=config.num_meter_classes,
                    hidden_dim=config.beat_head_hidden_dim,
                    dropout=config.dropout,
                )
                self.inter_chord_heads[layer_str] = ChordHead(
                    input_dim=config.hidden_size,
                    num_root_chord_classes=config.num_root_chord_classes,
                    hidden_dim=config.chord_head_hidden_dim,
                    dropout=config.dropout,
                )

                # ダウンサンプリング用 Conv1d
                self.inter_beat_down_convs[layer_str] = nn.Conv1d(
                    num_beat_classes,
                    num_beat_classes,
                    kernel_size=config.time_downsample_factor,
                    stride=config.time_downsample_factor,
                )
                self.inter_chord_down_convs[layer_str] = nn.Conv1d(
                    num_chord_classes,
                    num_chord_classes,
                    kernel_size=config.time_downsample_factor,
                    stride=config.time_downsample_factor,
                )

                # 予測トークンへの射影
                self.inter_beat_feedback_projs[layer_str] = nn.Linear(
                    num_beat_classes, config.hidden_size
                )
                self.inter_chord_feedback_projs[layer_str] = nn.Linear(
                    num_chord_classes, config.hidden_size
                )

                # ゲートパラメータ (初期値を 0 付近にするために -3.0 で初期化)
                self.inter_feedback_gates_beat[layer_str] = nn.Parameter(
                    torch.tensor(-3.0)
                )
                self.inter_feedback_gates_chord[layer_str] = nn.Parameter(
                    torch.tensor(-3.0)
                )

    @staticmethod
    def _match_lowres_time_length(x: torch.Tensor, target_frames: int) -> torch.Tensor:
        if x.shape[1] < target_frames:
            return F.pad(x, (0, 0, 0, target_frames - x.shape[1]))
        return x[:, :target_frames, :]

    def _make_intermediate_callback(
        self,
        target_frames: int,
        include_beat: bool,
        include_chord: bool,
        feedback_beat: bool,
        feedback_chord: bool,
        detach_beat_feedback: bool,
        detach_chord_feedback: bool,
        include_aux_outputs: bool,
        intermediates_dict: dict[str, dict[str, torch.Tensor]],
    ) -> Callable[[torch.Tensor, int], torch.Tensor]:
        def callback(x: torch.Tensor, layer_idx: int) -> torch.Tensor:
            if layer_idx not in self.inter_refine_layers:
                return x

            layer_str = str(layer_idx)
            batch_size, time_steps_low, pitch_and_global, hidden_size = x.shape
            pitch_steps = pitch_and_global - self.backbone.num_global_tokens

            # 予測情報トークンを分離して元の pitch + global を取り出す (再帰的な結合を防ぐため)
            pure_token_count = pitch_steps + self.backbone.num_global_tokens
            x_pure = x[:, :, :pure_token_count, :]

            pitch_part = x_pure[:, :, :pitch_steps, :]  # [B, T_low, P, D]
            global_part = x_pure[:, :, pitch_steps:, :]  # [B, T_low, G, D]

            # 1. テンソルの時間方向のアップサンプリング
            # グローバルトークンのアップサンプリング
            upsampled_global = einops.rearrange(global_part, "b t g d -> (b g) d t")
            upsampled_global = self.inter_global_up_convs[layer_str](upsampled_global)
            upsampled_global = einops.rearrange(
                upsampled_global,
                "(b g) d t -> b t g d",
                b=batch_size,
                g=self.backbone.num_global_tokens,
            )
            highres_global_tokens = self.backbone._match_time_length(
                upsampled_global, target_frames
            )  # [B, T_high, G, D]

            # 高解像度グローバルトークンをマージして global_features とする
            highres_global_flat = einops.rearrange(
                highres_global_tokens, "b t g d -> b t (g d)"
            )
            global_features = self.inter_global_token_merges[layer_str](
                highres_global_flat
            )  # [B, T_high, D]

            # 2. 中間予測の実行
            refinement_tokens = []

            if feedback_beat:
                # 教師がある batch では通常の計算グラフを使い、教師がない相互 feedback では
                # head 側へ勾配を流さず、推定確率だけを条件トークンとして使う。
                if include_beat or not detach_beat_feedback:
                    beat_features = self.inter_beat_adapters[layer_str](global_features)
                    beat_outputs = self.inter_beat_heads[layer_str](beat_features)
                else:
                    with torch.no_grad():
                        beat_features = self.inter_beat_adapters[layer_str](
                            global_features
                        )
                        beat_outputs = self.inter_beat_heads[layer_str](beat_features)

                # ロス計算用に保存
                if include_beat and include_aux_outputs:
                    intermediates_dict[layer_str]["beat"] = beat_outputs

                # 確率計算
                beat_prob = torch.sigmoid(beat_outputs["beat_logits"]).unsqueeze(-1)
                downbeat_prob = torch.sigmoid(
                    beat_outputs["downbeat_logits"]
                ).unsqueeze(-1)
                meter_prob = torch.softmax(beat_outputs["meter_logits"], dim=-1)
                beat_probs_concat = torch.cat(
                    [beat_prob, downbeat_prob, meter_prob], dim=-1
                )  # [B, T_high, C_beat]
                if detach_beat_feedback:
                    beat_probs_concat = beat_probs_concat.detach()

                # ダウンサンプリング
                beat_probs_low = self.inter_beat_down_convs[layer_str](
                    beat_probs_concat.transpose(1, 2)
                ).transpose(1, 2)
                beat_probs_low = self._match_lowres_time_length(
                    beat_probs_low, time_steps_low
                )  # [B, T_low, C_beat]

                # 予測トークンへの射影
                beat_pred_token = self.inter_beat_feedback_projs[layer_str](
                    beat_probs_low
                )  # [B, T_low, D]
                # ゲートを掛ける
                gate = torch.sigmoid(self.inter_feedback_gates_beat[layer_str])
                beat_token = gate * beat_pred_token  # [B, T_low, D]
                refinement_tokens.append(beat_token.unsqueeze(2))  # [B, T_low, 1, D]

            if feedback_chord:
                # 教師がない batch の chord feedback は detach し、chord head は chord loss
                # だけで学習させる。
                if include_chord or not detach_chord_feedback:
                    chord_features = self.inter_chord_adapters[layer_str](
                        global_features
                    )
                    chord_outputs = self.inter_chord_heads[layer_str](chord_features)
                else:
                    with torch.no_grad():
                        chord_features = self.inter_chord_adapters[layer_str](
                            global_features
                        )
                        chord_outputs = self.inter_chord_heads[layer_str](
                            chord_features
                        )

                # ロス計算用に保存
                if include_chord and include_aux_outputs:
                    intermediates_dict[layer_str]["chord"] = chord_outputs

                # 確率計算
                chord_boundary_prob = torch.sigmoid(
                    chord_outputs["chord_boundary_logits"]
                ).unsqueeze(-1)
                root_chord_prob = torch.softmax(
                    chord_outputs["root_chord_logits"], dim=-1
                )
                bass_prob = torch.softmax(chord_outputs["bass_logits"], dim=-1)
                key_boundary_prob = torch.sigmoid(
                    chord_outputs["key_boundary_logits"]
                ).unsqueeze(-1)
                key_prob = torch.softmax(chord_outputs["key_logits"], dim=-1)
                chord_pitch_prob = torch.sigmoid(chord_outputs["chord_pitch_logits"])

                chord_probs_concat = torch.cat(
                    [
                        chord_boundary_prob,
                        root_chord_prob,
                        bass_prob,
                        key_boundary_prob,
                        key_prob,
                        chord_pitch_prob,
                    ],
                    dim=-1,
                )  # [B, T_high, C_chord]
                if detach_chord_feedback:
                    chord_probs_concat = chord_probs_concat.detach()

                # ダウンサンプリング
                chord_probs_low = self.inter_chord_down_convs[layer_str](
                    chord_probs_concat.transpose(1, 2)
                ).transpose(1, 2)
                chord_probs_low = self._match_lowres_time_length(
                    chord_probs_low, time_steps_low
                )  # [B, T_low, C_chord]

                # 予測トークンへの射影
                chord_pred_token = self.inter_chord_feedback_projs[layer_str](
                    chord_probs_low
                )  # [B, T_low, D]
                # ゲートを掛ける
                gate = torch.sigmoid(self.inter_feedback_gates_chord[layer_str])
                chord_token = gate * chord_pred_token  # [B, T_low, D]
                refinement_tokens.append(chord_token.unsqueeze(2))  # [B, T_low, 1, D]

            # 3. トークン結合
            if len(refinement_tokens) > 0:
                x = torch.cat([x_pure] + refinement_tokens, dim=2)
            else:
                x = x_pure

            return x

        return callback

    def forward(
        self,
        midi_frames: torch.Tensor,
        *,
        include_beat: bool,
        include_chord: bool,
        feedback_beat: bool | None = None,
        feedback_chord: bool | None = None,
        detach_beat_feedback: bool = False,
        detach_chord_feedback: bool = False,
        include_aux_outputs: bool = True,
    ) -> dict[str, torch.Tensor | dict[str, dict[str, torch.Tensor]]]:
        if not include_beat and not include_chord:
            raise ValueError("At least one task must be enabled")
        if feedback_beat is None:
            feedback_beat = include_beat
        if feedback_chord is None:
            feedback_chord = include_chord

        # 時間軸でランダムな位置を無音にするデータ拡張 (訓練時のみ)
        if self.training and self.config.time_mask_prob > 0.0:
            frames_to_mask = int(
                round(
                    self.config.time_mask_duration_ms
                    / 1000.0
                    * self.config.sample_rate
                    / self.config.hop_length
                )
            )
            B, C, T, P = midi_frames.shape
            if T > frames_to_mask:
                midi_frames = midi_frames.clone()
                for b in range(B):
                    if torch.rand(1).item() < self.config.time_mask_prob:
                        t_start = torch.randint(0, T - frames_to_mask, (1,)).item()
                        midi_frames[b, :, t_start : t_start + frames_to_mask, :] = 0.0

        intermediates_dict = {str(idx): {} for idx in self.inter_refine_layers}
        target_frames = int(midi_frames.shape[2])

        callback = None
        if len(self.inter_refine_layers) > 0:
            callback = self._make_intermediate_callback(
                target_frames=target_frames,
                include_beat=include_beat,
                include_chord=include_chord,
                feedback_beat=feedback_beat,
                feedback_chord=feedback_chord,
                detach_beat_feedback=detach_beat_feedback,
                detach_chord_feedback=detach_chord_feedback,
                include_aux_outputs=include_aux_outputs,
                intermediates_dict=intermediates_dict,
            )

        backbone_outputs = self.backbone(midi_frames, intermediate_callback=callback)
        outputs: dict[str, torch.Tensor | dict[str, dict[str, torch.Tensor]]] = {}
        if include_aux_outputs:
            outputs.update(
                {
                    "global_features": backbone_outputs.global_features,
                    "lowres_global_features": backbone_outputs.lowres_global_features,
                    "global_token_features": backbone_outputs.global_token_features,
                    "lowres_global_tokens": backbone_outputs.lowres_global_tokens,
                }
            )

        if include_beat:
            beat_features = self.beat_adapter(backbone_outputs.global_features)
            if include_aux_outputs:
                outputs["beat_features"] = beat_features
            outputs.update(self.beat_head(beat_features))

        if include_chord:
            chord_features = self.chord_adapter(backbone_outputs.global_features)
            if include_aux_outputs:
                outputs["chord_features"] = chord_features
            outputs.update(self.chord_head(chord_features))

        if include_aux_outputs and len(self.inter_refine_layers) > 0:
            outputs["intermediate_outputs"] = intermediates_dict

        return outputs
