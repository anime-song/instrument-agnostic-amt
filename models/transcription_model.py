import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import einops
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any

try:
    from .spec_augment import SpecAugment
except ImportError:
    SpecAugment = None

try:
    from .transformer import RMSNorm, Transformer
    from .cqt import RecursiveCQT
except ImportError:
    from transformer import RMSNorm, Transformer
    from cqt import RecursiveCQT


class AudioFeatureExtractor(nn.Module):
    """
    音声波形から HCQT (Harmonic CQT) 特徴量を抽出するクラス。
    1つの大きなCQTを計算し、それを基準にシフトすることで各倍音の特徴量を抽出します。
    倍音成分のランダムドロップアウトもサポートします。
    """

    def __init__(
        self,
        sampling_rate: int,
        hop_length: int,
        cqt_fmin: float = 27.5,
        cqt_n_bins: int = 312,
        cqt_bins_per_octave: int = 36,
        cqt_filter_scale: float = 0.475,
        harmonics: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0),
        harmonic_dropout_p: float = 0.0,
        spec_augment_params: Optional[Dict[str, Any]] = None,
        peak_normalize_waveform: bool = False,
        cqt_log_scale: bool = False,
        **kwargs: Any,  # 後方互換性のため
    ):
        super().__init__()
        self.sampling_rate = int(sampling_rate)
        self.hop_length = int(hop_length)
        self.cqt_fmin = float(cqt_fmin)
        self.cqt_n_bins = int(cqt_n_bins)
        self.cqt_bins_per_octave = int(cqt_bins_per_octave)
        self.cqt_filter_scale = float(cqt_filter_scale)
        self.harmonics = tuple(float(h) for h in harmonics)
        self.harmonic_dropout_p = float(harmonic_dropout_p)
        self.peak_normalize_waveform = bool(peak_normalize_waveform)
        self.cqt_log_scale = bool(cqt_log_scale)

        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.cqt_n_bins <= 0:
            raise ValueError("cqt_n_bins must be positive")
        if not self.harmonics:
            raise ValueError("harmonics must be non-empty")
        if any(h <= 0.0 for h in self.harmonics):
            raise ValueError("harmonics must contain only positive values")

        self.input_audio_channels = 2
        self.num_harmonics = len(self.harmonics)
        self.num_audio_channels = self.input_audio_channels * self.num_harmonics
        self.n_bins = self.cqt_n_bins

        # 1つの巨大なCQTを計算するためのパラメータ
        self.min_h = min(self.harmonics)
        self.max_h = max(self.harmonics)
        self.fmin_large = self.cqt_fmin * self.min_h

        # 必要な最大ビン数を計算
        # max_h のときの最後のビン(cqt_n_bins - 1)をカバーできる必要がある
        self.n_bins_large = math.ceil(
            self.cqt_n_bins
            + self.cqt_bins_per_octave * math.log2(self.max_h / self.min_h)
        )

        # ナイキスト周波数を超えるビンはCQT計算から除外し、後でゼロ埋めする
        nyquist = self.sampling_rate / 2.0
        max_valid_bins = math.floor(
            self.cqt_bins_per_octave * math.log2(nyquist / self.fmin_large) + 1
        )
        self.actual_cqt_bins = min(self.n_bins_large, max_valid_bins)

        self.cqt = RecursiveCQT(
            sr=self.sampling_rate,
            hop_length=self.hop_length,
            fmin=self.fmin_large,
            n_bins=self.actual_cqt_bins,
            bins_per_octave=self.cqt_bins_per_octave,
            filter_scale=self.cqt_filter_scale,
        )

        # 各 harmonic のシフト量を計算 (ビン単位)
        # shift = B * log2(h / min_h)
        self.register_buffer(
            "harmonic_shifts",
            torch.tensor(
                [
                    self.cqt_bins_per_octave * math.log2(h / self.min_h)
                    for h in self.harmonics
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.spec_augment = (
            SpecAugment(**spec_augment_params)
            if spec_augment_params and SpecAugment is not None
            else None
        )

    def _normalize_spec(self, spec: torch.Tensor) -> torch.Tensor:
        reduce_dims = tuple(range(1, spec.ndim))
        mean = spec.mean(dim=reduce_dims, keepdim=True)
        std = spec.std(dim=reduce_dims, keepdim=True).clamp_min(1e-8)
        return (spec - mean) / std

    def _apply_spec_augment_to_large_cqt(
        self,
        large_cqt_spec: torch.Tensor,
    ) -> torch.Tensor:
        """
        絶対周波数側の CQT に SpecAugment を適用する。

        HCQT へ分解する前に大域 CQT 上でマスクすることで、
        別 harmonic channel への周波数リークを抑える。
        random fill を安全に使うため、一度サンプル単位で標準化してから
        SpecAugment をかけ、最後に元スケールへ戻す。
        """
        if self.spec_augment is None or not self.training:
            return large_cqt_spec

        # SpecAugment は 3D 入力時に [B, T, F] を想定している。
        large_cqt_bt_f = large_cqt_spec.transpose(1, 2)
        reduce_dims = tuple(range(1, large_cqt_bt_f.ndim))
        mean = large_cqt_bt_f.mean(dim=reduce_dims, keepdim=True)
        std = large_cqt_bt_f.std(dim=reduce_dims, keepdim=True).clamp_min(1e-8)

        normalized_large_cqt = (large_cqt_bt_f - mean) / std
        augmented_large_cqt, _ = self.spec_augment(normalized_large_cqt)
        restored_large_cqt = augmented_large_cqt * std + mean

        # CQT 振幅として使うので、数値誤差や random fill 由来の負値は切り落とす。
        return restored_large_cqt.transpose(1, 2).clamp_min(0.0)

    def forward(
        self,
        waveform: torch.Tensor,
    ) -> "BackboneContext":
        # waveform: [B, 2, T]
        if waveform.ndim != 3 or waveform.shape[1] != self.input_audio_channels:
            raise ValueError(
                f"waveform must have shape [B, {self.input_audio_channels}, T]"
            )

        # CQT の hop_length に応じた出力フレーム数を基準にする
        crop_length = math.ceil(waveform.shape[-1] / self.hop_length)
        batch_size = int(waveform.shape[0])

        waveform_features = waveform
        if self.peak_normalize_waveform:
            peak = waveform_features.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
            waveform_features = waveform_features / peak

        waveform_flat = einops.rearrange(waveform_features, "b c t -> (b c) t").float()

        # 1つの巨大なCQTを計算
        # large_cqt_spec: [B * audio_channels, F_large, T]
        large_cqt_spec = self.cqt(waveform_flat)

        # ナイキスト周波数を超えて除外した高周波ビンがある場合、ゼロパディングして次元を合わせる
        if self.actual_cqt_bins < self.n_bins_large:
            pad_amount = self.n_bins_large - self.actual_cqt_bins
            large_cqt_spec = F.pad(large_cqt_spec, (0, 0, 0, pad_amount))

        # HCQT へ分解する前に、絶対周波数側の CQT へ SpecAugment を適用する。
        large_cqt_spec = self._apply_spec_augment_to_large_cqt(large_cqt_spec)

        F_large = large_cqt_spec.shape[1]

        specs = []
        base_bins = torch.arange(
            self.cqt_n_bins, device=large_cqt_spec.device, dtype=large_cqt_spec.dtype
        )

        for i, h in enumerate(self.harmonics):
            shift = self.harmonic_shifts[i]
            p = base_bins + shift
            p = p.clamp(0, F_large - 1)
            p0 = torch.floor(p).long()
            p1 = (p0 + 1).clamp(max=F_large - 1)
            alpha = (p - p0).unsqueeze(0).unsqueeze(-1)  # [1, cqt_n_bins, 1]

            val0 = large_cqt_spec[:, p0, :]  # [B, cqt_n_bins, T]
            val1 = large_cqt_spec[:, p1, :]  # [B, cqt_n_bins, T]

            # 線形補間でシフト後のCQTを取得
            hcqt_spec = val0 + alpha * (val1 - val0)

            if self.cqt_log_scale:
                hcqt_spec = torch.log(hcqt_spec + 1e-8)

            specs.append(hcqt_spec)

        # [B, audio_channels, num_harmonics, cqt_n_bins, T]
        spec = torch.stack(specs, dim=1)
        spec = einops.rearrange(
            spec,
            "(b c) h f t -> b c h f t",
            b=batch_size,
            c=self.input_audio_channels,
        )
        spec = self._normalize_spec(spec)

        # Harmonic Dropout: 学習時のみ。h=1.0 (基本波) はドロップアウト対象から除外する
        if self.training and self.harmonic_dropout_p > 0.0:
            keep_mask = (
                torch.rand(
                    batch_size,
                    1,
                    self.num_harmonics,
                    1,
                    1,
                    device=spec.device,
                )
                >= self.harmonic_dropout_p
            )
            fundamental_mask = torch.tensor(
                [abs(h - 1.0) <= 1e-5 for h in self.harmonics],
                device=spec.device,
                dtype=torch.bool,
            ).view(1, 1, self.num_harmonics, 1, 1)
            spec = spec * (keep_mask | fundamental_mask).to(dtype=spec.dtype)

        # Stereo channels and harmonics are treated as Backbone input channels.
        spec = einops.rearrange(spec, "b c h f t -> b (c h) f t").contiguous()

        spec = spec.to(waveform.dtype)

        # Backbone が想定する [B, C, T, F] 形式に変換
        spec = einops.rearrange(spec, "b c f t -> b c t f").contiguous()

        return BackboneContext(
            spec=spec,
            crop_length=crop_length,
        )


@dataclass
class BackboneContext:
    spec: torch.Tensor
    crop_length: int


@dataclass
class BackboneOutput:
    """Backbone の出力を band / pitch query に分離して保持する。"""

    band_features: torch.Tensor  # [B, num_bands, T, D]
    global_features: Optional[torch.Tensor]  # [B, T, D] or None
    pitch_query_features: torch.Tensor  # [B, T, num_pitch_queries, D]
    lowres_band_features: torch.Tensor  # [B, T/8, num_bands, D]
    lowres_global_features: Optional[torch.Tensor]  # [B, T/8, D] or None
    lowres_pitch_query_features: torch.Tensor  # [B, T/8, num_pitch_queries, D]


def checkpoint(
    module: nn.Module,
    x: torch.Tensor,
    *,
    use_checkpoint: bool,
    **kwargs: Any,
) -> torch.Tensor:
    if not use_checkpoint:
        return module(x, **kwargs)

    def forward_fn(tensor: torch.Tensor) -> torch.Tensor:
        return module(tensor, **kwargs)

    return torch.utils.checkpoint.checkpoint(
        forward_fn,
        x,
        use_reentrant=False,
    )


class StemConv(nn.Module):
    """
    入力:
        x: [B, in_ch, T, F]

    出力:
        y: [B, 4 * base_ch, T/8, F/4]
    """

    def __init__(
        self,
        in_ch: int,
        base_ch: int,
        kernel_size: int = 3,
        n_bins: int = 312,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv2d(in_ch, base_ch, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(base_ch, base_ch, kernel_size=5, padding=2)
        self.freq_embed = nn.Parameter(torch.zeros(1, base_ch, 1, n_bins))
        nn.init.normal_(self.freq_embed, std=0.02)

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=base_ch,
                out_channels=base_ch * 2,
                kernel_size=(kernel_size, kernel_size),
                stride=(2, 1),
                padding=(pad, pad),
            ),
            nn.GroupNorm(4, base_ch * 2),
            nn.GELU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=base_ch * 2,
                out_channels=base_ch * 4,
                kernel_size=(kernel_size, kernel_size),
                stride=(2, 2),
                padding=(pad, pad),
            ),
            nn.GroupNorm(4, base_ch * 4),
            nn.GELU(),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=base_ch * 4,
                out_channels=base_ch * 4,
                kernel_size=(kernel_size, kernel_size),
                stride=(2, 2),
                padding=(pad, pad),
            ),
            nn.GroupNorm(4, base_ch * 4),
            nn.GELU(),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=base_ch * 4,
                out_channels=base_ch * 4,
                kernel_size=(kernel_size, kernel_size),
                padding=(pad, pad),
            ),
            nn.GroupNorm(4, base_ch * 4),
        )

        self.out_ch = base_ch * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x) + self.freq_embed[:, :, :, : x.shape[-1]]
        x = self.conv2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


class PixelUnshuffleStem(nn.Module):
    """
    入力:
        x: [B, in_ch, T, F]

    出力:
        y: [B, 4 * base_ch, ceil(T/8), ceil(F/4)]
    """

    def __init__(
        self,
        in_ch: int,
        base_ch: int,
        n_bins: int = 312,
        time_downsample: int = 8,
        freq_downsample: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_ch <= 0:
            raise ValueError("in_ch must be positive")
        if base_ch <= 0:
            raise ValueError("base_ch must be positive")
        if time_downsample <= 0 or freq_downsample <= 0:
            raise ValueError("downsample factors must be positive")

        self.time_downsample = int(time_downsample)
        self.freq_downsample = int(freq_downsample)
        self.out_ch = base_ch * 4
        self.out_freq_bins = math.ceil(n_bins / self.freq_downsample)

        patch_dim = in_ch * self.time_downsample * self.freq_downsample
        self.proj = nn.Conv2d(patch_dim, self.out_ch, kernel_size=1)
        self.norm = nn.GroupNorm(4, self.out_ch)
        self.dropout = nn.Dropout(dropout)
        self.freq_embed = nn.Parameter(
            torch.zeros(1, self.out_ch, 1, self.out_freq_bins)
        )
        nn.init.normal_(self.freq_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, time_steps, freq_bins = x.shape
        pad_t = (-time_steps) % self.time_downsample
        pad_f = (-freq_bins) % self.freq_downsample
        if pad_t > 0 or pad_f > 0:
            x = F.pad(x, (0, pad_f, 0, pad_t))

        # Anisotropic pixel unshuffle for [T, F] -> [T/8, F/4].
        x = einops.rearrange(
            x,
            "b c (t pt) (f pf) -> b (c pt pf) t f",
            pt=self.time_downsample,
            pf=self.freq_downsample,
        )
        x = self.proj(x) + self.freq_embed[:, :, :, : x.shape[-1]]
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class PitchQueryEmbedding(nn.Module):
    def __init__(self, num_pitch=88, dim=160):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, midi_pitches):
        # midi_pitches: [P], e.g. 21..108
        x = midi_pitches.float()
        feat = torch.stack(
            [
                x / 128.0,
                torch.sin(2 * math.pi * x / 12.0),
                torch.cos(2 * math.pi * x / 12.0),
                x.floor() * 0.0 + 1.0,
            ],
            dim=-1,
        )
        return self.mlp(feat)


class Backbone(nn.Module):
    def __init__(
        self,
        feature_extractor: AudioFeatureExtractor,
        hidden_size: int,
        base_ch: int,
        output_dim: Optional[int] = None,
        num_layers: int = 1,
        num_heads: int = 8,
        num_pitch_queries: int = 88,
        pitch_query_expansion_size: int = 4,
        use_global_token: bool = False,
        dropout: float = 0.1,
        use_gradient_checkpoint: bool = True,
    ):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if base_ch <= 0:
            raise ValueError("base_ch must be positive")
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        if num_pitch_queries <= 0:
            raise ValueError("num_pitch_queries must be positive")

        self.feature_extractor = feature_extractor
        self.hidden_size = hidden_size
        self.base_ch = base_ch
        self.output_dim = output_dim if output_dim is not None else hidden_size
        self.use_gradient_checkpoint = use_gradient_checkpoint
        self.num_pitch_queries = num_pitch_queries
        self.use_global_token = bool(use_global_token)

        self.num_audio_channels = feature_extractor.num_audio_channels
        self.n_bins = feature_extractor.n_bins
        self.model_dim = hidden_size

        # Stem は時間方向を 1/8 に、周波数方向を 1/4 に圧縮する。
        self.stem = StemConv(
            in_ch=self.num_audio_channels,
            base_ch=base_ch,
            n_bins=self.n_bins,
            dropout=dropout,
        )

        # semi-CRF に渡す pitch-wise feature を作るための MIDI pitch 条件付き query token
        self.pitch_query_embed = PitchQueryEmbedding(
            num_pitch=num_pitch_queries,
            dim=base_ch * 4,
        )
        self.register_buffer(
            "midi_pitches",
            torch.arange(21, 21 + num_pitch_queries, dtype=torch.float32),
            persistent=False,
        )

        self.band_type_embed = nn.Parameter(torch.zeros(1, 1, 1, base_ch * 4))
        self.global_token = (
            nn.Parameter(torch.zeros(1, 1, 1, base_ch * 4))
            if self.use_global_token
            else None
        )
        self.global_type_embed = (
            nn.Parameter(torch.zeros(1, 1, 1, base_ch * 4))
            if self.use_global_token
            else None
        )
        self.pitch_type_embed = nn.Parameter(torch.zeros(1, 1, 1, base_ch * 4))

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            time_roformer = Transformer(
                input_dim=base_ch * 4,
                head_dim=hidden_size // num_heads,
                num_layers=1,
                num_heads=num_heads,
                ffn_hidden_size_factor=4,
                dropout=dropout,
            )
            band_roformer = Transformer(
                input_dim=base_ch * 4,
                head_dim=hidden_size // num_heads,
                num_layers=1,
                num_heads=num_heads,
                ffn_hidden_size_factor=4,
                dropout=dropout,
            )
            self.layers.append(nn.ModuleList([time_roformer, band_roformer]))

        self.stem_dim = base_ch * pitch_query_expansion_size  # Transformer の入出力次元
        self.query_feature_dim = self.stem_dim
        self.final_norm = RMSNorm(self.stem_dim)

        self.up_conv = nn.ConvTranspose1d(
            self.stem_dim,
            self.stem_dim,
            kernel_size=8,
            stride=8,
        )
        self.global_up_conv = (
            nn.ConvTranspose1d(
                self.stem_dim,
                self.stem_dim,
                kernel_size=8,
                stride=8,
            )
            if self.use_global_token
            else None
        )

    @staticmethod
    def _match_time_length(x: torch.Tensor, target_T: int) -> torch.Tensor:
        """x の時間次元（dim=2）を target_T に合わせる。"""
        if x.shape[2] < target_T:
            return F.pad(x, (0, 0, 0, target_T - x.shape[2]))
        return x[:, :, :target_T]

    def forward(
        self,
        waveform: torch.Tensor,
        context: Optional[BackboneContext] = None,
    ) -> BackboneOutput:
        if context is None:
            context = self.feature_extractor(waveform)

        use_checkpoint = (
            self.use_gradient_checkpoint and self.training and torch.is_grad_enabled()
        )

        # mel 特徴量を stem で時間圧縮し、周波数方向は band token に切る。
        # stem 出力: [B, D_stem, T/8, F'] → rearrange → [B, T/8, F', D_stem]
        stem_features = self.stem(context.spec)
        x = einops.rearrange(stem_features, "b d t f -> b t f d")

        B, T, num_bands, D = x.shape

        pitch_query = self.pitch_query_embed(self.midi_pitches)  # [P, D]
        pitch_query = pitch_query.unsqueeze(0).unsqueeze(0)  # [1, 1, P, D]
        pitch_query = pitch_query.expand(B, T, -1, -1)  # [B, T, P, D]

        # [B, T, num_bands + (global) + P, D]
        x = x + self.band_type_embed
        global_token = None
        if self.use_global_token:
            if self.global_token is None or self.global_type_embed is None:
                raise RuntimeError("global token parameters must be initialized")
            global_token = self.global_token.expand(B, T, -1, -1)
            global_token = global_token + self.global_type_embed
        pitch_query = pitch_query + self.pitch_type_embed
        tokens = [x]
        if global_token is not None:
            tokens.append(global_token)
        tokens.append(pitch_query)
        x = torch.cat(tokens, dim=2)

        for time_roformer, band_roformer in self.layers:
            B, T, K, D = x.shape
            # バンド軸Transformer
            x = x.reshape(B * T, K, D)
            x = checkpoint(band_roformer, x, use_checkpoint=use_checkpoint)
            x = x.reshape(B, T, K, D)

            # 時間軸Transformer
            x = einops.rearrange(x, "b t k d -> (b k) t d")
            x = checkpoint(time_roformer, x, use_checkpoint=use_checkpoint)
            x = einops.rearrange(x, "(b k) t d -> b t k d", k=K)

        x = self.final_norm(x)

        # band / global / pitch query を分離
        band_part = x[:, :, :num_bands, :]  # [B, T, num_bands, D]
        pitch_start = num_bands
        global_part = None
        if self.use_global_token:
            global_part = x[:, :, pitch_start : pitch_start + 1, :]
            pitch_start += 1
        pitch_part = x[:, :, pitch_start:, :]  # [B, T, P, D]

        # pitch_part のみアップサンプリング
        B, T, P, D = pitch_part.shape
        pitch_part = einops.rearrange(pitch_part, "b t p d -> (b p) d t")
        pitch_part = self.up_conv(pitch_part)
        pitch_part = einops.rearrange(pitch_part, "(b p) d t -> b p t d", b=B, p=P)

        target_T = context.crop_length
        # STFT/mel の center 処理に合わせて最終長をラベル側に揃える。
        pitch_part = self._match_time_length(pitch_part, target_T)
        # pitch_part: [B, P, T_out, D]

        lowres_global_features = None
        if global_part is not None:
            lowres_global_features = global_part.squeeze(2).contiguous()

        global_features = None
        if global_part is not None:
            if self.global_up_conv is None:
                raise RuntimeError("global_up_conv must be initialized")
            global_part = einops.rearrange(global_part, "b t g d -> (b g) d t")
            global_part = self.global_up_conv(global_part)
            global_part = einops.rearrange(
                global_part, "(b g) d t -> b g t d", b=B, g=1
            )
            global_part = self._match_time_length(global_part, target_T)
            global_features = global_part.squeeze(1).contiguous()

        return BackboneOutput(
            band_features=band_part.permute(0, 2, 1, 3).contiguous(),
            global_features=global_features,
            pitch_query_features=pitch_part.permute(0, 2, 1, 3).contiguous(),
            lowres_band_features=band_part.contiguous(),
            lowres_global_features=lowres_global_features,
            lowres_pitch_query_features=x[:, :, pitch_start:, :].contiguous(),
        )
