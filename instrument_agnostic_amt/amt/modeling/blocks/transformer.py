"""Transformer blocks used by the AMT backbone."""

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import einops
from typing import Tuple, Optional


def choose_low_precision_dtype() -> torch.dtype:
    """
    GPU の機能を調べて BF16 → FP16 → FP32 の順に
    最も高速な演算 dtype を返す。
    """
    if not torch.cuda.is_available():
        return torch.float32  # CPU 実行なら FP32 一択

    # Ampere (sm80) 以降ならほぼ BF16 演算に対応
    if torch.cuda.is_bf16_supported():  # PyTorch 2.1+
        return torch.bfloat16

    major_cc, _ = torch.cuda.get_device_capability()
    # Pascal (sm60) 以降なら FP16 演算ユニットあり
    if major_cc >= 6:
        return torch.float16

    return torch.float32  # それ以前の Maxwell など


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 5.960464477539063e-08):  # 0x1p-24
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        l2_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        denom = torch.maximum(l2_norm, torch.full_like(l2_norm, self.eps))
        normalized_x = x / denom
        return normalized_x * self.scale * self.gamma


class RoPE(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        """
        標準 Rotary Position Embedding (RoPE)
        Args:
            dim (int): 埋め込み次元（偶数）。
            base (float): RoPE の基数（例: 10000）。
        """
        super().__init__()
        assert dim % 2 == 0, "dim は偶数を想定しています"
        self.dim = dim
        self.base = float(base)

        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)  # module.to(device) に追従

    def _get_cos_sin_emb(self, pos: torch.Tensor, dtype: torch.dtype):
        """
        pos: (B, T) の絶対位置（0..T-1）
        返り値: cos/sin ともに (B, 1, T, D/2)
        """
        angles = torch.einsum("bt,j->btj", pos.to(self.inv_freq.dtype), self.inv_freq)
        cos = angles.cos().unsqueeze(1).to(dtype)
        sin = angles.sin().unsqueeze(1).to(dtype)
        return cos, sin

    def _apply_rotary_emb(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        x: (B, H, T, D), cos/sin: (B, 1, T, D/2)
        偶奇ペアごとに 2D 回転して元の並びにインタリーブで戻す。
        """
        x_even = x[..., 0::2]  # (B, H, T, D/2)
        x_odd = x[..., 1::2]  # (B, H, T, D/2)

        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_odd * cos + x_even * sin

        y = torch.empty_like(x)
        y[..., 0::2] = rot_even
        y[..., 1::2] = rot_odd
        return y

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """
        q, k: (B, H, T, D)
        """
        B, _, Tq, Dq = q.shape
        _, _, Tk, Dk = k.shape
        assert Dq == self.dim and Dk == self.dim, (
            "q/k の最終次元は dim と一致させてください。"
        )

        # 標準RoPE: pad を考慮せず絶対位置を使用
        pos_q = (
            torch.arange(Tq, device=q.device, dtype=self.inv_freq.dtype)
            .unsqueeze(0)
            .expand(B, -1)
        )
        pos_k = (
            torch.arange(Tk, device=k.device, dtype=self.inv_freq.dtype)
            .unsqueeze(0)
            .expand(B, -1)
        )

        q_cos, q_sin = self._get_cos_sin_emb(pos_q, dtype=q.dtype)
        k_cos, k_sin = self._get_cos_sin_emb(pos_k, dtype=k.dtype)

        q_rot = self._apply_rotary_emb(q, q_cos, q_sin)
        k_rot = self._apply_rotary_emb(k, k_cos, k_sin)
        return q_rot, k_rot


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_hidden_size_factor=4, dropout=0.0):
        super().__init__()
        dim_inner = int(dim * ffn_hidden_size_factor)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads=8,
        head_dim=64,
        dropout=0.0,
        use_rope: bool = True,
    ):
        super().__init__()
        self.hidden_size = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.norm_q = RMSNorm(input_dim)
        self.norm_context = RMSNorm(input_dim)

        self.to_q = nn.Linear(input_dim, self.hidden_size, bias=True)
        self.to_k = nn.Linear(input_dim, self.hidden_size, bias=True)
        self.to_v = nn.Linear(input_dim, self.hidden_size, bias=True)

        self.to_gates = nn.Linear(input_dim, num_heads)
        self.to_out = nn.Sequential(
            nn.Linear(self.hidden_size, input_dim),
            nn.Dropout(dropout),
        )

        self.use_rope = use_rope
        self.rope = RoPE(dim=self.head_dim, base=10000.0) if use_rope else None
        self.lowp_dtype = choose_low_precision_dtype()

    def forward(
        self,
        x,
        context=None,
        is_causal: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        attention_bias_alpha: Optional[torch.Tensor] = None,
    ):
        if context is None:
            context = x

        x = self.norm_q(x)
        context = self.norm_context(context)

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # 形状: (B, H, T, D)
        q, k, v = (
            einops.rearrange(q, "b tq (h d) -> b h tq d", h=self.num_heads),
            einops.rearrange(k, "b tk (h d) -> b h tk d", h=self.num_heads),
            einops.rearrange(v, "b tk (h d) -> b h tk d", h=self.num_heads),
        )

        if self.rope is not None:
            q, k = self.rope(q, k)

        if self.training:
            q = q.to(self.lowp_dtype)
            k = k.to(self.lowp_dtype)
            v = v.to(self.lowp_dtype)

        if attention_bias is not None:
            if attention_mask is not None or is_causal:
                raise ValueError(
                    "attention_bias is only supported for non-causal attention without attention_mask."
                )

            bias = attention_bias.to(device=q.device, dtype=q.dtype)
            if bias.dim() != 2:
                raise ValueError(
                    f"attention_bias must be 2D [Tq, Tk], got shape {tuple(bias.shape)}"
                )
            if bias.shape != (q.shape[2], k.shape[2]):
                raise ValueError(
                    f"attention_bias shape {tuple(bias.shape)} must match attention scores {(q.shape[2], k.shape[2])}"
                )

            if attention_bias_alpha is None:
                alpha = torch.tensor(1.0, device=q.device, dtype=q.dtype)
            elif isinstance(attention_bias_alpha, torch.Tensor):
                alpha = attention_bias_alpha.to(device=q.device, dtype=q.dtype)
            else:
                alpha = torch.tensor(
                    float(attention_bias_alpha), device=q.device, dtype=q.dtype
                )

            # additive bias は SDPA の float attn_mask に落とす。
            attn_mask = (alpha * bias).unsqueeze(0).unsqueeze(0)
            with sdpa_kernel(
                [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ]
            ):
                fetched = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, is_causal=False
                )
        else:
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.to(device=q.device)
                if attn_mask.dtype != torch.bool:
                    attn_mask = attn_mask != 0
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
                elif attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(1)

            # is_causal=True と attn_mask の同時指定はsdpaでエラーになるため、
            # 両方必要な場合は因果マスクを手動で結合する
            if is_causal and attn_mask is not None:
                seq_len = q.shape[2]
                causal_mask = torch.ones(
                    seq_len, seq_len, device=q.device, dtype=torch.bool
                ).tril()
                attn_mask = attn_mask & causal_mask.unsqueeze(0).unsqueeze(0)
                is_causal = False

            with sdpa_kernel(
                [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ]
            ):
                fetched = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, is_causal=is_causal
                )

        gates = self.to_gates(x)
        gates = gates.sigmoid()

        out = fetched.float() * einops.rearrange(gates, "b t h -> b h t 1")
        out = einops.rearrange(out, "b h t d -> b t (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_hidden_size_factor: int = 4,
        dropout: float = 0.0,
        use_cross_attention: bool = False,
        output_norm: bool = False,
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.layers = nn.ModuleList([])

        for _ in range(num_layers):
            self_attention = MultiHeadAttention(
                input_dim=input_dim,
                head_dim=head_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            blocks = [self_attention]

            if use_cross_attention:
                cross_attn = MultiHeadAttention(
                    input_dim=input_dim,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_rope=False,  # Cross attentionではQ/Kの系列が異なるためRoPEを使わない
                )
                blocks.append(cross_attn)

            ffn = FeedForward(
                dim=input_dim,
                ffn_hidden_size_factor=ffn_hidden_size_factor,
                dropout=dropout,
            )
            blocks.append(ffn)

            self.layers.append(nn.ModuleList(blocks))

        self.norm = RMSNorm(input_dim) if output_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None,
        causal_self_attn: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        attention_bias_alpha: Optional[torch.Tensor] = None,
    ):
        for blocks in self.layers:
            self_attn = blocks[0]
            residual = x
            x = self_attn(
                x,
                is_causal=causal_self_attn,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                attention_bias_alpha=attention_bias_alpha,
            )
            x = x + residual

            idx = 1
            if self.use_cross_attention:
                assert context is not None, (
                    "use_cross_attention=True の場合は context を与えてください"
                )
                cross_attn = blocks[idx]
                residual = x
                x = cross_attn(
                    x,
                    context,
                    is_causal=False,
                    attention_mask=context_attention_mask,
                )
                x = x + residual
                idx += 1

            residual = x
            ffn = blocks[idx]
            x = ffn(x) + residual

        x = self.norm(x)
        return x
