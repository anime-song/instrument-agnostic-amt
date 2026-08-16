from __future__ import annotations

import torch


def _mps_is_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def resolve_device(
    preference: str | torch.device | None = "auto",
) -> torch.device:
    """推論デバイスを CUDA、MPS、CPU の順で解決する。"""
    requested = "auto" if preference is None else str(preference).strip().lower()
    if not requested or requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available")
    if device.type == "mps" and not _mps_is_available():
        raise RuntimeError("MPS was requested, but it is not available")
    return device


def is_amp_supported(device: torch.device | str) -> bool:
    """推論時のautocastを利用できるデバイスか返す。"""
    return torch.device(device).type in {"cuda", "mps"}


def cuda_supports_native_bfloat16() -> bool:
    """CUDAデバイスがBF16をエミュレーションなしで実行できるか返す。"""
    return torch.cuda.is_bf16_supported(including_emulation=False)


def empty_device_cache(device: torch.device | str) -> None:
    """選択したアクセラレータの未使用キャッシュを解放する。"""
    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps":
        torch.mps.empty_cache()


def resolve_amp_dtype(
    device: torch.device | str,
    dtype_name: str | None,
) -> torch.dtype:
    """AMPの明示指定、またはデバイスごとの既定dtypeを解決する。"""
    target_device = torch.device(device)
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name is not None:
        raise ValueError(f"Unsupported AMP dtype: {dtype_name}")
    if target_device.type == "cuda" and cuda_supports_native_bfloat16():
        return torch.bfloat16
    if target_device.type == "cpu":
        return torch.float32
    return torch.float16


def maybe_compile_forward(
    model: torch.nn.Module,
    *,
    enabled: bool,
    mode: str = "default",
) -> torch.nn.Module:
    """backbone内のTransformerだけを必要時にコンパイルする。"""
    if not enabled:
        return model
    backbone = getattr(model, "backbone", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise ValueError("Regional compile requires model.backbone.layers")
    targets = tuple(module for pair in layers for module in pair)
    if not targets:
        raise ValueError("Regional compile found no Transformer modules")
    for target in targets:
        target.compile(
            backend="inductor",
            mode=mode,
            fullgraph=False,
            dynamic=None,
        )
    return model
