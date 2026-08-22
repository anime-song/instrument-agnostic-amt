from __future__ import annotations

from collections.abc import Sequence

import torch


def _mps_is_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def resolve_device(
    preference: str | torch.device | None = "auto",
) -> torch.device:
    """推論に使うデバイスを解決する。"""
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


def copy_tensors_to_cpu_once(
    tensors: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """同じdevice・dtypeの推論出力を連結し、device-to-host転送を1回にする。"""
    values = tuple(tensors)
    if not values:
        return ()
    if values[0].device.type == "cpu":
        return tuple(value.cpu() for value in values)

    source_device = values[0].device
    source_dtype = values[0].dtype
    if any(
        value.device != source_device or value.dtype != source_dtype
        for value in values[1:]
    ):
        raise ValueError("tensors must share a device and dtype")

    shapes = tuple(tuple(value.shape) for value in values)
    sizes = tuple(value.numel() for value in values)
    flattened = torch.cat([value.reshape(-1) for value in values]).cpu()
    return tuple(
        chunk.reshape(shape)
        for chunk, shape in zip(flattened.split(sizes), shapes)
    )


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
