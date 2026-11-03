"""Device-agnostic utilities for CUDA and NPU.

Provides unified VRAM tracking, cache management, and device availability
checks that work across ``torch.cuda`` and ``torch.npu`` backends.
"""

from __future__ import annotations

import torch


def is_device_available(device_type: str) -> bool:
    """Check whether a compute device type is available.

    Args:
        device_type: ``"cuda"`` or ``"npu"``.

    Returns:
        ``True`` if the device is present and usable.
    """
    if device_type == "cuda":
        return torch.cuda.is_available()
    if device_type == "npu":
        return hasattr(torch, "npu") and torch.npu.is_available()
    return False


def detect_device_type() -> str | None:
    """Return the first available compute device type.

    Returns:
        ``"cuda"``, ``"npu"``, or ``None`` if neither is available.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    return None


def get_vram_mb(device: torch.device | None = None) -> float:
    """Get peak VRAM / HBM usage in MiB on the given or active compute device.

    Args:
        device: Specific device to query, or ``None`` for the active device.

    Returns:
        Allocated memory in MiB, or 0.0 if no compute device is available.
    """
    if device is not None:
        if device.type == "cuda":
            return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "npu":
            return torch.npu.max_memory_allocated(device) / (1024 * 1024)
        return 0.0
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.npu.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_peak_memory(device: torch.device | None = None) -> None:
    """Reset peak memory tracking on the given or active compute device.

    Args:
        device: Specific device to reset, or ``None`` for the active device.
    """
    if device is not None:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        elif device.type == "npu":
            torch.npu.reset_peak_memory_stats(device)
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.reset_peak_memory_stats()


def empty_cache(device: torch.device | None = None) -> None:
    """Release unused cached memory on the given or active compute device.

    Args:
        device: Specific device to clear, or ``None`` for the active device.
    """
    if device is not None:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "npu":
            torch.npu.empty_cache()
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
