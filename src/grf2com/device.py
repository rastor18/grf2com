"""Device selection: cuda -> xpu -> mps -> cpu, so moving to an NVIDIA GPU needs no code change."""
from __future__ import annotations

import torch


def xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def mps_available() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if xpu_available():
        return torch.device("xpu")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def available_devices() -> list[str]:
    out = ["cpu"]
    if torch.cuda.is_available():
        out.append("cuda")
    if xpu_available():
        out.append("xpu")
    if mps_available():
        out.append("mps")
    return out


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    if device.type == "xpu":
        return f"xpu ({torch.xpu.get_device_name(device)})"
    return device.type
