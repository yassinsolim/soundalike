"""GPU + cuDNN inspection utilities.

This module is a learning tool as much as a practical one. It lets you *see*
how NVIDIA's libraries choose the low-level algorithm ("solver") for an
operation at runtime:

* cuDNN keeps several algorithms for a convolution (implicit GEMM, Winograd,
  FFT, ...). With the autotuner enabled (`torch.backends.cudnn.benchmark`), it
  times the candidates for your exact tensor shape and caches the winner.
* The selected algorithm shows up in the *name* of the CUDA kernel that runs,
  which we surface with `torch.profiler`.
* cuDNN can also emit its own heuristic/selection log via environment variables
  (see `enable_cudnn_logging`), which must be set before the first cuDNN call.

None of this modifies or risks the hardware — it is pure observability.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional


def enable_cudnn_logging(level: int = 2, dest: str = "stdout") -> None:
    """Ask cuDNN to log its API calls / algorithm selection.

    Must be called BEFORE torch initializes CUDA (before the first GPU op),
    otherwise cuDNN has already read these variables. Prefer setting the same
    variables in the shell for guaranteed effect.
    """
    os.environ["CUDNN_LOGLEVEL_DBG"] = str(level)
    os.environ["CUDNN_LOGDEST_DBG"] = dest
    # Older cuDNN toggle names (harmless if unused by newer versions).
    os.environ["CUDNN_LOGINFO_DBG"] = "1" if level >= 2 else "0"


def report_device() -> str:
    import torch

    if not torch.cuda.is_available():
        return "CUDA is not available (running on CPU)."
    i = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(i)
    props = torch.cuda.get_device_properties(i)
    lines = [
        f"Device            : {torch.cuda.get_device_name(i)}",
        f"Compute capability: sm_{cc[0]}{cc[1]}",
        f"Multiprocessors   : {props.multi_processor_count}",
        f"Total VRAM        : {props.total_memory / 1024**3:.1f} GiB",
        f"torch / CUDA      : {torch.__version__} / {torch.version.cuda}",
        f"cuDNN version     : {torch.backends.cudnn.version()}",
        f"Built arch list   : {', '.join(torch.cuda.get_arch_list())}",
    ]
    return "\n".join(lines)


@dataclass
class ConvSolverResult:
    shape: str
    autotune: bool
    ms_per_iter: float
    kernels: List[str]


def _event_device_us(e) -> float:
    """CUDA time for a profiler event, across PyTorch naming changes.

    PyTorch renamed `cuda_time_total` -> `device_time_total` in newer releases.
    """
    for attr in ("device_time_total", "cuda_time_total", "self_device_time_total"):
        val = getattr(e, attr, None)
        if val:
            return float(val)
    return 0.0


def _profile_conv(module, x, iters: int) -> ConvSolverResult:
    import torch
    from torch.profiler import ProfilerActivity, profile

    # Warm up so autotuning + allocation don't pollute the measurement.
    for _ in range(10):
        module(x)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            module(x)
        torch.cuda.synchronize()

    events = [e for e in prof.key_averages() if _event_device_us(e) > 0]
    # Kernels that look like the convolution implementation (algorithm is encoded
    # in the kernel name, e.g. implicit_gemm / winograd / wgrad / fft / xmma).
    conv_kernels = sorted(
        {e.key for e in events if any(
            tag in e.key.lower()
            for tag in ("conv", "gemm", "winograd", "fft", "cudnn", "wgrad", "scudnn", "xmma")
        )}
    )
    total_cuda_us = sum(_event_device_us(e) for e in events)
    return ConvSolverResult(
        shape=str(tuple(x.shape)),
        autotune=torch.backends.cudnn.benchmark,
        ms_per_iter=total_cuda_us / 1000.0 / iters,
        kernels=conv_kernels or sorted({e.key for e in events})[:5],
    )


def inspect_conv_solver(
    batch: int = 16,
    in_ch: int = 32,
    out_ch: int = 64,
    size: int = 128,
    kernel: int = 3,
    iters: int = 30,
    autotune: Optional[bool] = None,
) -> ConvSolverResult:
    """Run one conv configuration and report the CUDA kernel cuDNN selected."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for solver inspection.")
    if autotune is not None:
        torch.backends.cudnn.benchmark = autotune

    conv = torch.nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2).cuda().eval()
    x = torch.randn(batch, in_ch, size, size, device="cuda")
    with torch.no_grad():
        return _profile_conv(conv, x, iters)


def demo() -> None:
    """Show device info and how autotuning changes the selected conv algorithm."""
    import torch

    print(report_device())
    if not torch.cuda.is_available():
        return

    print("\n--- cuDNN convolution solver selection ---")
    for autotune in (False, True):
        res = inspect_conv_solver(autotune=autotune)
        mode = "autotuner ON " if autotune else "autotuner OFF"
        print(f"\n[{mode}] shape={res.shape}  ~{res.ms_per_iter:.3f} ms/iter")
        for k in res.kernels:
            print(f"    kernel: {k}")


if __name__ == "__main__":
    demo()
