# RFC: SGLang SRT Platform Abstraction System

**Status**: Draft
**Date**: 2026-03-04
**Inspired by**: [`sglang/multimodal_gen/runtime/platforms/`](python/sglang/multimodal_gen/runtime/platforms/)

---

## Abstract

This RFC proposes porting the **`Platform` abstraction pattern** already used in
`sglang/multimodal_gen/runtime/platforms/` into the SRT core runtime
(`python/sglang/srt/`).

The result is a single `current_platform` singleton that encapsulates every
hardware-specific decision, replacing the current scatter of `is_npu()` / `is_hip()`
guards, hardcoded dictionaries, and giant if-elif factories across 10+ files.

Third-party hardware vendors (Moore Threads, Enflame, Cambricon, …) point one
environment variable at their class (`SGLANG_PLATFORM=my_pkg.MyPlatform`) and ship a
single Python package. **No upstream files need to be modified.**

---

## Motivation

### Current Pain Points

SGLang's SRT runtime supports CUDA, ROCm, Intel XPU, Ascend NPU, Habana HPU, and
CPU inference. However, **each integration required modifying 9+ upstream files**.
The Ascend NPU example:

| File | What had to change |
|---|---|
| `utils/common.py` | Add `is_npu()` detection function |
| `server_args.py` | Inject hardware-specific defaults |
| `layers/attention/attention_registry.py` | Register attention backend |
| `model_executor/model_runner.py` | Add device init branch + graph runner entry |
| `model_executor/model_runner_kv_cache_mixin.py` | Insert NPU branch in 200-line KV pool factory |
| `layers/quantization/__init__.py` | Register quantization methods |
| `layers/moe/utils.py` | Add MoE backend enum values |
| `environ.py` | Add NPU environment variables |
| `distributed/parallel_state.py` | Register communication backend |

The same cost applies to every future hardware vendor.

### Root Cause

Hardware knowledge is **scattered**. Every subsystem re-implements its own switch:

```python
# model_runner.py  (~line 2027) — hardcoded dict
device_to_graph_runner = {
    "cuda": CudaGraphRunner, "npu": NPUGraphRunner, "cpu": CPUGraphRunner,
}

# utils/common.py — each added when a new device appeared
def is_npu(): ...
def is_hip(): ...
def is_xpu(): ...

# model_runner_kv_cache_mixin.py — 200 lines of hardware-specific if-elif
if self.server_args.attention_backend == "ascend":
    if self.use_mla_backend:
        self.token_to_kv_pool = NPUMLATokenToKVPool(...)
    ...
elif ...:
    ...
```

### The Fix Already Exists in This Repo

`sglang/multimodal_gen/runtime/platforms/` solves exactly this problem: a `Platform`
class + `current_platform` singleton. The pattern is:

```python
from sglang.srt.platforms import current_platform

graph_runner_cls = current_platform.get_graph_runner_class()
kv_pool          = current_platform.create_kv_pool(runner, ...)
comm_cls         = current_platform.get_device_communicator_cls()
current_platform.initialize_device(rank, local_rank)
```

**One object. One import. No scattered device guards.**

---

## Goals

- **G1** — A hardware vendor ships one Python package and sets
  `SGLANG_PLATFORM=my_pkg.MyPlatform`. No upstream file changes needed.
- **G2** — All hardware decisions (attention, KV pool, graph runner, communicator,
  quantization, MoE) live in the `Platform` class, not scattered across 10+ files.
- **G3** — Built-in platforms (CUDA, ROCm, NPU, XPU, HPU, CPU) continue to work
  unchanged; this is a pure extension, not a rewrite.
- **G4** — The platform API is stable enough for vendors to maintain their own
  release cadence independently of SGLang upstream.

## Non-Goals

- Not changing the Scheduler core logic.
- Not changing the model registry (`SGLANG_EXTERNAL_MODEL_PACKAGE` stays).
- Not supporting hot-reload of platforms at runtime (startup-time load only).
- Not breaking existing `AttentionBackend` method signatures.

---

## Background

### The `multimodal_gen` Pattern

`multimodal_gen/runtime/platforms/` (adapted from vLLM) uses:

1. **`Platform` base class** (`interface.py`) — all hardware methods live here.
2. **Concrete subclasses** per device — `cuda.py`, `rocm.py`, `npu.py`, `musa.py`, …
3. **Lazy `current_platform` singleton** (`__init__.py`) — resolved on first access
   via `__getattr__`, supports out-of-tree override via `SGLANG_PLATFORM` env var.
4. **`PlatformEnum.OOT`** — an explicit enum value for out-of-tree platforms.

This RFC brings the identical structure to SRT's runtime, adding the SRT-specific
extension points (`create_kv_pool`, `get_graph_runner_class`, etc.) that have no
equivalent in the diffusion subsystem.

### Current Built-in Backend Landscape

| Device | Graph Runner | Attention Backends | Communicator |
|---|---|---|---|
| CUDA (NVIDIA) | `CudaGraphRunner` | flashinfer, fa3, fa4, cutlass_mla, trtllm_* | PyNCCL |
| ROCm (AMD) | `CudaGraphRunner` | aiter, wave, flashinfer | PyNCCL |
| NPU (Ascend) | `NPUGraphRunner` | ascend | NPUCommunicator |
| XPU (Intel) | `CudaGraphRunner` | intel_xpu, intel_amx | XPUCommunicator |
| HPU (Habana) | — | torch_native | HPUCommunicator |
| CPU | `CPUGraphRunner` | torch_native, triton | — |

All of these become implementations of `Platform` subclasses.

---

## Proposed Design

### Directory Layout

```
python/sglang/srt/platforms/        ← mirrors multimodal_gen/runtime/platforms/
├── __init__.py                     discovery logic + current_platform singleton
├── interface.py                    Platform base class + PlatformEnum
├── cuda.py                         NVIDIA CUDA
├── rocm.py                         AMD ROCm
├── npu.py                          Huawei Ascend NPU  (wraps existing hardware_backend/npu/)
├── xpu.py                          Intel XPU
├── hpu.py                          Habana HPU
└── cpu.py                          CPU-only
```

Out-of-tree vendors ship their own package; no file is added to this directory.

---

### `interface.py` — `Platform` Base Class

```python
# python/sglang/srt/platforms/interface.py
from __future__ import annotations
import enum
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.model_executor.model_runner import ModelRunner


class PlatformEnum(enum.Enum):
    CUDA  = enum.auto()
    ROCM  = enum.auto()
    NPU   = enum.auto()
    XPU   = enum.auto()
    HPU   = enum.auto()
    CPU   = enum.auto()
    OOT   = enum.auto()   # Out-of-Tree; used by all third-party vendors


class Platform:
    """
    Single abstraction point for all hardware-specific behaviour in SRT.

    Built-in platforms: add a subclass in sglang/srt/platforms/<device>.py
    and a detection function in __init__.py.

    Out-of-tree platforms: subclass Platform in your own package, set
    _enum = PlatformEnum.OOT, and export the qualified name via
    SGLANG_PLATFORM=<your.module.YourPlatform>.
    """

    # ── Class-level identity ──────────────────────────────────────────────────
    _enum: PlatformEnum
    device_name: str      # e.g. "cuda", "npu", "musa"  — used in torch.device()
    device_type: str      # torch device type string
    dispatch_key: str = "CPU"   # PyTorch dispatch key

    # ── Type predicates (lru_cache: evaluated once per process) ──────────────
    @lru_cache(maxsize=1)
    def is_cuda(self)        -> bool: return self._enum == PlatformEnum.CUDA
    @lru_cache(maxsize=1)
    def is_rocm(self)        -> bool: return self._enum == PlatformEnum.ROCM
    @lru_cache(maxsize=1)
    def is_npu(self)         -> bool: return self._enum == PlatformEnum.NPU
    @lru_cache(maxsize=1)
    def is_xpu(self)         -> bool: return self._enum == PlatformEnum.XPU
    @lru_cache(maxsize=1)
    def is_hpu(self)         -> bool: return self._enum == PlatformEnum.HPU
    @lru_cache(maxsize=1)
    def is_cpu(self)         -> bool: return self._enum == PlatformEnum.CPU
    @lru_cache(maxsize=1)
    def is_out_of_tree(self) -> bool: return self._enum == PlatformEnum.OOT

    @lru_cache(maxsize=1)
    def is_cuda_alike(self) -> bool:
        """True for CUDA and ROCm (both use the torch 'cuda' device namespace)."""
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)

    # ── Hardware introspection ────────────────────────────────────────────────
    def get_device_name(self, device_id: int = 0) -> str:
        raise NotImplementedError

    def get_device_total_memory(self, device_id: int = 0) -> int:
        """Total device memory in bytes."""
        raise NotImplementedError

    def get_device_capability(self, device_id: int = 0):
        """(major, minor) compute capability tuple, or None."""
        return None

    def get_available_memory(self, device_id: int = 0) -> float:
        """Available device memory in GiB."""
        raise NotImplementedError

    # ── Lifecycle hooks ───────────────────────────────────────────────────────
    def initialize_device(self, rank: int, local_rank: int) -> None:
        """
        Called once per worker process before model loading.

        Replaces scattered:
            if _is_npu:  init_npu_backend()
            elif _is_xpu: init_xpu_backend()
        """
        pass

    def apply_server_args_defaults(self, args: "ServerArgs") -> None:
        """
        Inject hardware-specific ServerArgs defaults.
        Called at the end of ServerArgs.__post_init__().
        Only override values the user left as None.
        """
        pass

    # ── Attention ─────────────────────────────────────────────────────────────
    def get_attention_backends(self) -> Dict[str, Callable]:
        """
        Return {backend_name: factory_fn} for all attention backends this
        platform provides.
          factory_fn signature: (runner: ModelRunner) -> AttentionBackend

        These are merged into attention_registry.ATTENTION_BACKENDS at startup.
        Built-in names already in the registry are never overwritten.
        """
        return {}

    def get_default_attention_backend(self) -> Optional[str]:
        """
        The preferred attention backend name when --attention-backend is not set.
        Return None to keep SGLang's existing auto-selection logic.
        """
        return None

    # ── Execution graph ───────────────────────────────────────────────────────
    def get_graph_runner_class(self) -> Optional[Type]:
        """
        The GraphRunner subclass for this platform.

        Replaces the hardcoded dict:
            {"cuda": CudaGraphRunner, "npu": NPUGraphRunner, ...}

        Return None to fall back to the built-in dict (backward compat).
        """
        return None

    # ── KV cache memory ───────────────────────────────────────────────────────
    def create_kv_pool(self, runner: "ModelRunner", **kwargs):
        """
        Instantiate and return the KV cache memory pool.

        Replaces the 200-line if-elif factory in model_runner_kv_cache_mixin.py.
        Return None to fall back to the built-in selection logic (CUDA/CPU/Mamba/NSA).
        """
        return None

    def get_allocator_class(self) -> Optional[Type]:
        """Custom KV cache allocator class, or None for the built-in."""
        return None

    # ── Communication ─────────────────────────────────────────────────────────
    def get_device_communicator_cls(self) -> Optional[str]:
        """
        Fully-qualified class name of the GroupCoordinator subclass for
        collective communication (NCCL, HCCL, MCCL, …).
        Return None to use the default NCCL path.
        """
        return None

    # ── Quantization & MoE ────────────────────────────────────────────────────
    def get_quantization_methods(self) -> Dict[str, Type]:
        """{method_name: QuantizationConfig subclass} — hardware-specific additions."""
        return {}

    def get_moe_a2a_backends(self) -> Dict[str, str]:
        """Additional MoE all-to-all backend names."""
        return {}

    def get_disaggregation_backends(self) -> Dict[str, Type]:
        """Custom KV-transfer backends for disaggregated prefill/decode."""
        return {}
```

---

### `__init__.py` — Discovery & `current_platform` Singleton

Identical structure to `multimodal_gen/runtime/platforms/__init__.py`:
**lazy initialisation via `__getattr__`**, priority-ordered detection, and explicit
override via `SGLANG_PLATFORM`.

```python
# python/sglang/srt/platforms/__init__.py
import logging, os, traceback
from sglang.srt.platforms.interface import Platform, PlatformEnum  # noqa: F401

logger = logging.getLogger(__name__)


# ── Built-in detection functions ─────────────────────────────────────────────
# Each returns a fully-qualified Platform subclass name, or None.

def _rocm_plugin() -> str | None:
    try:
        import amdsmi
        amdsmi.amdsmi_init()
        found = len(amdsmi.amdsmi_get_processor_handles()) > 0
        amdsmi.amdsmi_shut_down()
        if found:
            return "sglang.srt.platforms.rocm.RocmPlatform"
    except Exception:
        pass
    return None

def _cuda_plugin() -> str | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        found = pynvml.nvmlDeviceGetCount() > 0
        pynvml.nvmlShutdown()
        if found:
            return "sglang.srt.platforms.cuda.CudaPlatform"
    except Exception:
        # Jetson fallback: CUDA without NVML
        import os as _os
        if _os.path.isfile("/etc/nv_tegra_release"):
            return "sglang.srt.platforms.cuda.CudaPlatform"
    return None

def _npu_plugin() -> str | None:
    try:
        import torch
        if torch.npu.is_available():
            return "sglang.srt.platforms.npu.NpuPlatform"
    except Exception:
        pass
    return None

def _xpu_plugin() -> str | None:
    try:
        import torch
        if torch.xpu.is_available():
            return "sglang.srt.platforms.xpu.XpuPlatform"
    except Exception:
        pass
    return None

def _hpu_plugin() -> str | None:
    try:
        import torch
        if torch.hpu.is_available():
            return "sglang.srt.platforms.hpu.HpuPlatform"
    except Exception:
        pass
    return None

def _cpu_plugin() -> str | None:
    return "sglang.srt.platforms.cpu.CpuPlatform"   # always succeeds


# Detection order: first match wins.
# ROCm must precede CUDA — ROCm machines expose CUDA-compatible devices.
_BUILTIN_PLUGINS = [
    _rocm_plugin,
    _cuda_plugin,
    _npu_plugin,
    _xpu_plugin,
    _hpu_plugin,
    _cpu_plugin,    # final fallback
]


def _resolve_platform_cls_qualname() -> str:
    # 1. Explicit override (OOT vendors use this exclusively)
    if qualname := os.environ.get("SGLANG_PLATFORM", "").strip():
        logger.info(f"[Platform] Using SGLANG_PLATFORM={qualname}")
        return qualname

    # 2. Auto-detection
    for plugin in _BUILTIN_PLUGINS:
        if qualname := plugin():
            return qualname

    raise RuntimeError(
        "No SRT platform detected. "
        "Set SGLANG_PLATFORM=<fully.qualified.ClassName> or install "
        "the appropriate hardware support package."
    )


# ── Lazy singleton (same pattern as multimodal_gen) ──────────────────────────
_current_platform: Platform | None = None
_init_trace: str = ""
current_platform: Platform          # declared for type checkers; resolved below


def __getattr__(name: str):
    if name == "current_platform":
        # Lazy init: out-of-tree platforms must be able to do
        #   `from sglang.srt.platforms import Platform`
        # *before* current_platform is resolved, so we cannot resolve it at
        # import time.
        global _current_platform
        if _current_platform is None:
            from sglang.srt.utils.common import resolve_obj_by_qualname
            qualname = _resolve_platform_cls_qualname()
            _current_platform = resolve_obj_by_qualname(qualname)()
            global _init_trace
            _init_trace = "".join(traceback.format_stack())
            logger.info(f"[Platform] Activated: {_current_platform.device_name}")
        return _current_platform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Platform", "PlatformEnum", "current_platform", "_init_trace"]
```

---

### Built-in Platform Implementations (sketches)

**`cuda.py`** — NVIDIA GPU

```python
# python/sglang/srt/platforms/cuda.py
from sglang.srt.platforms.interface import Platform, PlatformEnum

class CudaPlatform(Platform):
    _enum        = PlatformEnum.CUDA
    device_name  = "cuda"
    device_type  = "cuda"
    dispatch_key = "CUDA"

    def get_device_name(self, device_id=0):
        import torch; return torch.cuda.get_device_name(device_id)

    def get_device_total_memory(self, device_id=0):
        import torch; return torch.cuda.get_device_properties(device_id).total_memory

    def get_device_capability(self, device_id=0):
        import torch; return torch.cuda.get_device_capability(device_id)

    def get_available_memory(self, device_id=0):
        import torch
        free, _ = torch.cuda.mem_get_info(device_id)
        return free / (1 << 30)

    def get_attention_backends(self):
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
        from sglang.srt.layers.attention.triton_attn_backend import TritonAttnBackend
        # fa3, fa4, cutlass_mla, trtllm_*, torch_native, flex_attention …
        return {
            "flashinfer":   lambda r: FlashInferAttnBackend(r),
            "triton":       lambda r: TritonAttnBackend(r),
            # …
        }

    def get_default_attention_backend(self):
        return "flashinfer"

    def get_graph_runner_class(self):
        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
        return CudaGraphRunner

    def get_device_communicator_cls(self):
        return "sglang.srt.distributed.device_communicators.pynccl.PyNcclCommunicator"
```

**`npu.py`** — Huawei Ascend (wraps existing `hardware_backend/npu/`, zero NPU code deleted)

```python
# python/sglang/srt/platforms/npu.py
from sglang.srt.platforms.interface import Platform, PlatformEnum

class NpuPlatform(Platform):
    _enum        = PlatformEnum.NPU
    device_name  = "npu"
    device_type  = "npu"
    dispatch_key = "NPU"

    def initialize_device(self, rank, local_rank):
        from sglang.srt.hardware_backend.npu.utils import init_npu_backend
        init_npu_backend()                              # existing logic, same call

    def apply_server_args_defaults(self, args):
        from sglang.srt.hardware_backend.npu.utils import set_default_server_args
        set_default_server_args(args)                   # existing logic, same call

    def get_attention_backends(self):
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
        return {"ascend": lambda r: AscendAttnBackend(r)}

    def get_default_attention_backend(self):
        return "ascend"

    def get_graph_runner_class(self):
        from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
        return NPUGraphRunner

    def create_kv_pool(self, runner, **kwargs):
        if runner.use_mla_backend:
            from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
            return NPUMLATokenToKVPool(runner, **kwargs)
        from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
        return NPUMHATokenToKVPool(runner, **kwargs)

    def get_device_communicator_cls(self):
        return "sglang.srt.distributed.device_communicators.npu_communicator.NpuCommunicator"
```

---

### Out-of-Tree Platform (vendor ships this)

```python
# sglang_musa_plugin/platform.py   — published as sglang-musa-plugin on PyPI

from sglang.srt.platforms import Platform, PlatformEnum

class MusaPlatform(Platform):
    _enum        = PlatformEnum.OOT   # OOT for all third-party vendors
    device_name  = "musa"
    device_type  = "musa"
    dispatch_key = "MUSA"

    def initialize_device(self, rank, local_rank):
        import torch_musa
        torch_musa.set_device(local_rank)

    def apply_server_args_defaults(self, args):
        if args.attention_backend is None:
            args.attention_backend = "musa_flash"
        args.disable_custom_all_reduce = True

    def get_attention_backends(self):
        from sglang_musa_plugin.attention import MusaFlashAttnBackend
        return {"musa_flash": lambda r: MusaFlashAttnBackend(r)}

    def get_default_attention_backend(self):
        return "musa_flash"

    def get_graph_runner_class(self):
        from sglang_musa_plugin.graph_runner import MusaGraphRunner
        return MusaGraphRunner

    def create_kv_pool(self, runner, **kwargs):
        from sglang_musa_plugin.memory_pool import MusaMHAPool
        return MusaMHAPool(runner, **kwargs)

    def get_device_communicator_cls(self):
        return "sglang_musa_plugin.communicator.MCCLCommunicator"
```

**Activation — one env var, no upstream changes:**

```bash
SGLANG_PLATFORM="sglang_musa_plugin.platform.MusaPlatform" \
    python -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct
```

---

### How SRT Core Files Change

All hardware-specific guards in SRT core collapse to `current_platform` calls.

#### `model_executor/model_runner.py`

```python
# ── BEFORE (line ~189) ─────────────────────────────────────────────────────
if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend
    init_npu_backend()

# ── AFTER — one line, works for every platform ────────────────────────────
current_platform.initialize_device(rank=self.tp_rank, local_rank=self.gpu_id)
```

```python
# ── BEFORE (line ~2027) ───────────────────────────────────────────────────
device_to_graph_runner = {
    "cuda": CudaGraphRunner, "musa": CudaGraphRunner,
    "cpu":  CPUGraphRunner,  "npu":  NPUGraphRunner,
}
graph_runner_cls = device_to_graph_runner.get(self.device, CudaGraphRunner)

# ── AFTER ─────────────────────────────────────────────────────────────────
graph_runner_cls = current_platform.get_graph_runner_class()
# (CudaPlatform.get_graph_runner_class() returns CudaGraphRunner, etc.)
```

#### `model_executor/model_runner_kv_cache_mixin.py`

```python
# ── BEFORE — 200-line if-elif chain ───────────────────────────────────────
if self.server_args.attention_backend == "ascend":
    if self.use_mla_backend:
        from ... import NPUMLATokenToKVPool
        self.token_to_kv_pool = NPUMLATokenToKVPool(...)
    else:
        ...
elif ...:   # 180 more lines
    ...

# ── AFTER — single dispatch, builtin chain kept for CUDA/CPU/Mamba/NSA ────
pool = current_platform.create_kv_pool(self, ...)
if pool is None:
    pool = _builtin_kv_pool_factory(self, ...)   # existing CUDA/CPU/Mamba logic
self.token_to_kv_pool = pool
```

#### `server_args.py`

```python
def __post_init__(self):
    # … all existing logic unchanged …

    # Append at end — platform injects defaults only for None values
    from sglang.srt.platforms import current_platform
    current_platform.apply_server_args_defaults(self)
```

#### `layers/attention/attention_registry.py`

```python
# Append after all @register_attention_backend decorators
def _register_platform_backends():
    from sglang.srt.platforms import current_platform
    for name, factory in current_platform.get_attention_backends().items():
        if name not in ATTENTION_BACKENDS:   # built-ins are never overwritten
            ATTENTION_BACKENDS[name] = factory

_register_platform_backends()
```

---

## Comparison with `multimodal_gen/runtime/platforms/`

| Aspect | `multimodal_gen` | SRT (this RFC) |
|---|---|---|
| Base class | `Platform` | `Platform` (same) |
| Singleton | `current_platform` | `current_platform` (same) |
| Lazy init | `__getattr__` | `__getattr__` (same) |
| OOT override env var | planned TODO | `SGLANG_PLATFORM=<qualname>` |
| `PlatformEnum.OOT` | ✅ | ✅ included |
| Attention | `get_attn_backend_cls_str()` returns one class qualname | `get_attention_backends()` returns a dict — needed because SRT users select by name via `--attention-backend` |
| KV pool | N/A | `create_kv_pool()` |
| Graph runner | N/A | `get_graph_runner_class()` |
| Server args defaults | N/A | `apply_server_args_defaults()` |
| Device init | N/A | `initialize_device()` |

The structural difference in attention handling is intentional: SRT users specify
`--attention-backend flashinfer` by name, so the platform exposes a registry dict
(`{name: factory}`) rather than a single resolved class.

---

## Implementation Plan

### Phase 1 — New module, zero side-effects

Create `python/sglang/srt/platforms/` with `interface.py`, `__init__.py`, and all
six built-in platform files. No existing file is modified. CI proves the module
imports cleanly on all targets.

**New files**:
- [python/sglang/srt/platforms/\_\_init\_\_.py](python/sglang/srt/platforms/__init__.py)
- [python/sglang/srt/platforms/interface.py](python/sglang/srt/platforms/interface.py)
- [python/sglang/srt/platforms/cuda.py](python/sglang/srt/platforms/cuda.py)
- [python/sglang/srt/platforms/rocm.py](python/sglang/srt/platforms/rocm.py)
- [python/sglang/srt/platforms/npu.py](python/sglang/srt/platforms/npu.py)
- [python/sglang/srt/platforms/xpu.py](python/sglang/srt/platforms/xpu.py)
- [python/sglang/srt/platforms/hpu.py](python/sglang/srt/platforms/hpu.py)
- [python/sglang/srt/platforms/cpu.py](python/sglang/srt/platforms/cpu.py)

### Phase 2 — Wire `current_platform` into SRT core

Replace the five hardware-switch sites with `current_platform` calls:

| File | Change |
|---|---|
| [python/sglang/srt/model_executor/model_runner.py](python/sglang/srt/model_executor/model_runner.py) | Replace `if _is_npu: init_npu_backend()` → `current_platform.initialize_device(...)` |
| [python/sglang/srt/model_executor/model_runner.py](python/sglang/srt/model_executor/model_runner.py) | Replace `device_to_graph_runner` dict → `current_platform.get_graph_runner_class()` |
| [python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py](python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py) | Dispatch to `current_platform.create_kv_pool(...)` before entering the existing if-elif chain |
| [python/sglang/srt/server_args.py](python/sglang/srt/server_args.py) | Append `current_platform.apply_server_args_defaults(self)` at end of `__post_init__` |
| [python/sglang/srt/layers/attention/attention_registry.py](python/sglang/srt/layers/attention/attention_registry.py) | Append `_register_platform_backends()` after built-in decorators |

### Phase 3 — Replace scattered hardware detection calls

`grep -r 'is_npu\|is_hip\|is_xpu\|_is_npu\|_is_hip' python/sglang/srt/` yields ~40
call-sites. Replace each with `current_platform.is_npu()` etc. This is mechanical;
existing `is_npu()` / `is_hip()` helpers in `utils/common.py` are kept as
thin wrappers for backward compatibility.

### Phase 4 — Documentation

- Add `docs/developer_guide/hardware_platform_guide.md`
- Provide `sglang-platform-template` skeleton repository
- Add a `MockPlatform`-based integration test in CI

---

## Verification

### Unit tests

```python
# tests/srt/test_platforms.py
from sglang.srt.platforms import Platform, PlatformEnum

class MockPlatform(Platform):
    _enum       = PlatformEnum.OOT
    device_name = "mock"
    device_type = "cpu"
    def get_attention_backends(self):
        return {"mock_attn": lambda r: object()}
    def get_graph_runner_class(self):
        return object   # sentinel
    def apply_server_args_defaults(self, args):
        if args.attention_backend is None:
            args.attention_backend = "mock_attn"

def test_current_platform_overridable(monkeypatch):
    import sglang.srt.platforms as p
    monkeypatch.setattr(p, "_current_platform", MockPlatform())
    assert p.current_platform.device_name == "mock"

def test_attention_backends_injected(monkeypatch):
    import sglang.srt.platforms as p
    import sglang.srt.layers.attention.attention_registry as reg
    monkeypatch.setattr(p, "_current_platform", MockPlatform())
    reg._register_platform_backends()
    assert "mock_attn" in reg.ATTENTION_BACKENDS

def test_builtin_attention_not_overridden():
    import sglang.srt.layers.attention.attention_registry as reg
    assert "triton" in reg.ATTENTION_BACKENDS   # built-in must survive
```

### Integration smoke test

```bash
SGLANG_PLATFORM="tests.srt.mock_platform.MockPlatform" \
    python -c "from sglang.srt.platforms import current_platform; \
               print(current_platform.device_name)"
# expected: mock
```

### Regression: all built-in platforms unaffected

```bash
python -m pytest tests/srt/ -k "cuda"           -v   # NVIDIA CI
python -m pytest tests/srt/ -k "hip or rocm"    -v   # AMD CI
python -m pytest tests/srt/ -k "npu or ascend"  -v   # Ascend CI
```

---

## Open Questions

1. **Q1 — Priority when multiple platforms are available** — e.g., a node with both
   ROCm and a PCIe NPU. Current answer: the detection list is ordered (ROCm before
   CUDA before NPU…). `SGLANG_PLATFORM` always wins. Is this sufficient?

2. **Q2 — Capability flags** — should `Platform` declare `supports_mla: bool`,
   `supports_cuda_graph: bool` etc., so the Scheduler can make decisions without
   calling into platform code (avoids circular imports)?

3. **Q3 — Communicator lifecycle** — the communicator is initialised inside
   `distributed/parallel_state.py` during `init_process_group`. Does `Platform` need
   a hook *before* that call, or is `initialize_device()` early enough?

4. **Q4 — API stability** — proposed contract: adding new optional methods to
   `Platform` is a minor-version change; changing existing method signatures requires
   a major-version bump and a deprecation cycle.

5. **Q5 — Coexistence with `multimodal_gen/runtime/platforms/`** — should the two
   `Platform` base classes eventually converge into a shared
   `sglang.platforms.interface` module, or remain separate forever?

---

## Alternatives Considered

### A — Separate `HardwarePlugin` + per-subsystem registries *(original draft)*
Rejected: duplicates work already done in `multimodal_gen`; spreads hardware logic
across multiple registries instead of one object.

### B — Monkey-patching
Rejected: fragile, no interface guarantees, impossible to document or version.

### C — Document-only
Rejected: does not enable out-of-tree development.

### D — This proposal: `Platform` pattern (mirrors `multimodal_gen`)
Selected: consistent with existing SGLang codebase, proven pattern (adapted from
vLLM), OOT vendor cost is one env var + one Python class.
