# RFC: SGLang SRT Hardware Plugin System

**Status**: Draft
**Date**: 2026-03-04
**Target**: SRT Core/Plugin Refactor

---

## Abstract

This RFC proposes a unified **hardware plugin mechanism** for SGLang's SRT runtime, enabling third-party hardware vendors to integrate their devices (GPUs, NPUs, IPUs, or any accelerator) into SGLang **without modifying upstream source code**. The system relies on standard Python entry points for plugin discovery and provides a coherent ABC that covers all hardware-specific extension points.

---

## Motivation

### Current State

SGLang already supports multiple hardware backends: CUDA GPUs, AMD ROCm GPUs, Intel XPUs, Huawei Ascend NPUs, Habana HPUs, and CPU-only inference. However, each integration required modifying **9+ upstream core files**. The Ascend NPU integration — the most comprehensive example — touched:

| File | Reason for modification |
|---|---|
| `utils/common.py` | Add `is_npu()` hardware detection function |
| `server_args.py` | Inject NPU-specific default arguments |
| `layers/attention/attention_registry.py` | Register attention backend |
| `model_executor/model_runner.py` | Add device initialization branch |
| `model_executor/model_runner_kv_cache_mixin.py` | Insert NPU branch into 400-line if-elif factory |
| `layers/quantization/__init__.py` | Register quantization methods |
| `layers/moe/utils.py` | Add MoE backend enum values |
| `environ.py` | Add NPU-specific environment variables |
| `distributed/parallel_state.py` | Register communication backend |

The same pattern applies to every other hardware vendor wanting to integrate, including emerging players like Moore Threads (MUSA), Enflame, and future accelerators.

### Core Problems

1. **Code coupling** — Hardware-specific code is scattered across 15+ files with no clear boundaries
2. **No plugin discovery** — All backends must be registered in-tree; out-of-tree development is impossible
3. **Inconsistent extension points** — Attention backends have a decorator registry, but graph runners, memory pools, and communicators use hardcoded dicts or if-elif chains
4. **Base class pollution** — `base_attn_backend.py` contains direct `is_npu()` calls; the base class must not know about specific hardware
5. **Giant conditional factories** — `model_runner_kv_cache_mixin.py` has a 200-line if-elif chain for memory pool selection

---

## Goals

- **G1**: A hardware vendor ships a single Python package; `pip install` activates support in SGLang with no upstream changes required
- **G2**: All hardware-related extension points (attention, memory pool, graph runner, communicator, quantization, MoE) share a consistent registration mechanism
- **G3**: Existing in-tree backends (CUDA, ROCm, Ascend NPU, Intel XPU, Habana HPU, CPU) continue to work without any modification
- **G4**: The plugin API is stable enough for vendors to maintain independent release cadences

## Non-Goals

- Not refactoring Scheduler core scheduling logic
- Not changing the model architecture registration mechanism (`SGLANG_EXTERNAL_MODEL_PACKAGE` remains unchanged)
- Not introducing runtime plugin hot-reload (startup-time only)
- Not changing the existing `AttentionBackend` method signatures

---

## Background: Current Architecture

### What Works Today (Reusable)

- ✅ **Attention Backend Registry** (`attention_registry.py`) — decorator-based registration, good pattern to generalize
- ✅ **External Model Package** (`SGLANG_EXTERNAL_MODEL_PACKAGE`) — external models already supported
- ✅ `add_attention_backend_choices()` / `add_quantization_method_choices()` — exist but undocumented
- ✅ `hardware_backend/npu/` — good directory layout as a template for vendor-owned code

### Current Built-in Backend Landscape

The plugin system must coexist with these built-in paths that will **not** be removed:

| Device | Graph Runner | Attention Backends | Communicator |
|---|---|---|---|
| CUDA (NVIDIA) | `CudaGraphRunner` | flashinfer, fa3, fa4, cutlass_mla, trtllm_* | PyNCCL / custom all-reduce |
| ROCm (AMD) | `CudaGraphRunner` | aiter, wave, flashinfer | PyNCCL |
| NPU (Ascend) | `NPUGraphRunner` | ascend | NPUCommunicator |
| XPU (Intel) | `CudaGraphRunner` | intel_xpu, intel_amx | XPUCommunicator |
| HPU (Habana) | — | torch_native | HPUCommunicator |
| CPU | `CPUGraphRunner` | torch_native, triton | — |

### Pain Points to Fix

- ❌ **Graph runner selection**: hardcoded dict in `model_runner.py:2027`
- ❌ **Memory pool factory**: 200-line if-elif in `model_runner_kv_cache_mixin.py`
- ❌ **Hardware detection**: hardcoded `is_npu()`, `is_hip()`, etc. in `utils/common.py`
- ❌ **Device init**: hardcoded `if _is_npu: init_npu_backend()` in `model_runner.py:189`
- ❌ **Server args defaults**: no hook for vendors to inject hardware-specific defaults

---

## Proposed Design

### Overall Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        SGLang SRT Core                            │
│                                                                    │
│  ┌────────────────┐   ┌─────────────────┐   ┌────────────────┐  │
│  │  PluginLoader  │   │  PluginRegistry  │   │  Core Runtime  │  │
│  │                │──▶│                  │──▶│  model_runner  │  │
│  │  entry_points  │   │  Active Plugin   │   │  scheduler     │  │
│  │  SGLANG_PLUGINS│   │  + built-ins     │   │  attention...  │  │
│  └────────────────┘   └─────────────────┘   └────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
          ▲                      ▲
          │                      │
┌─────────┴──────┐   ┌───────────┴────────┐   ┌────────────────┐
│  sglang-musa-  │   │  sglang-enflame-   │   │  sglang-xyz-   │
│  plugin        │   │  plugin            │   │  plugin        │
│  MusaPlugin()  │   │  EnflamePlugin()   │   │  XyzPlugin()   │
└────────────────┘   └────────────────────┘   └────────────────┘
```

### Design Principle: Built-ins First, Plugin as Fallback

The plugin system **never replaces** built-in paths; it extends them. The pattern used everywhere:

```python
# Pseudocode used for every extension point
def select_X(device):
    # 1. Built-in paths always run first (unchanged)
    if device in _BUILTIN_X_MAP:
        return _BUILTIN_X_MAP[device]

    # 2. Plugin extension — triggered only for unknown devices
    plugin = PluginRegistry.get_active_plugin()
    if plugin and (result := plugin.get_X()):
        return result

    return _default_X  # existing fallback unchanged
```

---

### Core Abstraction 1: `HardwarePlugin` Interface

**New file**: `python/sglang/srt/plugin/hardware_plugin.py`

```python
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Type

class HardwarePlugin(ABC):
    """
    SGLang hardware plugin interface.

    Implement this class and publish it as a Python package with the
    entry point group 'sglang.hardware_plugins' to integrate a new
    hardware backend without modifying SGLang upstream.

    Only device_name and is_available() are mandatory. Every other
    method has a no-op default, so vendors only implement what they need.
    """

    # ── Required ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def device_name(self) -> str:
        """
        Device name string used in torch.device and logging.
        Examples: 'musa', 'npu', 'xpu', 'mlu', 'ncore'
        Must not conflict with existing built-in device names.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if and only if this hardware is detected in the current environment."""
        ...

    # ── Optional: Lifecycle ─────────────────────────────────────────

    def initialize_device(self, rank: int, local_rank: int) -> None:
        """
        Device-level initialization (replaces hardcoded if _is_npu: init_npu_backend()).
        Called once per worker process before model loading.
        """
        pass

    def apply_server_args_defaults(self, args: "ServerArgs") -> None:
        """
        Inject hardware-specific ServerArgs defaults.
        Only applied when the user has not explicitly set the argument.
        Called at the end of ServerArgs.__post_init__().
        """
        pass

    # ── Optional: Kernel / Backend Extensions ───────────────────────

    def get_attention_backends(self) -> Dict[str, Callable]:
        """
        Return {backend_name: factory_fn} for attention backends.
        factory_fn signature: (runner: ModelRunner) -> AttentionBackend
        """
        return {}

    def get_default_attention_backend(self) -> Optional[str]:
        """Return the preferred attention backend name for this hardware, or None."""
        return None

    def get_graph_runner_class(self) -> Optional[Type]:
        """
        Return a custom GraphRunner subclass (replaces the hardcoded device->class dict).
        Return None to use the built-in selection logic.
        """
        return None

    def get_memory_pool_factory(self) -> Optional[Callable]:
        """
        Return a factory: (runner, **kwargs) -> BaseTokenToKVPool.
        Replaces the per-device branch in model_runner_kv_cache_mixin.py.
        Return None to use built-in memory pool selection.
        """
        return None

    def get_allocator_factory(self) -> Optional[Callable]:
        """Return a factory for the KV cache allocator, or None for the built-in."""
        return None

    def get_communicator_class(self) -> Optional[Type]:
        """
        Return a custom GroupCoordinator subclass for collective communication.
        Return None to use the built-in communicator (NCCL or device-specific).
        """
        return None

    # ── Optional: Methods / Quantization ────────────────────────────

    def get_quantization_methods(self) -> Dict[str, Type]:
        """Return {method_name: QuantizationConfig subclass} for hardware quantization."""
        return {}

    def get_moe_backends(self) -> Dict[str, str]:
        """Return additional MoE A2A or runner backend names."""
        return {}

    def get_disaggregation_backends(self) -> Dict[str, Type]:
        """Return custom KV-transfer backends for prefill/decode disaggregation."""
        return {}

    # ── Optional: Configuration ──────────────────────────────────────

    def get_environment_variables(self) -> List:
        """
        Declare hardware-specific environment variables (EnvField instances).
        Used for documentation generation. Not required for the variables to work.
        """
        return []
```

---

### Core Abstraction 2: `PluginRegistry`

**New file**: `python/sglang/srt/plugin/plugin_registry.py`

```python
import importlib, logging, os
from typing import Dict, Optional
from sglang.srt.plugin.hardware_plugin import HardwarePlugin

logger = logging.getLogger(__name__)

class PluginRegistry:
    """
    Central registry for SGLang hardware plugins.

    Discovery order (highest priority first):
      1. SGLANG_PLUGINS env var — for development / CI without installing a package
      2. Python entry_points group 'sglang.hardware_plugins' — for production use
      3. Manual register() calls — for testing

    Only one plugin is "active" at a time, selected as the first plugin
    whose is_available() returns True.
    """

    _plugins: Dict[str, "HardwarePlugin"] = {}
    _active_plugin: Optional["HardwarePlugin"] = None
    _initialized: bool = False

    @classmethod
    def discover_and_load(cls) -> None:
        """Call once at process startup. Idempotent."""
        if cls._initialized:
            return
        cls._initialized = True
        cls._load_from_env()          # higher priority: env var
        cls._load_from_entry_points() # standard packaging mechanism
        cls._activate_for_current_device()

    @classmethod
    def _load_from_entry_points(cls) -> None:
        try:
            from importlib.metadata import entry_points
            for ep in entry_points(group="sglang.hardware_plugins"):
                try:
                    plugin: HardwarePlugin = ep.load()()
                    cls._plugins.setdefault(plugin.device_name, plugin)
                    logger.debug(f"[Plugin] Discovered: {plugin.device_name} from {ep.name}")
                except Exception as e:
                    logger.warning(f"[Plugin] Failed to load {ep.name}: {e}")
        except Exception:
            pass

    @classmethod
    def _load_from_env(cls) -> None:
        """
        SGLANG_PLUGINS=pkg.module.ClassName[,pkg2.module2.ClassName2]
        Useful for development without packaging.
        """
        for spec in os.environ.get("SGLANG_PLUGINS", "").split(","):
            spec = spec.strip()
            if not spec:
                continue
            try:
                module_path, class_name = spec.rsplit(".", 1)
                plugin: HardwarePlugin = getattr(
                    importlib.import_module(module_path), class_name
                )()
                cls._plugins[plugin.device_name] = plugin
                logger.info(f"[Plugin] Loaded from SGLANG_PLUGINS: {plugin.device_name}")
            except Exception as e:
                logger.warning(f"[Plugin] Could not load '{spec}': {e}")

    @classmethod
    def _activate_for_current_device(cls) -> None:
        for device_name, plugin in cls._plugins.items():
            try:
                if plugin.is_available():
                    cls._active_plugin = plugin
                    logger.info(f"[Plugin] Active: {device_name}")
                    return
            except Exception as e:
                logger.debug(f"[Plugin] is_available() failed for {device_name}: {e}")

    @classmethod
    def get_active_plugin(cls) -> Optional["HardwarePlugin"]:
        return cls._active_plugin

    @classmethod
    def register(cls, plugin: "HardwarePlugin") -> None:
        """Programmatic registration; intended for unit tests."""
        cls._plugins[plugin.device_name] = plugin
        if plugin.is_available():
            cls._active_plugin = plugin
```

---

### Integration Points

#### Point 1: Plugin Discovery (startup)

**File**: `python/sglang/srt/model_executor/model_runner.py`

```python
# Add near the top of the file, after existing hardware detection flags
from sglang.srt.plugin import PluginRegistry
PluginRegistry.discover_and_load()  # no-op if no plugins installed

# In ModelRunner.__init__, AFTER the existing if _is_npu: block (~line 193)
# (the existing block is untouched)
else:
    plugin = PluginRegistry.get_active_plugin()
    if plugin:
        plugin.initialize_device(rank=self.tp_rank, local_rank=self.gpu_id)
```

#### Point 2: Attention Backend Auto-registration

**File**: `python/sglang/srt/layers/attention/attention_registry.py`

```python
# At module initialization, after all built-in @register_attention_backend decorators
def _register_plugin_attention_backends() -> None:
    plugin = PluginRegistry.get_active_plugin()
    if plugin:
        for name, factory in plugin.get_attention_backends().items():
            if name not in ATTENTION_BACKENDS:  # built-ins are never overridden
                ATTENTION_BACKENDS[name] = factory

_register_plugin_attention_backends()
```

#### Point 3: Graph Runner Selection

**File**: `python/sglang/srt/model_executor/model_runner.py` (~line 2027)

```python
# BEFORE (unchanged):
_BUILTIN_GRAPH_RUNNERS = {
    "cuda": CudaGraphRunner,
    "musa": CudaGraphRunner,   # existing entry stays
    "cpu":  CPUGraphRunner,
    "npu":  NPUGraphRunner,
    # ...
}
graph_runner_cls = _BUILTIN_GRAPH_RUNNERS.get(self.device)

# AFTER (append only):
if graph_runner_cls is None:
    plugin = PluginRegistry.get_active_plugin()
    if plugin:
        graph_runner_cls = plugin.get_graph_runner_class()
if graph_runner_cls is None:
    graph_runner_cls = CudaGraphRunner  # existing fallback
```

#### Point 4: Memory Pool Factory

**File**: `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`

```python
# At the very end of the existing if-elif chain, replace the final else:
else:
    plugin = PluginRegistry.get_active_plugin()
    if plugin and (factory := plugin.get_memory_pool_factory()):
        self.token_to_kv_pool = factory(self, ...)
    else:
        self.token_to_kv_pool = MHATokenToKVPool(...)  # original fallback preserved
```

#### Point 5: Server Args Defaults

**File**: `python/sglang/srt/server_args.py`

```python
def __post_init__(self):
    # ... all existing logic untouched ...

    # Append at the end — plugin injects defaults only where user left values as None
    plugin = PluginRegistry.get_active_plugin()
    if plugin:
        plugin.apply_server_args_defaults(self)
```

---

### Packaging Convention for Vendor Plugins

Any hardware vendor publishes a package following this layout:

```
sglang_<vendor>_plugin/
├── __init__.py
├── plugin.py             # VendorPlugin(HardwarePlugin) — main entry
├── attention/
│   └── vendor_attn.py    # AttentionBackend subclass
├── memory/
│   ├── vendor_pool.py    # BaseTokenToKVPool subclass
│   └── vendor_alloc.py
├── graph_runner/
│   └── vendor_runner.py
├── distributed/
│   └── vendor_comm.py    # GroupCoordinator subclass
└── quantization/
    └── vendor_quant.py   # QuantizationConfig subclass

# pyproject.toml of the vendor package:
[project.entry-points."sglang.hardware_plugins"]
<vendor-device> = "sglang_<vendor>_plugin.plugin:VendorPlugin"
```

**VendorPlugin skeleton**:

```python
# sglang_<vendor>_plugin/plugin.py
from sglang.srt.plugin import HardwarePlugin

class VendorPlugin(HardwarePlugin):

    @property
    def device_name(self) -> str:
        return "<device>"   # e.g. "musa", "mlu", "ncore"

    def is_available(self) -> bool:
        try:
            import vendor_torch_extension
            return vendor_torch_extension.is_available()
        except ImportError:
            return False

    def initialize_device(self, rank: int, local_rank: int) -> None:
        import vendor_torch_extension
        vendor_torch_extension.set_device(local_rank)

    def apply_server_args_defaults(self, args) -> None:
        if args.attention_backend is None:
            args.attention_backend = "<vendor>_flash"
        # Only set what the user hasn't already configured

    def get_attention_backends(self):
        from sglang_<vendor>_plugin.attention.vendor_attn import VendorFlashAttn
        return {
            "<vendor>_flash": lambda runner: VendorFlashAttn(runner),
        }

    def get_graph_runner_class(self):
        from sglang_<vendor>_plugin.graph_runner.vendor_runner import VendorGraphRunner
        return VendorGraphRunner

    def get_communicator_class(self):
        from sglang_<vendor>_plugin.distributed.vendor_comm import VendorCommunicator
        return VendorCommunicator
```

**Activation** — two modes:

```bash
# Production: install the package; auto-activated on the target device
pip install sglang-<vendor>-plugin

# Development (no install needed):
SGLANG_PLUGINS="sglang_vendor_plugin.plugin.VendorPlugin" \
    python -m sglang.launch_server --model ...
```

---

## Implementation Plan

### Phase 1 — Infrastructure (additive only, zero changes to existing code paths)

- Create `python/sglang/srt/plugin/` module
  - `hardware_plugin.py` — HardwarePlugin ABC
  - `plugin_registry.py` — PluginRegistry
  - `__init__.py` — public exports
- Add `PluginRegistry.discover_and_load()` call in `model_runner.py` (2 lines)
- Declare `sglang.hardware_plugins` entry point group in `pyproject.toml` (3 lines)
- Write unit tests with a `MockHardwarePlugin`

**Files touched**:
- `python/sglang/srt/plugin/` (new directory, 3 files)
- `python/sglang/srt/model_executor/model_runner.py` (+2 lines)
- `python/pyproject.toml` (+3 lines)

### Phase 2 — Extension Point Hooks (append-only pattern)

- Attention backend auto-registration from plugin (`attention_registry.py`)
- Graph runner plugin fallback in `model_runner.py`
- Server args defaults hook in `server_args.py`
- Device initialization hook in `model_runner.py`

**Files touched**:
- `python/sglang/srt/layers/attention/attention_registry.py` (+8 lines)
- `python/sglang/srt/model_executor/model_runner.py` (+10 lines total)
- `python/sglang/srt/server_args.py` (+5 lines)

### Phase 3 — Memory Pool Plugin Fallback

- Add plugin fallback at the end of the existing if-elif chain in `model_runner_kv_cache_mixin.py`
- No existing branches are modified

**Files touched**:
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` (+8 lines)

### Phase 4 — Documentation & Template

- Write *Hardware Backend Plugin Development Guide* (docs/)
- Publish `sglang-plugin-template` skeleton repository
- Add integration test with a mock plugin in CI

---

## Critical File Paths

### Files to Modify

| File | Change |
|---|---|
| `python/sglang/srt/model_executor/model_runner.py` | Add discover_and_load; device init and graph runner plugin fallback |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | Memory pool plugin fallback in else branch |
| `python/sglang/srt/server_args.py` | Plugin default injection at end of `__post_init__` |
| `python/sglang/srt/layers/attention/attention_registry.py` | Plugin batch-register after built-ins |
| `python/pyproject.toml` | Declare `sglang.hardware_plugins` entry point group |

### New Files

| File | Content |
|---|---|
| `python/sglang/srt/plugin/__init__.py` | Public API exports |
| `python/sglang/srt/plugin/hardware_plugin.py` | HardwarePlugin ABC |
| `python/sglang/srt/plugin/plugin_registry.py` | PluginRegistry |

### Reference Files (patterns to reuse)

| File | Pattern |
|---|---|
| `layers/attention/attention_registry.py` | Decorator registry → generalize to all extension points |
| `hardware_backend/npu/utils.py` | `set_default_server_args()` → standardize as plugin hook |
| `models/registry.py` | External package loading → align with entry_points |

---

## Verification

### Unit Tests

```python
# tests/srt/test_plugin_system.py

class MockPlugin(HardwarePlugin):
    @property
    def device_name(self): return "mock_hw"
    def is_available(self): return True
    def get_attention_backends(self):
        return {"mock_attn": lambda runner: MockAttnBackend(runner)}
    def get_graph_runner_class(self):
        return MockGraphRunner
    def apply_server_args_defaults(self, args):
        args.attention_backend = args.attention_backend or "mock_attn"

def test_plugin_registration():
    PluginRegistry.register(MockPlugin())
    assert PluginRegistry.get_active_plugin().device_name == "mock_hw"

def test_attention_backend_injected():
    # After registration, attention registry should include mock_attn
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    assert "mock_attn" in ATTENTION_BACKENDS

def test_builtin_not_overridden():
    # Built-in 'triton' must survive plugin injection
    assert "triton" in ATTENTION_BACKENDS

def test_server_args_defaults():
    from sglang.srt.server_args import ServerArgs
    args = ServerArgs(model_path="dummy")
    assert args.attention_backend == "mock_attn"
```

### Integration Test (env-var path)

```bash
SGLANG_PLUGINS="tests.mock_hw_plugin.MockPlugin" \
    python -m pytest tests/srt/test_plugin_integration.py -v
```

### Regression: Built-in Backends Unaffected

```bash
# CUDA path unchanged
python -m pytest tests/srt/ -k "cuda" -v

# ROCm path unchanged (on ROCm CI)
python -m pytest tests/srt/ -k "hip or rocm" -v

# NPU path unchanged (on Ascend CI)
python -m pytest tests/srt/ -k "npu or ascend" -v
```

---

## Open Questions

1. **Q1 — Plugin priority**: If two installed plugins report `is_available() == True` (e.g., on a multi-device node), how should the active plugin be determined? Options: first-found, explicit `--plugin <name>` flag, or environment variable.
2. **Q2 — Explicit activation**: Should `--plugin <device>` be added to `ServerArgs` to allow explicit selection independent of hardware auto-detection?
3. **Q3 — Capability flags**: Should the plugin declare capability flags like `supports_mla: bool` or `supports_cuda_graph: bool` to let the scheduler make informed decisions without calling into vendor code?
4. **Q4 — API stability guarantees**: What is the breaking-change policy for `HardwarePlugin`? Proposal: minor version bumps may add optional methods; major version bumps may change required methods.
5. **Q5 — Communicator lifecycle**: The current communicator is initialized deep inside `distributed/` during process group setup. Does the plugin need an earlier hook for this?

---

## Alternatives Considered

### Option A — Document-only (rejected)
Document the files that must be modified for each new backend, with no code changes.
**Rejected**: Does not solve version tracking or out-of-tree development; vendors must still fork.

### Option B — Monkey-patching (rejected)
Allow external packages to patch SGLang internals at runtime.
**Rejected**: Fragile, hard to debug, zero interface stability guarantees.

### Option C — Fork-friendly consolidation (rejected)
Consolidate all hardware logic into a single file to make forking easier.
**Rejected**: Still requires a fork; does not enable pip-installable integration.

### Option D — This proposal: standard Python entry_points
Use `importlib.metadata` entry points (PEP 517/518 compliant).
**Selected**: Aligns with Python ecosystem standards (same mechanism as pytest plugins, Flask extensions, etc.), version management is handled by pip, and `SGLANG_PLUGINS` env var covers development workflows.
