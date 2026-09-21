"""Optional dependency loading, runtime ownership and CPU topology discovery."""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
import threading
from importlib import metadata
from pathlib import Path
from typing import Any

from unisim.optional import OptionalDependencyError

# --------------------------------------------------------------------- #
# Optional, process-local loading of the supported SuperDex Python runtime
# --------------------------------------------------------------------- #

# Temporary: the unilabsim superdex-uni wheels carry the native batch executor
# until the upstream project_superdex PR merges and publishes equivalent
# superdex-physics/superdex-robotics wheels; switch these names back then.
_DISTRIBUTIONS = ("superdex-physics-uni", "superdex-robotics-uni")
_SUPPORTED_PYTHON = ((3, 12), (3, 13))
_SUPPORTED_PYTHON_TEXT = "3.12 or 3.13"
_HINT = (
    f"Use Python {_SUPPORTED_PYTHON_TEXT} and install unisim-core[superdex] "
    "(SuperDex 1.0.0, superdex-uni build)."
)


class SuperDexDependencyError(OptionalDependencyError):
    """The optional SuperDex ABI or distribution is unavailable."""


def _prioritize_local_native_extension() -> None:
    """Restore a local pybind directory after a spawned facade mutated sys.path."""
    for entry in reversed(os.environ.get("PYTHONPATH", "").split(os.pathsep)):
        if not entry:
            continue
        path = Path(entry)
        if path.is_dir() and any(path.glob("mochi_physics*.so")):
            entry_text = str(path)
            if entry_text in sys.path:
                sys.path.remove(entry_text)
            sys.path.insert(0, entry_text)


def superdex_dependencies_available() -> bool:
    """Check package metadata without importing the native runtime."""
    if sys.version_info[:2] not in _SUPPORTED_PYTHON:
        return False
    try:
        return all(metadata.version(name) == "1.0.0" for name in _DISTRIBUTIONS)
    except metadata.PackageNotFoundError:
        return False


def load_superdex_dependencies() -> tuple[Any, Any]:
    """Load the precision-consistent Physics and Robotics public facades lazily."""
    if sys.version_info[:2] not in _SUPPORTED_PYTHON:
        raise SuperDexDependencyError(
            f"superdex requires CPython {_SUPPORTED_PYTHON_TEXT}. {_HINT}"
        )
    for name in _DISTRIBUTIONS:
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise SuperDexDependencyError(f"Missing {name}==1.0.0. {_HINT}") from exc
        if installed != "1.0.0":
            raise SuperDexDependencyError(
                f"superdex requires {name}==1.0.0; found {installed}. {_HINT}"
            )
    try:
        # A source-built SceneBatchExecutor is supplied through PYTHONPATH for
        # local integration. Preload it before the public facade inserts its
        # packaged `_native` directory ahead of Python's normal search path;
        # spawn collectors then inherit the same selected extension.
        try:
            _prioritize_local_native_extension()
            importlib.import_module("mochi_physics")
        except ImportError:
            pass
        return (
            importlib.import_module("superdex.physics"),
            importlib.import_module("superdex.robotics"),
        )
    except (ImportError, OSError) as exc:
        raise SuperDexDependencyError(f"Could not load SuperDex: {exc}. {_HINT}") from exc


# --------------------------------------------------------------------- #
# Reference-counted ownership of SuperDex's process-global CPU runtime
# --------------------------------------------------------------------- #

_LOCK = threading.RLock()
_PID: int | None = None
_USERS = 0
_NUM_WORKER_THREADS: int | None = None


def acquire_runtime(physics: Any, num_worker_threads: int = 0) -> None:
    """Initialize the consistently configured SDK once per spawn process."""
    global _NUM_WORKER_THREADS, _PID, _USERS
    with _LOCK:
        if _PID is not None and _PID != os.getpid():
            raise RuntimeError("superdex runtime was inherited by fork; use spawn collectors")
        if not _USERS:
            if physics.is_initialized():
                raise RuntimeError(
                    "SuperDex was initialized outside UniSim; close that runtime before "
                    "constructing a backend so initialization/shutdown ownership is unambiguous"
                )
            physics.initialize(num_worker_threads=num_worker_threads)
            _PID = os.getpid()
            _NUM_WORKER_THREADS = num_worker_threads
        elif num_worker_threads != _NUM_WORKER_THREADS:
            raise RuntimeError(
                "SuperDex runtime is already initialized with "
                f"num_worker_threads={_NUM_WORKER_THREADS}; requested {num_worker_threads}"
            )
        _USERS += 1


def release_runtime(physics: Any) -> None:
    """Shut down only after the last backend has destroyed its native resources."""
    global _NUM_WORKER_THREADS, _PID, _USERS
    with _LOCK:
        if _PID != os.getpid() or not _USERS:
            return
        _USERS -= 1
        if not _USERS:
            physics.shutdown()
            _PID = None
            _NUM_WORKER_THREADS = None


# --------------------------------------------------------------------- #
# Portable physical CPU discovery for the SuperDex native worker pool
# --------------------------------------------------------------------- #

def _available_cpu_ids() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return list(range(os.cpu_count() or 1))


def _linux_physical_groups(cpu_ids: list[int]) -> list[list[int]]:
    groups: dict[tuple[str, str], list[int]] = {}
    topology = Path("/sys/devices/system/cpu")
    for cpu_id in cpu_ids:
        try:
            package = (topology / f"cpu{cpu_id}/topology/physical_package_id").read_text().strip()
            core = (topology / f"cpu{cpu_id}/topology/core_id").read_text().strip()
        except OSError:
            return []
        groups.setdefault((package, core), []).append(cpu_id)
    return list(groups.values())


def physical_cpu_groups() -> list[list[int]]:
    """Return affinity-visible logical CPUs grouped by physical core.

    Linux exposes an exact package/core mapping. macOS does not expose a
    stable logical-CPU-to-core mapping, so its physical count is used to form
    deterministic contiguous groups; this keeps the worker count correct and
    avoids relying on Linux-only affinity APIs.
    """
    cpu_ids = _available_cpu_ids()
    if platform.system() == "Linux":
        groups = _linux_physical_groups(cpu_ids)
        if groups:
            return groups
    if platform.system() == "Darwin":
        try:
            physical = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.physicalcpu"], text=True, timeout=1
                ).strip()
            )
            physical = max(1, min(physical, len(cpu_ids)))
            groups = [[] for _ in range(physical)]
            for index, cpu_id in enumerate(cpu_ids):
                groups[min(index * physical // max(1, len(cpu_ids)), physical - 1)].append(cpu_id)
            return [group for group in groups if group]
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return [[cpu_id] for cpu_id in cpu_ids]


def physical_cpu_count() -> int:
    """Return the number of physical cores visible to this process."""
    return max(1, len(physical_cpu_groups()))
