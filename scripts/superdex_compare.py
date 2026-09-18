#!/usr/bin/env python3
"""Numerical comparison of the SuperDex adapter against the direct SDK.

One tool consolidates the former qualification runners (FR3, registered bots,
floating hands, prefabs, rigid sweeps, scenes and controller components), their
control profiles and synthetic fixtures. Every supported native model found
under the local asset roots — or named explicitly on the command line — is
driven twice with identical inputs: once through the UniSim adapter and once
through a direct SuperDex SDK scene materialized independently in the same
process. Models run in isolated subprocesses so a native crash cannot hide
later results.

Check families preserved from the historical runners, with their original
tolerances and rollout lengths:

* state and body pose — inventories, default pose, masses, world transforms;
* controls — per-joint response matrices, routing, effort clipping;
* contacts — penetrating self-contact recovery and contact-query parity;
* cameras and controllers — authored camera parameters/poses and the three
  built-in native controllers (``--controller-config``);
* reset — whole-scene, selective, round trips and rest-spring targets;
* cleanup and recreation — repeated create/step/close cycles;
* composition — a robot plus nested rigid prefab fragments;
* batch — batch-mode equivalence against the serial rollout and the SDK.

Known-unsupported models (custom components, deformable actors, URDF inputs)
are classified with precise blockers, stay visible in every report and never
count as passed. Results print to the terminal by default; JSON reports,
limitation records and reproducibility fingerprints are written only with
``--out``, normally into the ignored ``results/superdex/`` directory:

    export SUPERDEX_ASSETS_PATH="$PWD/assets/superdex"
    uv run scripts/superdex_compare.py --all
    uv run scripts/superdex_compare.py assets/superdex/bots/arms/fr3_v2/fr3_v2.superdex_bot
    uv run scripts/superdex_compare.py --all --out results/superdex

Exit code 0 means every attempted model passed or was classified as a known
unsupported input; 1 marks unexpected failures, timeouts, native crashes, or a
previously qualified model that became unsupported.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from unisim.backend.superdex.assets import (  # noqa: E402
    resolve_reference,
)

INVENTORY_REPORT = REPOSITORY_ROOT / "docs" / "superdex-assets-inventory.json"

SIM_DT = 0.002
GRAVITY = (0.0, 0.0, -9.81)

# Rollout lengths (physics steps) and decimation preserved per family.
DECIMATION = 8  # 62.5 Hz control on 500 Hz physics
SWEEP_CTRL_STEPS = 130  # 130 * 8 = 1040 physics steps (bot PD sweep)
TRAJECTORY_STEPS = 1000  # scene/prefab/rigid rollouts
FLOATING_STEPS = 1040  # floating free-fall rollouts
CONTACT_STEPS = 400  # penetrating contact recovery
CLIP_STEPS = 200  # effort-clipping rollout
PROBE_STEPS = 5  # per-joint response probes
DISTURB_STEPS = 100  # reset disturbance
ISOLATION_CTRL_STEPS = 60  # environment isolation replay
LIFECYCLE_CYCLES = 3

# Tolerances preserved from the historical runners.
TOL_STRUCTURE = 1e-6
TOL_TRAJECTORY = 1e-7  # adapter vs direct SDK, identical code path
TOL_RESET = 1e-6
TOL_STATE_ATOL = 2e-6  # floating state comparisons
TOL_STATE_RTOL = 1e-6
TOL_SCENE = 3e-5  # native scene comparisons
TOL_RIGID = 2e-5  # generic rigid asset comparisons
TOL_CONTACT_FORCE = 2e-4

CONTACT_PENETRATION = 0.8  # rad beyond the authored lower limit
REACHABLE_FRACTION = 0.6

SUBPROCESS_TIMEOUT = 900

NATIVE_SUFFIXES = (".superdex_bot", ".superdex_bot_archive", ".mochi_scene", ".mochi_prefab")

# Status values: "passed" and the five distinguished non-pass outcomes.
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"  # numerical or lifecycle check failure
STATUS_BLOCKED = "blocked"  # adapter rejects; classified capability gap
STATUS_UNSUPPORTED = "unsupported"  # input format the adapter never accepted
STATUS_UNVERIFIED = "unverified"  # optional dependency unavailable locally
STATUS_CRASHED = "crashed"  # native subprocess terminated
STATUS_TIMEOUT = "timeout"


# --------------------------------------------------------------------- #
# Roots, discovery and fingerprints
# --------------------------------------------------------------------- #


def resolve_assets_root(cli_value: str | None) -> Path:
    """Resolve the local asset bundle root or exit with a setup diagnostic."""
    root = (
        Path(
            cli_value
            or os.environ.get("SUPERDEX_ASSETS_PATH")
            or REPOSITORY_ROOT / "assets" / "superdex"
        )
        .expanduser()
        .resolve()
    )
    if not root.is_dir() or not any(root.rglob(".superdex_root")):
        print(
            f"error: local SuperDex asset bundle not found at {root}.\n"
            "Cloning UniSim does not provide the ignored asset payloads.\n"
            "Copy the asset tree to assets/superdex (see docs/superdex.md) or set\n"
            "SUPERDEX_ASSETS_PATH to a local bundle root containing .superdex_root.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return root


def default_roots() -> list[Path]:
    roots = []
    for name in ("superdex", "superdex-physics"):
        candidate = REPOSITORY_ROOT / "assets" / name
        if candidate.is_dir():
            roots.append(candidate.resolve())
    return roots


def resolve_model_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file():
        return path.resolve()
    for root in [Path(os.environ.get("SUPERDEX_ASSETS_PATH", "assets/superdex")), *default_roots()]:
        if (root / path).is_file():
            return (root / path).resolve()
    raise FileNotFoundError(f"model not found: {path}")


def _root_of(path: Path) -> Path:
    for directory in path.parents:
        if (directory / ".superdex_root").is_file():
            return directory
    configured = os.environ.get("SUPERDEX_ASSETS_PATH")
    return Path(configured).expanduser().resolve() if configured else path.parent


@dataclass(frozen=True)
class ModelRef:
    path: Path
    kind: str  # bot | archive | scene | prefab | urdf | xml
    label: str  # stable reporting label (root name + relative path)

    @property
    def suffix(self) -> str:
        return self.path.suffix


def _kind_of(path: Path) -> str:
    return {
        ".superdex_bot": "bot",
        ".superdex_bot_archive": "archive",
        ".mochi_scene": "scene",
        ".mochi_prefab": "prefab",
        ".urdf": "urdf",
        ".xml": "xml",
    }.get(path.suffix.lower(), "unknown")


def discover_models(roots: list[Path]) -> list[ModelRef]:
    """Find every native model plus explicitly unsupported direct formats."""
    found: dict[Path, ModelRef] = {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            kind = _kind_of(path)
            if kind in {"unknown", "xml"} or path.name == "package.xml":
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            found.setdefault(
                path.resolve(),
                ModelRef(path.resolve(), kind, f"{root.name}:{relative.as_posix()}"),
            )
    return sorted(found.values(), key=lambda item: item.label)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def code_fingerprint(extra: Iterable[Path] = ()) -> dict[str, Any]:
    """Reproducibility fingerprint over the adapter sources and this tool."""
    files = sorted((REPOSITORY_ROOT / "src" / "unisim" / "backend" / "superdex").glob("*.py"))
    files = files + [Path(__file__).resolve(), *extra]
    digest = hashlib.sha256()
    names = []
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
        names.append(path.name)
    import importlib.metadata
    import platform

    packages = {}
    for name in ("superdex-physics-uni", "superdex-robotics-uni"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    physics = sys.modules.get("superdex.physics")
    return {
        "code_commit": _git("rev-parse", "HEAD"),
        "implementation_sha256": digest.hexdigest(),
        "implementation_files": names,
        "working_tree": _git("status", "--short"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sdk_packages": packages,
        "sdk_precision": ("float64" if physics.uses_double_precision() else "float32")
        if physics is not None
        else "see worker runtime",
        "inventory_sha256": hashlib.sha256(INVENTORY_REPORT.read_bytes()).hexdigest()
        if INVENTORY_REPORT.is_file()
        else None,
    }


# --------------------------------------------------------------------- #
# Static limitation classification (SDK-free)
# --------------------------------------------------------------------- #

ADAPTER_RESTRICTION = "adapter_restriction"
SDK_LIMITATION = "sdk_limitation"
UNRESOLVED_DEPENDENCY = "unresolved_dependency"

CHECKED_STATIC = "static_validation"
CHECKED_COMPONENTS = "resolved_sdk_components"
CHECKED_RUNTIME = "runtime_execution"

_CAMERA_SENSOR = "SENSOR_CAMERA"


@dataclass
class Limitation:
    path: str
    format: str
    unsupported_feature: str
    diagnostic: str
    blocker_class: str
    checked: str
    enabling_capability: str

    def record(self) -> dict[str, str]:
        return {
            "model": self.path,
            "format": self.format,
            "unsupported_feature": self.unsupported_feature,
            "diagnostic": self.diagnostic,
            "blocker_class": self.blocker_class,
            "checked": self.checked,
            "enabling_capability": self.enabling_capability,
        }


def _bot_component_blockers(document: dict) -> list[tuple[str, str]]:
    """Custom actuator/sensor component blockers in one authored bot document."""
    blockers = []
    for link in document.get("links", []):
        for item in link.get("actuators", []):
            kind = item.get("type", "")
            if kind:
                blockers.append((f"actuator component {item.get('name', '?')!r}", str(kind)))
        for item in link.get("sensors", []):
            kind = str(item.get("type") or item.get("typeName") or "")
            if kind and kind != _CAMERA_SENSOR:
                blockers.append((f"sensor component {item.get('name', '?')!r}", kind))
    return blockers


def _recipe_documents(path: Path, root: Path, stack: tuple[Path, ...] = ()) -> list[Path]:
    """Resolve recipe base/attachment chains without loading the SDK."""
    documents = [path]
    data = json.loads(path.read_text(encoding="utf-8"))
    references = []
    if "base" in data:
        references.append(str(data["base"]))
    for modification in data.get("modifications") or []:
        if not isinstance(modification, dict):
            continue
        for kind in ("AttachBot", "ReplaceLinkWithBot"):
            if kind in modification and isinstance(modification[kind], dict):
                references.append(str(modification[kind].get("path", "")))
    for reference in references:
        try:
            resolved, _ = resolve_reference(path.parent, reference, root)
        except Exception:  # noqa: BLE001 — recorded as an unresolved dependency
            continue
        if resolved in stack or not resolved.is_file() or resolved.suffix != ".superdex_bot":
            continue
        documents.extend(_recipe_documents(resolved, root, (*stack, resolved)))
    return documents


def classify_static(ref: ModelRef) -> Limitation | None:
    """Classify known-unsupported inputs from file content alone, SDK-free."""
    kind = ref.kind
    if kind == "urdf":
        return Limitation(
            path=_display_path(ref),
            format="urdf",
            unsupported_feature="direct URDF model loading",
            diagnostic=(
                "superdex model must be .superdex_bot, .superdex_bot_archive, "
                ".mochi_scene, .mochi_prefab or audited .xml MJCF"
            ),
            blocker_class=ADAPTER_RESTRICTION,
            checked=CHECKED_STATIC,
            enabling_capability=(
                "URDF ingestion (or conversion to the audited MJCF profile); the "
                "SDK URDF loader itself drops primitive box/cylinder/sphere geometry"
            ),
        )
    try:
        if kind == "archive":
            with zipfile.ZipFile(ref.path) as archive:
                metadata = json.loads(archive.read(".mochi_bot_archive_metadata"))
                document = json.loads(archive.read(metadata["target"]))
                documents = [document]
        elif kind == "bot":
            documents = [
                json.loads(p.read_text(encoding="utf-8"))
                for p in _recipe_documents(ref.path, _root_of(ref.path))
            ]
        else:
            data = json.loads(ref.path.read_text(encoding="utf-8"))
            actors = data.get("actors", {}) if isinstance(data, dict) else {}
            soft = list(actors.get("soft", [])) + list(actors.get("softSkinned", []))
            if soft:
                sample = soft[0]
                flavor = "soft" if actors.get("soft") else "softSkinned"
                return Limitation(
                    path=_display_path(ref),
                    format=ref.suffix,
                    unsupported_feature=f"deformable ({flavor}) actors",
                    diagnostic=(
                        f"superdex scene/prefab audit allows only articulated and rigid "
                        f"actors; actor {str(sample.get('name', '?'))!r} is deformable"
                    ),
                    blocker_class=ADAPTER_RESTRICTION,
                    checked=CHECKED_STATIC,
                    enabling_capability="soft-body actor support in the audited scene profile",
                )
            return None
    except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile):
        return None
    blockers = []
    for document in documents:
        blockers.extend(_bot_component_blockers(document))
    if not blockers:
        return None
    feature = "; ".join(f"{what} ({kind_})" for what, kind_ in blockers[:3])
    if len(blockers) > 3:
        feature += f"; … {len(blockers) - 3} more"
    inherited = len(documents) > 1
    return Limitation(
        path=_display_path(ref),
        format=ref.suffix,
        unsupported_feature="custom actuator/sensor components"
        + (" inherited through a recipe" if inherited else ""),
        diagnostic=feature,
        blocker_class=ADAPTER_RESTRICTION,
        checked=CHECKED_STATIC,
        enabling_capability=(
            "translation of authored custom components (SDK execution was not checked; the "
            "adapter supports only SENSOR_CAMERA components)"
        ),
    )


def _display_path(ref: ModelRef) -> str:
    for root in default_roots():
        try:
            return root.name + "/" + ref.path.relative_to(root).as_posix()
        except ValueError:
            continue
    return str(ref.path)


# --------------------------------------------------------------------- #
# Control profiles for registered bots
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class BotProfile:
    relpath: str
    effort_limits: tuple[float, ...] | None = None
    kp: float = 60.0
    armature_floor: float = 1e-3
    sweep_amplitude: float = 0.3
    sweep_period: float = 2.0
    contact_joint: int | None = 1
    max_qvel_bound: float | None = None


PROFILES: dict[str, BotProfile] = {
    "fr3": BotProfile(
        "bots/arms/fr3/fr3.superdex_bot",
        (87, 87, 87, 87, 12, 12, 12),
        kp=400,
        sweep_amplitude=0.35,
        max_qvel_bound=10.0,
    ),
    "fr3_v2": BotProfile(
        "bots/arms/fr3_v2/fr3_v2.superdex_bot",
        (87, 87, 87, 87, 12, 12, 12),
        kp=400,
        sweep_amplitude=0.35,
        max_qvel_bound=10.0,
    ),
    "fr3_v2_with_eyes": BotProfile(
        "bots/fun/arm_eyes_combos/fr3_v2_with_eyes.superdex_bot",
        (87, 87, 87, 87, 12, 12, 12),
        kp=400,
        sweep_amplitude=0.35,
        max_qvel_bound=10.0,
    ),
    "googly_eyes": BotProfile(
        "bots/fun/googly_eyes/googly_eyes.superdex_bot",
        (2.0,),
        kp=5.0,
        sweep_amplitude=0.5,
        contact_joint=None,
    ),
    "openarm_v20_torso": BotProfile(
        "bots/torsos/openarm_v20/openarm_v20_torso.superdex_bot", armature_floor=0.05
    ),
    "openarm_v20_combo": BotProfile(
        "bots/arm_hand_combos/openarm_v20/openarm_v20.superdex_bot", armature_floor=0.05
    ),
    "fr3_v2_2f_85": BotProfile(
        "bots/arm_hand_combos/fr3_v2_2f_85/fr3_v2_2f_85.superdex_bot",
        (87, 87, 87, 87, 12, 12, 12, 1.0),
        kp=400,
        sweep_amplitude=0.35,
        max_qvel_bound=10.0,
        contact_joint=None,
    ),
    "openarm_v20_left_arm": BotProfile(
        "bots/arms/openarm_v20/left/openarm_v20_left_arm.superdex_bot", armature_floor=0.05
    ),
    "openarm_v20_right_arm": BotProfile(
        "bots/arms/openarm_v20/right/openarm_v20_right_arm.superdex_bot", armature_floor=0.05
    ),
    "openarm_v20_wuji": BotProfile(
        "bots/arm_hand_combos/openarm_v20/openarm_v20_wuji.superdex_bot",
        kp=40,
        sweep_amplitude=0.25,
        armature_floor=0.05,
    ),
    "fr3_v2_allegro_v5_right": BotProfile(
        "bots/arm_hand_combos/fr3_v2_allegro_v5/right/fr3_v2_allegro_v5_right.superdex_bot",
        (87, 87, 87, 87, 12, 12, 12, *([7.0] * 16)),
        kp=400,
        sweep_amplitude=0.35,
        max_qvel_bound=10.0,
    ),
    "fr3_dg5f_short_left": BotProfile(
        "bots/arm_hand_combos/fr3_dg5f_short/left/fr3_dg5f_short_left.superdex_bot",
        (87, 87, 87, 87, 12, 12, 12, *([5.0] * 20)),
        kp=400,
        sweep_amplitude=0.35,
        max_qvel_bound=10.0,
        contact_joint=None,
    ),
    "fr3_dg5f_short_right": BotProfile(
        "bots/arm_hand_combos/fr3_dg5f_short/right/fr3_dg5f_short_right.superdex_bot",
        (87, 87, 87, 87, 12, 12, 12, *([5.0] * 20)),
        kp=400,
        sweep_amplitude=0.35,
        max_qvel_bound=10.0,
        contact_joint=None,
    ),
}


def profile_for(path: Path, root: Path) -> BotProfile:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        return BotProfile(path.name)
    for profile in PROFILES.values():
        if profile.relpath == relative:
            return profile
    return BotProfile(relative)


# Floating FREE-root bots use the gentle historical floating profile: tiny
# gains and amplitudes, and no penetrating self-contact probe (deep hand
# self-collisions trip the debug SDK's MOCHI contact assertion).
FLOATING_PROFILE = BotProfile(
    relpath="",
    kp=0.03,
    armature_floor=1e-6,
    sweep_amplitude=0.1,
    contact_joint=None,
    max_qvel_bound=None,
)


def profile_for_model(ref: ModelRef, reference: BotReference) -> BotProfile:
    """Registered profile when it pins effort limits, else a gentle fallback.

    Exotic joint trees (floating roots, spherical joints) need gentle gains
    and a small effort ceiling even when registered: unit-scale probes drive
    their self-contacts into the debug SDK's MOCHI assertion.
    """
    profile = profile_for(ref.path, _root_of(ref.path))
    spherical = any(size == 3 for _name, size in reference.column_sizes)
    if reference.floating and (profile.effort_limits is None or spherical):
        return FLOATING_PROFILE
    if spherical:
        return BotProfile(
            relpath="",
            kp=0.3,
            armature_floor=1e-5,
            sweep_amplitude=0.05,
            contact_joint=None,
            max_qvel_bound=None,
        )
    return profile


def gravity_for_model(reference: BotReference) -> tuple[float, float, float]:
    """World gravity for the comparison; near-massless spherical chains
    (e.g. Oculus XR hands) explode under load in the debug SDK build, so
    their rollout runs in zero gravity on both sides of the comparison."""
    if any(size == 3 for _name, size in reference.column_sizes):
        return (0.0, 0.0, 0.0)
    return GRAVITY


# Effort ceiling for the spherical fallback family (uniform, scaled to the
# adapter's coordinate count by resolve_efforts).
SPHERICAL_EFFORT = 0.1


def resolve_gains(profile: BotProfile, armature: np.ndarray, efforts: np.ndarray):
    kd = 2.0 * np.sqrt(profile.kp * np.maximum(armature, profile.armature_floor))
    return np.asarray(kd), np.asarray(efforts)


def sweep_amplitudes(profile: BotProfile, q0: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    width = np.minimum(np.abs(ranges[:, 1] - ranges[:, 0]), 2 * np.pi) * REACHABLE_FRACTION
    return np.minimum(profile.sweep_amplitude, 0.5 * width) * (width > 0)


def pd_torque(kp, kd, effort, target, q, v) -> np.ndarray:
    return np.clip(kp * (target - q) - kd * v, -effort, effort)


def sweep_target(q0: np.ndarray, amplitudes: np.ndarray, ctrl_index: int, period: float):
    t = ctrl_index * SIM_DT * DECIMATION
    phase = np.linspace(0.0, 3.0 * math.pi / 2.0, len(q0))
    return q0 + amplitudes * np.sin(2.0 * math.pi * t / period + phase)


# --------------------------------------------------------------------- #
# Independent direct-SDK reference implementations
# --------------------------------------------------------------------- #


def _load_prefab(bot_path: Path):
    """Load one bot prefab through the SDK, silencing its stdout chatter."""
    import superdex.robotics as robotics

    with contextlib.redirect_stdout(io.StringIO()):
        return robotics.load_bot_prefab_from_file(str(bot_path))


def authored_joint_metadata(cfg) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Extract (names, per-coordinate ranges, efforts, armature) from a prefab.

    Spherical joints expand to one row per rotation-vector axis, matching the
    adapter's coordinate layout; scalar joints contribute one row each.
    """
    import superdex.physics as physics

    jt = physics.ArticulatedJointType
    names, ranges, efforts, armature = [], [], [], []
    for joint in cfg.joints:
        if joint.type in (jt.HARD, jt.FREE):
            continue
        if joint.type == jt.REVOLUTE and joint.min_limit is not None:
            axis = np.asarray(joint.axis, dtype=float)
            axis = axis / np.linalg.norm(axis)
            ranges.append(
                [float(np.dot(joint.min_limit, axis)), float(np.dot(joint.max_limit, axis))]
            )
        elif joint.type == jt.SPHERICAL and joint.min_limit is not None:
            lo = np.asarray(joint.min_limit, dtype=float)
            hi = np.asarray(joint.max_limit, dtype=float)
            for component in range(3):
                ranges.append([float(lo[component]), float(hi[component])])
        else:
            ranges.append([-math.inf, math.inf])
        names.append(str(joint.name))
        efforts.append(float(joint.effort_limit))
        armature.append(float(joint.inertia or 0))
    return (
        tuple(names),
        np.asarray(ranges).reshape(-1, 2),
        np.asarray(efforts),
        np.asarray(armature),
    )


class BotReference:
    """A direct SuperDex SDK scene of one bot, driven with identical inputs.

    The native runtime is process-global and reference-counted by the adapter,
    so the reference scene lives in the same process as the backends under
    test: a backend must exist (holding the runtime) before construction, and
    this scene must be destroyed before the last backend closes.
    """

    def __init__(
        self, bot_path: Path, dt: float = SIM_DT, gravity: tuple[float, ...] = GRAVITY
    ) -> None:
        import superdex.physics as physics
        import superdex.robotics as robotics

        self.p, self.r = physics, robotics
        self.kind = self.controller = None
        self.dtype = np.float64 if physics.uses_double_precision() else np.float32
        self.cfg = _load_prefab(bot_path)
        self.scene = physics.create_scene("unisim_compare_reference")
        self.scene.set_gravity(gravity)
        self.context = robotics.create_context()
        self.bot = robotics.create_bot(self.scene, self.cfg, self.context)
        self.actor = self.bot.get_articulated_actor()
        self.links = [self.scene.get_actor(h) for h in self.actor.get_nested_link_actors()]
        self.names, self.ranges, self.efforts, self.armature = authored_joint_metadata(self.cfg)
        self.n = len(self.names)
        self.floating = bool(self.cfg.joints) and (
            self.cfg.joints[0].type == physics.ArticulatedJointType.FREE
        )
        # Native DOF indices of the authored active joints: the optional free
        # root contributes six leading DOFs, HARD links contribute none. The
        # adapter applies torques on exactly these columns.
        info = self.actor.get_articulated_shape_info()
        sizes = {
            physics.ArticulatedJointType.HARD: 0,
            physics.ArticulatedJointType.FREE: 6,
            physics.ArticulatedJointType.REVOLUTE: 1,
            physics.ArticulatedJointType.PRISMATIC: 1,
            physics.ArticulatedJointType.SPHERICAL: 3,
        }
        self.dofs = []
        self.column_sizes = []
        self.native_size = 0
        for kind in info.joint_types:
            self.native_size += sizes.get(kind, 0)
        offset = 0
        for joint in self.cfg.joints:
            size = sizes.get(joint.type, 0)
            if joint.type not in (
                physics.ArticulatedJointType.HARD,
                physics.ArticulatedJointType.FREE,
            ):
                self.dofs.extend(range(offset, offset + size))
                self.column_sizes.append((str(joint.name), size))
            offset += size
        self.dofs = np.asarray(self.dofs, dtype=np.int32)
        # Adapter actuator names: spherical joints expand to /x /y /z columns.
        self.axis_columns = []
        for name, size in self.column_sizes:
            self.axis_columns.extend(
                (name,) if size == 1 else tuple(f"{name}/{axis}" for axis in "xyz")
            )
        self.dt = dt
        # Whole-scene snapshot for resets: matches SuperDexBackend.reset(),
        # which restores the complete native scene, not only joint DOFs. A
        # floating root pose mutated by earlier probes would otherwise drift
        # relative to the adapter forever.
        self.snapshot = self.scene.capture_state()
        self.nv = self.native_size
        self.n = len(self.dofs)
        self.offset = self.nv - self.n

    def restore_default(self) -> None:
        """Restore the captured whole-scene default state."""
        self.scene.restore_state(self.snapshot, release_immediately=False)

    def qvel(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.empty(self.native_size, dtype=self.dtype)
        v = np.empty_like(q)
        self.actor.get_articulated_pose(q)
        self.actor.get_articulated_joint_velocities(v)
        return q, v

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Authored active-joint coordinates in authored order (spherical
        joints expose their three rotation-vector components)."""
        q, v = self.qvel()
        return q[self.dofs], v[self.dofs]

    def set_joint_state(self, qpos: np.ndarray, qvel: np.ndarray, *, restore: bool = True) -> None:
        """Mirror SuperDexBackend.set_state's native sequence exactly.

        With ``restore`` (default), first re-capture the whole-scene default —
        identical to the adapter's snapshot-restore semantics — then seed the
        joint DOFs, leaving any free root at its authored pose.
        """
        if len(self.dofs) != len(qpos) or len(self.dofs) != len(qvel):
            raise ValueError("reference joint state must match authored active joints")
        if restore:
            self.restore_default()
        native_q = np.empty(self.native_size, dtype=self.dtype)
        native_v = np.zeros_like(native_q)
        self.actor.get_articulated_pose(native_q)
        native_q[self.dofs] = np.asarray(qpos, dtype=self.dtype)
        native_v[self.dofs] = np.asarray(qvel, dtype=self.dtype)
        self.actor.set_articulated_pose_from_joints(native_q)
        self.actor.set_articulated_joint_velocities(native_v)
        self.actor.set_external_forces_on_dofs(
            np.arange(self.native_size, dtype=np.int32),
            np.zeros(self.native_size, dtype=self.dtype),
        )
        self.scene.step(0)

    def step(self, torque: np.ndarray, nsteps: int = 1) -> None:
        """Re-apply the joint torque before every substep on native DOFs."""
        if len(self.dofs) != len(torque):
            raise ValueError(
                f"reference torque vector length {len(torque)} does not match "
                f"{len(self.dofs)} authored active joints"
            )
        forces = np.zeros(self.native_size, dtype=self.dtype)
        forces[self.dofs] = np.asarray(torque, dtype=self.dtype)
        all_dofs = np.arange(self.native_size, dtype=np.int32)
        for _ in range(nsteps):
            self.actor.set_external_forces_on_dofs(all_dofs, forces)
            self.scene.step(self.dt)

    def aabb(self, link_index: int) -> np.ndarray:
        box = self.links[link_index].get_aabb_world()
        return np.asarray([box.min, box.max], dtype=np.float64)

    def link_pos(self, link_index: int) -> np.ndarray:
        return np.asarray(self.links[link_index].get_root_transform().translation, dtype=np.float64)

    def register_contact_queries(self) -> list:
        probes = []
        for link in self.links[2 : min(9, len(self.links))]:
            if link.is_static():
                continue
            link.register_query(self.p.QueryType.CONTACT_POINTS)
            probes.append(link)
        return probes

    def contact_point_count(self, probes: list) -> int:
        return sum(len(link.get_contact_points_world()) for link in probes)

    def configure(self, config):
        self.kind = config["type_name"]
        self.controller = self.bot.create_controller(self.kind)
        self.controller.configure_from_scene_entry(config["param_args"], config["init_args"])

    def step_controller(self, command, limits):
        if self.kind == "MOCHI_ARTICULATED_POSE":
            self.controller.compute_output(self.r.ControllerMochiArticulatedPoseObsv(), command)
            effort = np.zeros(self.nv, dtype=self.dtype)
        else:
            obsv = self.controller.get_current_observations_from_mochi()
            if self.kind == "BASIC_JSC_PD":
                obsv.dt = self.dt
            effort = np.asarray(self.controller.compute_output(obsv, command)).copy()
            if np.any(effort[: self.offset] != 0):
                raise ValueError("reference controller actuates floating root")
        effort[self.dofs] = np.clip(effort[self.dofs], -limits, limits)
        self.actor.set_external_forces_on_dofs(np.arange(self.nv, dtype=np.int32), effort)
        self.scene.step(self.dt)

    def close(self) -> None:
        import superdex.robotics as robotics

        if self.kind == "MOCHI_ARTICULATED_POSE" and self.actor.has_articulated_pose_controller():
            self.actor.remove_articulated_pose_controller()
        robotics.destroy_bot(self.scene, self.bot)
        self.scene.release_state(self.snapshot)
        self.p.destroy_scene(self.scene)
        self.bot = None
        self.scene = None


class SceneReference:
    """A direct SuperDex SDK world of a native scene/prefab, independently loaded."""

    def __init__(self, path: Path, dt: float = SIM_DT) -> None:
        import superdex.physics as physics

        self.p = physics
        self.path = path
        self.root = _root_of(path)
        self.scene = physics.create_scene("unisim_compare_scene_reference")
        self.cfg = physics.prefab.load_from_file(str(path), str(self.root))
        result = physics.prefab.add_to_scene(self.cfg, self.scene)
        self.actors = list(result.actors)
        self.articulated = [a for a in self.actors if a.get_type() == physics.ActorType.ARTICULATED]
        self.rigids = [a for a in self.actors if a.get_type() == physics.ActorType.RIGID]
        self.dt = dt
        self.snapshot = self.scene.capture_state()

    def restore(self) -> None:
        self.scene.restore_state(self.snapshot, release_immediately=False)

    def step(self, nsteps: int = 1) -> None:
        for _ in range(nsteps):
            self.scene.step(self.dt)

    def close(self) -> None:
        self.scene.release_state(self.snapshot)
        self.p.destroy_scene(self.scene)
        self.scene = None


# --------------------------------------------------------------------- #
# Synthetic fixtures (public SDK schemas, temporary directories only).
# Ported from the historical superdex_rigid_fixtures generator: the cube is
# closed with nonzero volume so mass/inertia stay well defined, links use
# colliderType None, and coupling/tendon fields match the SDK schemas.
# --------------------------------------------------------------------- #

CUBE = (
    "v -.1 -.1 -.1\nv .1 -.1 -.1\nv .1 .1 -.1\nv -.1 .1 -.1\n"
    "v -.1 -.1 .1\nv .1 -.1 .1\nv .1 .1 .1\nv -.1 .1 .1\n"
    "f 1 3 2\nf 1 4 3\nf 5 6 7\nf 5 7 8\nf 1 2 6\nf 1 6 5\n"
    "f 4 8 7\nf 4 7 3\nf 1 5 8\nf 1 8 4\nf 2 3 7\nf 2 7 6\n"
)


def _write_fixture(path, data):
    import json as _json

    path.write_text(_json.dumps(data, indent=2) + "\n")
    return path


def synthetic_fixtures(root: Path) -> dict[str, Path]:
    """Write the synthetic rigid/scene fixture set into ``root``.

    Exercises archived bots, cameras, couplings, spherical joints,
    multi-articulation scenes, nested composition, contacts and cross-root
    dependency tags — through public SDK schemas, without touching
    repository asset bytes.
    """
    import copy
    import json as _json
    import zipfile as _zip

    root.mkdir(parents=True, exist_ok=True)
    (root / "cube.obj").write_text(CUBE)
    (root / ".superdex_root").write_text("")
    actor = {
        "name": "robot",
        "joints": [
            {"name": "base", "type": "Hard"},
            {"name": "a", "type": "Revolute", "axis": [0, 0, 1], "inertia": 0.1},
            {"name": "b", "type": "Revolute", "axis": [0, 0, 1], "inertia": 0.1},
        ],
        "links": [
            {"name": "root", "mass": 1, "shape": "./cube.obj"},
            {
                "name": "left",
                "mass": 1,
                "parentLink": 0,
                "shape": "./cube.obj",
                "parentJointFromLink": {"translation": [0.3, 0, 0]},
            },
            {
                "name": "right",
                "mass": 1,
                "parentLink": 0,
                "shape": "./cube.obj",
                "parentJointFromLink": {"translation": [-0.3, 0, 0]},
            },
        ],
    }
    for link in actor["links"]:
        link["colliderType"] = "None"
    bot = copy.deepcopy(actor)
    bot["defaultPose"] = [0, 0]
    for joint in bot["joints"]:
        joint["effortLimit"] = 1
    paths: dict[str, Path] = {
        "plain": _write_fixture(root / "plain.superdex_bot", bot),
    }
    transmission = copy.deepcopy(bot)
    transmission["linearTransmissions"] = [
        {
            "name": "coupler",
            "jointIndices": [1, 2],
            "jointCoefficients": [0.1, -0.1],
            "jointAxisDisps": [0, 0],
            "targetDisplacement": 0.01,
            "stiffness": 20,
            "damping": 1,
            "allowCompressiveForce": True,
        }
    ]
    tendon = copy.deepcopy(bot)
    tendon["spatialTendons"] = [
        {
            "name": "tendon",
            "routingElements": [
                {"type": "Waypoint", "index": 1, "localPosition": [0, 0.1, 0]},
                {"type": "Waypoint", "index": 2, "localPosition": [0, -0.1, 0]},
                {"type": "LinearJoint", "index": 2, "coefficient": 0.02},
            ],
            "targetDisplacement": -0.05,
            "stiffness": 20,
            "damping": 1,
            "allowCompressiveForce": False,
        }
    ]
    paths["transmission"] = _write_fixture(root / "transmission.superdex_bot", transmission)
    paths["tendon"] = _write_fixture(root / "tendon.superdex_bot", tendon)
    ball_bot = copy.deepcopy(bot)
    ball_bot["joints"][1].update(type="Spherical", minLimit=[-0.4] * 3, maxLimit=[0.4] * 3)
    ball_bot["defaultPose"] = [0] * 4
    paths["ball"] = _write_fixture(root / "ball.superdex_bot", ball_bot)
    archive = root / "robot.superdex_bot_archive"
    with _zip.ZipFile(archive, "w", _zip.ZIP_DEFLATED) as out:
        for name in ("plain.superdex_bot", "cube.obj", ".superdex_root"):
            out.writestr(_zip.ZipInfo(name), (root / name).read_bytes())
        out.writestr(
            _zip.ZipInfo(".mochi_bot_archive_metadata"),
            _json.dumps(
                {
                    "target": "plain.superdex_bot",
                    "date": "2026-09-17",
                    "botHash": "synthetic",
                    "commitHash": "synthetic",
                    "warnings": [],
                }
            ),
        )
    paths["archive"] = archive
    camera_bot = copy.deepcopy(ball_bot)
    camera_bot["links"][1]["sensors"] = [
        {
            "name": "camera",
            "type": "SENSOR_CAMERA",
            "params": "./camera.superdex_sensor",
            "parentFromSensor": {"translation": [0.1, 0.2, 0.3]},
        }
    ]
    camera_archive = root / "camera.superdex_bot_archive"
    with _zip.ZipFile(camera_archive, "w", _zip.ZIP_DEFLATED) as out:
        for name, contents in {
            ".superdex_root": "",
            "cube.obj": CUBE,
            "camera.superdex_bot": _json.dumps(camera_bot),
            "camera.superdex_sensor": _json.dumps({"imageWidth": 320, "imageHeight": 240}),
            ".mochi_bot_archive_metadata": _json.dumps(
                {
                    "target": "camera.superdex_bot",
                    "date": "2026-09-17",
                    "botHash": "synthetic",
                    "commitHash": "synthetic",
                    "warnings": [],
                }
            ),
        }.items():
            out.writestr(_zip.ZipInfo(name), contents)
    paths["camera_archive"] = camera_archive
    spherical = copy.deepcopy(actor)
    spherical["joints"][0]["type"] = "Free"
    spherical["joints"][1].update(type="Spherical", minLimit=[-0.4] * 3, maxLimit=[0.4] * 3)
    spherical["joints"][0]["parentLinkFromJoint"] = {
        "translation": [0.4, 0.3, 0.2],
        "rotation": [0, 0, 0.382683432365, 0.923879532511],
    }
    spherical["links"][0]["parentJointFromLink"] = {"translation": [0.1, 0.2, 0]}
    spherical["jointVelocities"] = [0.01] * 10
    paths["spherical"] = _write_fixture(
        root / "spherical.mochi_prefab",
        {"actors": {"articulated": [spherical]}, "scene": {"gravity": [0, 0, 0]}},
    )
    paths["multiple"] = _write_fixture(
        root / "multiple.mochi_scene",
        {
            "prefabs": [
                {"name": "first", "path": "./spherical.mochi_prefab"},
                {
                    "name": "second",
                    "path": "./spherical.mochi_prefab",
                    "translation": [2, 0, 0],
                    "rotation": [0, 0, 0.707106781186, 0.707106781186],
                },
            ],
            "scene": {"gravity": [0, 0, 0]},
        },
    )
    paths["nested"] = _write_fixture(
        root / "nested.mochi_scene",
        {
            "prefabs": [{"name": "outer", "path": "./multiple.mochi_scene"}],
            "scene": {"gravity": [0, 0, 0]},
        },
    )
    paths["contact"] = _write_fixture(
        root / "contact.mochi_scene",
        {
            "actors": {
                "rigid": [
                    {"name": "cube", "shape": "./cube.obj", "mass": 1, "translation": [0, 0.5, 0]},
                    {"name": "floor", "shape": "./cube.obj", "scale": [5, 1, 5], "isStatic": True},
                ]
            },
            "scene": {"gravity": [0, -9.81, 0]},
        },
    )
    dep = root / "dependencies"
    dep.mkdir(exist_ok=True)
    (dep / ".superdex_root").write_text("")
    (dep / "cube.obj").write_text(CUBE)
    external = root / "external"
    external.mkdir(exist_ok=True)
    _write_fixture(external / ".superdex_root", {"@shapes": "../dependencies"})
    external_bot = copy.deepcopy(bot)
    for link in external_bot["links"]:
        link["shape"] = "@shapes/cube.obj"
    paths["external"] = _write_fixture(external / "external.superdex_bot", external_bot)
    return paths


# --------------------------------------------------------------------- #
# Check suites
# --------------------------------------------------------------------- #


@dataclass
class CheckRun:
    """One model's check ledger."""

    model: str
    kind: str
    checks: dict[str, dict] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def record(self, name: str, passed: bool, evidence: dict) -> bool:
        self.checks[name] = {"passed": bool(passed), **evidence}
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}" + ("" if passed else f" — {evidence}"))
        if not passed:
            self.failures.append(name)
        return bool(passed)


def make_backend(
    path: Path,
    num_envs: int,
    *,
    serial: bool = False,
    effort_limits=None,
    controlled_joints=None,
    gravity=None,
):
    from unisim import create_backend
    from unisim.scene import SceneCfg

    kwargs: dict[str, Any] = {}
    if serial:
        kwargs.update(superdex_execution_mode="serial", superdex_num_workers=0)
    if effort_limits is not None:
        kwargs["superdex_effort_limits"] = effort_limits
    if controlled_joints is not None:
        kwargs["superdex_controlled_joints"] = controlled_joints
    backend = create_backend("superdex", SceneCfg(str(path)), num_envs, SIM_DT, **kwargs)
    if gravity is not None and tuple(gravity) != GRAVITY:
        backend.set_gravity(gravity)
    return backend


def resolve_efforts(ref: ModelRef, reference: BotReference) -> list[float]:
    """Profile limits first, then authored finite limits, then a gentle 0.1.

    The fallback is sized to the adapter's actuator count, which expands
    spherical joints into three coordinates (unlike ``reference.n``).
    """
    profile = profile_for_model(ref, reference)
    if profile.effort_limits is not None:
        return list(profile.effort_limits)
    authored = np.asarray(reference.efforts, dtype=float)
    if authored.size and np.isfinite(authored).all() and np.all(authored > 0):
        # Authored efforts are per joint; the adapter actuator count expands
        # spherical joints to three coordinates.
        expanded: list[float] = []
        for index, (_name, size) in enumerate(reference.column_sizes):
            expanded.extend([float(authored[index])] * max(1, size))
        return expanded
    if any(size == 3 for _name, size in reference.column_sizes):
        return [SPHERICAL_EFFORT] * len(reference.axis_columns)
    return [1.0] * len(reference.axis_columns)


def check_bot(ref: ModelRef, run: CheckRun, steps: int = SWEEP_CTRL_STEPS * DECIMATION) -> None:
    """Fixed-base and floating bot checks against a direct SDK scene.

    Advanced native assets (spherical joints, couplings, archives) are
    serial-only; they run the same checks in serial mode with one environment,
    skipping the two-environment isolation replay that batch mode provides.
    """
    probe_reference = BotReference(ref.path)
    reference = BotReference(ref.path, gravity=gravity_for_model(probe_reference))
    profile = profile_for_model(ref, probe_reference)
    probe_reference.close()
    del probe_reference
    holder = None
    serial_backend = None
    try:
        efforts = resolve_efforts(ref, reference)
        model_gravity = gravity_for_model(reference)
        probe = make_backend(ref.path, 1, serial=True, effort_limits=efforts, gravity=model_gravity)
        serial_only = bool(probe.model.serial_only)
        num_envs = 1 if serial_only else 2
        probe.close()
        from unisim import create_backend
        from unisim.scene import SceneCfg

        if serial_only:
            holder = create_backend(
                "superdex",
                SceneCfg(str(ref.path)),
                1,
                SIM_DT,
                superdex_execution_mode="serial",
                superdex_num_workers=0,
                superdex_effort_limits=efforts,
            )
        else:
            holder = create_backend(
                "superdex",
                SceneCfg(str(ref.path)),
                num_envs,
                SIM_DT,
                superdex_effort_limits=efforts,
            )
        if model_gravity != GRAVITY:
            # Both sides of the comparison use the same world gravity.
            holder.set_gravity(model_gravity)
        # The adapter's coordinate count: spherical joints expand to 3 columns.
        n = len(reference.axis_columns)
        # authored_joint_metadata already emits one range row per coordinate.
        reference_ranges = reference.ranges

        # ---- structure --------------------------------------------------
        model = holder.model
        authored_links = tuple(str(link.name) for link in reference.cfg.links)
        bodies_ok = tuple(model.body_names)[1:] == authored_links
        # Adapter joints expand spherical joints to xyz columns; compare both
        # sides through the adapter's joint coordinate list.
        joints_ok = tuple(holder.get_actuator_names()) == tuple(reference.axis_columns)
        ctrl_order_ok = tuple(holder.get_actuator_joint_names()) == tuple(reference.axis_columns)
        q0 = holder.get_default_qpos()
        adapter_q0 = holder.get_dof_pos()[0]
        ref_q0, _ = reference.joint_state()
        q0_dev = float(np.max(np.abs(adapter_q0 - ref_q0))) if n else 0.0
        masses = holder.get_body_mass()
        mass_max = 0.0
        for body_id in range(1, len(model.body_names)):
            ref_link = reference.links[body_id - 1]
            if ref_link.is_static():
                continue
            mass_max = max(mass_max, abs(float(masses[body_id]) - float(ref_link.get_mass())))
        body_ids = np.arange(len(model.body_names))
        pos_w = holder.get_body_pos_w(body_ids)[0]
        ref_pos = np.array([reference.link_pos(i) for i in range(len(reference.links))])
        pos_dev = float(np.max(np.abs(pos_w[1:] - ref_pos)))
        if n:
            adapter_ranges = holder.get_joint_range()
            range_gap = adapter_ranges - reference_ranges
            # inf - inf would be nan; matching infinite limits are equal.
            both_inf = np.isinf(adapter_ranges) & np.isinf(reference_ranges)
            range_gap = np.where(both_inf, 0.0, range_gap)
            ranges_dev = float(np.max(np.abs(range_gap)))
        else:
            ranges_dev = 0
        run.record(
            "structure",
            bodies_ok
            and joints_ok
            and ctrl_order_ok
            and q0_dev <= TOL_STRUCTURE
            and mass_max <= TOL_STRUCTURE
            and pos_dev <= TOL_STRUCTURE
            and ranges_dev <= TOL_STRUCTURE,
            {
                "body_inventory_ok": bodies_ok,
                "joint_inventory_ok": joints_ok,
                "control_order_ok": ctrl_order_ok,
                "default_qpos_max_dev_rad": q0_dev,
                "dynamic_body_mass_max_dev_kg": mass_max,
                "body_pos_max_dev_m": pos_dev,
                "joint_range_max_dev_rad": ranges_dev,
            },
        )

        # Spherical kd gains repeat per coordinate (authored armature is per
        # joint); expand to the adapter's coordinate layout. Passive bots with
        # no active joints keep an empty gain vector.
        if reference.column_sizes:
            expanded_armature = np.concatenate(
                [
                    np.full(
                        max(1, size),
                        float(
                            reference.armature[index] if index < len(reference.armature) else 0.0
                        ),
                    )
                    for index, (_name, size) in enumerate(reference.column_sizes)
                ]
            )
        else:
            expanded_armature = np.zeros(0)
        kd, effort = resolve_gains(profile, expanded_armature, np.asarray(efforts))
        kp = profile.kp
        serial_backend = create_backend(
            "superdex",
            SceneCfg(str(ref.path)),
            1,
            SIM_DT,
            superdex_execution_mode="serial",
            superdex_num_workers=0,
            superdex_effort_limits=efforts,
        )
        if model_gravity != GRAVITY:
            serial_backend.set_gravity(model_gravity)

        # ---- controls: per-joint response matrix + effort clipping --------
        if n:
            base_effort = float(np.min(effort))
            probe = min(5.0, base_effort * 0.5) if base_effort > 0 else 0.5
            spherical_chain = any(size == 3 for _name, size in reference.column_sizes)
            if spherical_chain:
                # Near-massless spherical chains (e.g. Oculus XR hands)
                # explode under probes sized for ordinary arms; scale by the
                # smallest joint inertia, floored at 1e-4 N·m, which still
                # yields a measurable response on every column. The same
                # ceiling becomes the effort limit so clipping stays exact.
                armature = float(np.min(reference.armature)) if reference.armature.size else 1.0
                ceiling = max(1e-4, armature * 0.05)
                if ceiling < base_effort:
                    efforts = [ceiling] * n
                    effort = np.asarray(efforts)
                    probe = ceiling * 0.5
            # Match the reference's float32 precision so equal torques compare
            # exactly even on tiny floating-hand limits.
            probe = np.float32(probe).item()

            def adapter_matrix() -> np.ndarray:
                backend = make_backend(
                    ref.path, 1, serial=serial_only, effort_limits=efforts, gravity=model_gravity
                )
                try:
                    backend.reset()
                    backend.step(np.zeros((1, n)), PROBE_STEPS)
                    base = backend.get_dof_vel()[0].copy()
                    matrix = np.zeros((n, n))
                    for i in range(n):
                        backend.reset()
                        torque = np.zeros((1, n))
                        torque[0, i] = probe
                        backend.step(torque, PROBE_STEPS)
                        matrix[i] = backend.get_dof_vel()[0] - base
                    return matrix
                finally:
                    backend.close()

            def reference_matrix() -> np.ndarray:
                holder.reset()
                reference.restore_default()
                NativeComparison(holder, reference.scene, [reference.actor]).seed_reference()
                reference.step(np.zeros(n), PROBE_STEPS)
                _, base = reference.joint_state()
                matrix = np.zeros((n, n))
                for i in range(n):
                    reference.restore_default()
                    NativeComparison(holder, reference.scene, [reference.actor]).seed_reference()
                    torque = np.zeros(n)
                    torque[i] = probe
                    reference.step(torque, PROBE_STEPS)
                    _, velocity = reference.joint_state()
                    matrix[i] = velocity - base
                return matrix

            def adapter_matrix_f32() -> np.ndarray:
                """Second adapter pass driven through the same float32 cast the
                direct SDK reference applies, for an exact comparison."""
                backend = make_backend(
                    ref.path, 1, serial=serial_only, effort_limits=efforts, gravity=model_gravity
                )
                try:
                    backend.reset()
                    backend.step(np.zeros((1, n)), PROBE_STEPS)
                    base = backend.get_dof_vel()[0].copy()
                    matrix = np.zeros((n, n))
                    for i in range(n):
                        backend.reset()
                        torque = np.zeros(n)
                        torque[i] = probe
                        backend.step(np.asarray(torque, dtype=np.float32)[None, :], PROBE_STEPS)
                        matrix[i] = backend.get_dof_vel()[0] - base
                    return matrix
                finally:
                    backend.close()

            if reference.floating:
                adapter = adapter_matrix_f32()
                adapter_dev = float(np.max(np.abs(adapter_matrix() - adapter)))
            else:
                adapter = adapter_matrix()
                adapter_dev = 0.0
            sdk_matrix = reference_matrix()
            wiring_ok = bool(np.all(np.diag(adapter) != 0.0))
            matrix_dev = float(np.max(np.abs(adapter - sdk_matrix)))

            saturated = make_backend(
                ref.path, 1, serial=serial_only, effort_limits=efforts, gravity=model_gravity
            )
            capped = make_backend(
                ref.path, 1, serial=serial_only, effort_limits=efforts, gravity=model_gravity
            )
            try:
                saturated.step(np.full((1, n), 1e4), CLIP_STEPS)
                capped.step(effort[None, :].astype(float), CLIP_STEPS)
                clip_ok = bool(
                    np.array_equal(saturated.get_state()["qpos"], capped.get_state()["qpos"])
                )
            finally:
                saturated.close()
                capped.close()
            run.record(
                "control",
                wiring_ok
                and matrix_dev
                <= max(TOL_TRAJECTORY if not reference.floating else TOL_RIGID, adapter_dev + 1e-9)
                and clip_ok,
                {
                    "probe_torque_nm": probe,
                    "control_columns_reach_own_joint": wiring_ok,
                    "response_matrix_vs_sdk_max_dev_rad_s": matrix_dev,
                    "float64_vs_float32_drive_dev": adapter_dev,
                    "effort_clip_exact": clip_ok,
                },
            )

        # ---- trajectory: batch + serial vs SDK (PD sweep or free fall) ----
        # Floating FREE-root bots use the historical floating tolerances
        # (atol 2e-6 / rtol 1e-6): tiny gains make one float32 ulp visible.
        trajectory_tol = TOL_STATE_ATOL if reference.floating else TOL_TRAJECTORY
        amplitudes = np.asarray(
            sweep_amplitudes(
                profile,
                q0[-n:] if n else q0[:0],
                holder.get_joint_range() if n else np.zeros((0, 2)),
            )
        )
        holder.reset()
        serial_backend.reset()
        reference.set_joint_state(q0[-n:], np.zeros(n)) if n else reference.set_joint_state(
            np.zeros(0), np.zeros(0)
        )
        trajectory_audits = [
            NativeComparison(b, reference.scene, [reference.actor])
            for b in (holder, serial_backend)
        ]
        trajectory_audits[0].seed_reference()
        batch_dev = serial_dev = max_qvel = camera_dev = 0.0
        ctrl_steps = steps // DECIMATION
        for k in range(ctrl_steps):
            if n:
                target = sweep_target(q0[-n:], amplitudes, k, profile.sweep_period)
                tau_b = pd_torque(
                    kp, kd, effort, target, holder.get_dof_pos()[0], holder.get_dof_vel()[0]
                )
                tau_s = pd_torque(
                    kp,
                    kd,
                    effort,
                    target,
                    serial_backend.get_dof_pos()[0],
                    serial_backend.get_dof_vel()[0],
                )
                q_ref, v_ref = reference.joint_state()
                tau_r = pd_torque(kp, kd, effort, target, q_ref, v_ref)
            else:
                tau_b = tau_s = tau_r = np.zeros((0,))
            holder.step(np.repeat(tau_b[None, :], holder.num_envs, axis=0), DECIMATION)
            serial_backend.step(tau_s[None, :], DECIMATION)
            reference.step(tau_r, DECIMATION)
            for audit in trajectory_audits:
                audit.sample()
            camera_dev = max(camera_dev, check_cameras(holder, reference.bot)["pose_max_dev"])
            q_ref, v_ref = reference.joint_state()
            if n:
                batch_dev = max(batch_dev, float(np.max(np.abs(holder.get_dof_pos()[0] - q_ref))))
                serial_dev = max(
                    serial_dev, float(np.max(np.abs(serial_backend.get_dof_pos()[0] - q_ref)))
                )
                max_qvel = max(max_qvel, float(np.max(np.abs(v_ref))))
        state = holder.get_state()
        finite = bool(np.isfinite(state["qpos"]).all() and np.isfinite(state["qvel"]).all())
        bound = profile.max_qvel_bound
        physics_steps = ctrl_steps * DECIMATION
        run.record(
            "trajectory_equivalence",
            batch_dev <= trajectory_tol
            and serial_dev <= trajectory_tol
            and finite
            and (bound is None or max_qvel < bound)
            and physics_steps == ctrl_steps * DECIMATION
            and (physics_steps >= 1000 or steps < SWEEP_CTRL_STEPS * DECIMATION),
            {
                "physics_steps": physics_steps,
                "batch_vs_sdk_max_dev_rad": batch_dev,
                "serial_vs_sdk_max_dev_rad": serial_dev,
                "reference_max_abs_qvel_rad_s": max_qvel,
                "finite_state": finite,
                "tolerance": trajectory_tol,
            },
        )

        run.record(
            "native_state_and_bodies",
            all(
                a.passed(TOL_RIGID if reference.floating else TOL_TRAJECTORY)
                for a in trajectory_audits
            ),
            {
                key: max(a.errors[key] for a in trajectory_audits)
                for key in trajectory_audits[0].errors
            },
        )
        run.record(
            "cameras",
            camera_dev <= TOL_RESET,
            {"camera_count": len(holder.get_camera_names()), "pose_max_dev": camera_dev},
        )

        # ---- contacts: penetrating self-contact recovery ------------------
        # Uses a fresh reference scene: registering CONTACT_POINTS queries on
        # a scene whose earlier rollouts produced deep self-collisions trips a
        # native MOCHI assertion in the debug SDK build when it later steps a
        # penetrating pose (an SDK behavior recorded in the report, not an
        # adapter defect — the adapter's own scenes never assert here). A
        # fresh scene whose history is only the penetrating probe is stable.
        if profile.contact_joint is not None and n and profile.contact_joint < n:
            joint = profile.contact_joint
            reference.close()
            reference = BotReference(ref.path)
            lo = float(reference.ranges[joint][0])
            if math.isfinite(lo):
                probes = reference.register_contact_queries()
                q_pen = q0.copy()
                q_pen[-n + joint] = lo - CONTACT_PENETRATION
                holder.set_state(np.array([0]), q_pen[None, :], np.zeros((1, holder.model.nv)))
                reference.set_joint_state(q_pen[-n:], np.zeros(n))
                set_agreement = float(
                    np.max(np.abs(holder.get_state()["qpos"][0][-n:] - reference.joint_state()[0]))
                )
                zero = np.zeros((holder.num_envs, n))
                contact_steps = 0
                max_dev = 0.0
                for _ in range(CONTACT_STEPS):
                    holder.step(zero, 1)
                    reference.step(np.zeros(n), 1)
                    q_ref, _ = reference.joint_state()
                    max_dev = max(
                        max_dev, float(np.max(np.abs(holder.get_state()["qpos"][0][-n:] - q_ref)))
                    )
                    if reference.contact_point_count(probes) > 0:
                        contact_steps += 1
                run.record(
                    "contact_recovery",
                    set_agreement <= TOL_TRAJECTORY
                    and contact_steps > 0
                    and max_dev <= TOL_TRAJECTORY,
                    {
                        "penetrating_joint": reference.names[joint],
                        "set_state_agreement_rad": set_agreement,
                        "reference_contact_steps": contact_steps,
                        "adapter_vs_sdk_max_dev_rad": max_dev,
                    },
                )
            else:
                run.record("contact_recovery", True, {"skipped": "joint has no finite lower limit"})
        else:
            run.record(
                "contact_recovery", True, {"skipped": "no representative self-contact geometry"}
            )

        # ---- reset ------------------------------------------------------
        if n:
            probe = float(np.min(effort)) * 0.5
            holder.reset()
            holder.step(np.full((holder.num_envs, n), probe), DISTURB_STEPS)
            holder.reset()
            full_dev = float(np.max(np.abs(holder.get_state()["qpos"] - q0)))
            reference.set_joint_state(np.asarray(q0[-n:], dtype=np.float32), np.zeros(n))
            ref_q0, _ = reference.joint_state()
            default_dev = float(np.max(np.abs(holder.get_state()["qpos"][0][-n:] - ref_q0)))
            holder.reset()
            holder.step(np.full((holder.num_envs, n), probe), DISTURB_STEPS)
            before = holder.get_state()
            holder.reset(np.array([0]))
            after = holder.get_state()
            selective_dev = float(np.max(np.abs(after["qpos"][0] - q0)))
            if serial_only:
                env1_untouched = True
                roundtrip_q = roundtrip_v = 0.0
                q_arb = q0[-n:] + np.linspace(0.1, -0.1, n)
                v_arb = np.linspace(0.05, -0.05, n)
                full_q = q0[None, :].copy()
                full_v = np.zeros((1, holder.model.nv))
                full_q[0, -n:] = q_arb
                full_v[0, -n:] = v_arb
                holder.set_state(np.array([0]), full_q, full_v)
                rb = holder.get_state()
                roundtrip_q = float(np.max(np.abs(rb["qpos"][0][-n:] - q_arb)))
                roundtrip_v = float(np.max(np.abs(rb["qvel"][0][-n:] - v_arb)))
            else:
                env1_untouched = bool(
                    np.array_equal(after["qpos"][1], before["qpos"][1])
                    and np.array_equal(after["qvel"][1], before["qvel"][1])
                )
                q_arb = q0[-n:] + np.linspace(0.1, -0.1, n)
                v_arb = np.linspace(0.05, -0.05, n)
                full_q = np.repeat(q0[None, :], 2, axis=0)
                full_v = np.zeros((2, holder.model.nv))
                full_q[1, -n:] = q_arb
                full_v[1, -n:] = v_arb
                holder.set_state(np.array([1]), full_q[1:2], full_v[1:2])
                rb = holder.get_state()
                roundtrip_q = float(np.max(np.abs(rb["qpos"][1][-n:] - q_arb)))
                roundtrip_v = float(np.max(np.abs(rb["qvel"][1][-n:] - v_arb)))
            run.record(
                "reset",
                full_dev <= TOL_RESET
                and default_dev <= TOL_RESET
                and selective_dev <= TOL_RESET
                and env1_untouched
                and roundtrip_q <= TOL_RESET
                and roundtrip_v <= TOL_RESET,
                {
                    "whole_reset_max_dev_rad": full_dev,
                    "reset_matches_sdk_default_rad": default_dev,
                    "selective_reset_env0_dev_rad": selective_dev,
                    "selective_reset_env1_untouched": env1_untouched,
                    "roundtrip_qpos_dev_rad": roundtrip_q,
                    "roundtrip_qvel_dev_rad": roundtrip_v,
                },
            )
        else:
            run.record("reset", True, {"skipped": "no actuated joints"})

        # ---- isolation ----------------------------------------------------
        if n and not serial_only:
            single = make_backend(ref.path, 1, effort_limits=efforts, gravity=model_gravity)

            def replay(env_index: int, hold: bool) -> float:
                holder.reset()
                single.reset()
                deviation = 0.0
                for k in range(ISOLATION_CTRL_STEPS):
                    target = sweep_target(q0[-n:], amplitudes, k, profile.sweep_period)
                    own_target = q0[-n:] if hold else target
                    tau_own = pd_torque(
                        kp,
                        kd,
                        effort,
                        own_target,
                        holder.get_dof_pos()[env_index],
                        holder.get_dof_vel()[env_index],
                    )
                    commands = np.zeros((holder.num_envs, n))
                    commands[env_index] = tau_own
                    other = 1 - env_index
                    other_target = q0[-n:] if not hold else target
                    commands[other] = pd_torque(
                        kp,
                        kd,
                        effort,
                        other_target,
                        holder.get_dof_pos()[other],
                        holder.get_dof_vel()[other],
                    )
                    holder.step(commands, DECIMATION)
                    tau_single = pd_torque(
                        kp, kd, effort, own_target, single.get_dof_pos()[0], single.get_dof_vel()[0]
                    )
                    single.step(tau_single[None, :], DECIMATION)
                    deviation = max(
                        deviation,
                        float(
                            np.max(
                                np.abs(holder.get_dof_pos()[env_index] - single.get_dof_pos()[0])
                            )
                        ),
                    )
                return deviation

            try:
                dev0 = replay(env_index=0, hold=False)
                dev1 = replay(env_index=1, hold=True)
            finally:
                single.close()
            run.record(
                "isolation",
                dev0 <= TOL_TRAJECTORY and dev1 <= TOL_TRAJECTORY,
                {"env0_vs_single_max_dev_rad": dev0, "env1_vs_single_max_dev_rad": dev1},
            )
        else:
            run.record(
                "isolation",
                True,
                {
                    "skipped": "no actuated joints"
                    if not n
                    else "serial-only advanced asset runs one environment"
                },
            )

    finally:
        if serial_backend is not None:
            serial_backend.close()
        if holder is not None:
            holder.close()
        if reference is not None:
            reference.close()

    # ---- lifecycle (independent backends) --------------------------------
    deviations = []
    for _ in range(LIFECYCLE_CYCLES):
        backend = make_backend(
            ref.path, 1, serial=serial_only, effort_limits=efforts, gravity=model_gravity
        )
        try:
            n_act = backend.num_actuators
            backend.step(np.zeros((1, n_act)), 10)
            backend.reset()
            gap = np.abs(backend.get_state()["qpos"] - backend.get_default_qpos())
            deviations.append(float(np.max(gap)) if gap.size else 0.0)
        finally:
            backend.close()
    run.record(
        "lifecycle",
        all(dev <= TOL_RESET for dev in deviations),
        {"cycles": len(deviations), "reset_deviations_rad": deviations},
    )


def maximum(value):
    return float(np.max(np.abs(value), initial=0))


def native_state(actor):
    import superdex.physics as p

    dtype = np.float64 if p.uses_double_precision() else np.float32
    q = np.empty(actor.get_num_dofs(), dtype=dtype)
    v = np.empty_like(q)
    actor.get_articulated_pose(q)
    actor.get_articulated_joint_velocities(v)
    return q, v


def native_actors(backend, env=0):
    actor = backend._actors[env]
    return list(actor.actors) if hasattr(actor, "actors") else [actor]


def native_controls(actors):
    """Derive coordinate names and force routes exclusively from the SDK."""
    from collections import Counter

    import superdex.physics as p

    raw = [a.get_name() or f"articulation_{i}" for i, a in enumerate(actors)]
    counts = Counter(raw)
    names, routes = [], []
    for ai, actor in enumerate(actors):
        name = raw[ai] + (f"#{ai}" if counts[raw[ai]] > 1 else "")
        prefix = f"{name}/" if len(actors) > 1 else ""
        info = actor.get_articulated_shape_info()
        for ji in range(len(actor.get_nested_link_actors())):
            if info.joint_types[ji] in (p.ArticulatedJointType.HARD, p.ArticulatedJointType.FREE):
                continue
            size = info.dof_info[ji].get_size()
            joint = str(info.joint_names[ji]) or f"joint_{ji}"
            for axis in range(size):
                names.append(prefix + joint + (f"/{'xyz'[axis]}" if size == 3 else ""))
                routes.append((ai, info.dof_info[ji].offset + axis))
    return names, routes


def apply_native_control(actors, routes, command):
    """Submit one force vector per actor, including zeros on passive DOFs."""
    forces = [np.zeros_like(native_state(a)[0]) for a in actors]
    for value, (ai, dof) in zip(command, routes, strict=True):
        forces[ai][dof] = value
    for actor, force in zip(actors, forces, strict=True):
        actor.set_external_forces_on_dofs(np.arange(len(force), dtype=np.int32), force)


def controller_state(actor):
    import superdex.physics as p

    if not actor.has_articulated_pose_controller():
        return None
    params = p.PoseControllerParams()
    for key in ("joint_tracking", "link_pos_tracking", "link_rot_tracking"):
        getattr(params, key).resize(len(actor.get_nested_link_actors()))
    actor.get_articulated_pose_controller_params(params)
    target = np.empty_like(native_state(actor)[0])
    actor.get_articulated_target_pose(target)
    return {
        "target": target.tolist(),
        **{
            key: [[item.stiffness, item.damping] for item in getattr(params, key)]
            for key in ("joint_tracking", "link_pos_tracking", "link_rot_tracking")
        },
    }


class NativeComparison:
    """White-box qualification audit; never used to construct the SDK reference."""

    def __init__(self, backend, world, actors, rigids=(), *, env=0, contacts=False):
        import superdex.physics as p

        self.backend, self.env = backend, env
        self.actual = native_actors(backend, env)
        self.reference = list(actors)
        self.rigids = list(rigids)
        self.body_pairs = list(
            zip(
                [*backend._links[env], *backend._rigids[env]],
                [
                    *[world.get_actor(h) for a in actors for h in a.get_nested_link_actors()],
                    *rigids,
                ],
                strict=True,
            )
        )
        self.actor_pairs = list(zip(self.actual, self.reference, strict=True))
        self.contact_pairs = []
        self.errors = dict(
            native_state_max_dev=0.0,
            body_pose_max_dev=0.0,
            body_velocity_max_dev=0.0,
            contact_force_max_dev=0.0,
        )
        self.contact_peak = 0.0
        self.contact_steps = 0
        if contacts:
            for actual, reference in [*self.body_pairs, *self.actor_pairs]:
                registered = []
                for actor in (actual, reference):
                    try:
                        actor.register_query(p.QueryType.TOTAL_CONTACT_FORCE)
                        registered.append(True)
                    except Exception as exc:
                        if not any(
                            s in str(exc) for s in ("contact sample points", "far SDF evaluation")
                        ):
                            raise
                        registered.append(False)
                if registered[0] != registered[1]:
                    raise AssertionError("contact-query availability differs from SDK")
                if registered[0]:
                    self.contact_pairs.append((actual, reference))

    def seed_reference(self):
        """Check authored state before normalizing float32 frame round-off."""
        for actual, reference in self.actor_pairs:
            aq, av = native_state(actual)
            rq, rv = native_state(reference)
            np.testing.assert_allclose(aq, rq, atol=2e-6, rtol=1e-6)
            np.testing.assert_allclose(av, rv, atol=2e-6, rtol=1e-6)
            reference.set_articulated_pose_from_joints(aq)
            reference.set_articulated_joint_velocities(av)
        for actual, reference in zip(self.backend._rigids[self.env], self.rigids, strict=True):
            np.testing.assert_allclose(self.pose(actual), self.pose(reference), atol=2e-6)
            if not reference.is_static():
                reference.set_root_transform(actual.get_root_transform())
                reference.set_velocity(actual.get_linear_velocity(), actual.get_angular_velocity())

    @staticmethod
    def pose(actor):
        t = actor.get_root_transform()
        return np.r_[np.asarray(t.translation), np.asarray(t.rotation)]

    def sample(self):
        for actual, reference in self.actor_pairs:
            for left, right in zip(native_state(actual), native_state(reference), strict=True):
                if not np.isfinite(left).all() or not np.isfinite(right).all():
                    raise AssertionError("non-finite native state")
                self.errors["native_state_max_dev"] = max(
                    self.errors["native_state_max_dev"], maximum(left - right)
                )
        for actual, reference in self.body_pairs:
            left, right = self.pose(actual), self.pose(reference)
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                raise AssertionError("non-finite body pose")
            self.errors["body_pose_max_dev"] = max(
                self.errors["body_pose_max_dev"], maximum(left - right)
            )
            if not reference.is_static():
                for method in ("get_linear_velocity", "get_angular_velocity"):
                    gap = np.asarray(getattr(actual, method)()) - np.asarray(
                        getattr(reference, method)()
                    )
                    if not np.isfinite(gap).all():
                        raise AssertionError("non-finite body velocity")
                    self.errors["body_velocity_max_dev"] = max(
                        self.errors["body_velocity_max_dev"], maximum(gap)
                    )
        for actual, reference in self.contact_pairs:
            left = np.asarray(actual.get_contact_force_world())
            right = np.asarray(reference.get_contact_force_world())
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                raise AssertionError("non-finite contact force")
            self.errors["contact_force_max_dev"] = max(
                self.errors["contact_force_max_dev"], maximum(left - right)
            )
            self.contact_peak = max(self.contact_peak, maximum(right))
            self.contact_steps += int(maximum(right) > 0)

    def passed(self, tolerance=TOL_RIGID):
        return all(
            value <= (TOL_CONTACT_FORCE if name == "contact_force_max_dev" else tolerance)
            for name, value in self.errors.items()
        )


def check_cameras(backend, bot, env=0):
    cameras = {bot.get_sensor(h).get_name(): bot.get_sensor(h) for h in bot.get_sensor_handles()}
    assert set(backend.get_camera_names()) == set(cameras)
    deviation = 0.0
    for name, camera in cameras.items():
        params = camera.get_params()
        for key, actual in backend.get_camera_parameters(name).items():
            expected = getattr(params, key)
            np.testing.assert_equal(
                actual, expected if isinstance(expected, str) else np.asarray(expected)
            )
        t = camera.get_world_transform()
        pose = np.r_[np.asarray(t.translation), np.asarray(t.rotation)[[3, 0, 1, 2]]]
        deviation = max(deviation, maximum(backend.get_camera_poses(name)[env] - pose))
    return {"camera_count": len(cameras), "pose_max_dev": deviation}


def check_scene(ref: ModelRef, run: CheckRun, steps: int = TRAJECTORY_STEPS) -> None:
    """Compare every articulation, body and contact in serial and supported batch modes."""
    from unisim.backend.superdex.scenes import native_settings

    probe = make_backend(ref.path, 1, serial=True, controlled_joints=[])
    serial_only = probe.model.serial_only
    probe.close()
    for mode in ("serial",) if serial_only else ("serial", "batch"):
        references, backend = [], None
        try:
            count = 1 if serial_only else 2
            references = [SceneReference(ref.path) for _ in range(count)]
            names, routes = native_controls(references[0].articulated)
            limits = np.ones(len(names))
            if ref.path.stem == "cart_pole":
                selected, limits = ["Cart"], np.array([3.0])
            elif ref.path.stem == "half_cheetah":
                selected = [
                    "BackThigh",
                    "BackShin",
                    "BackFoot",
                    "FrontThigh",
                    "FrontShin",
                    "FrontFoot",
                ]
                limits = np.array([120, 90, 60, 120, 60, 30.0])
            else:
                selected = names
            routes = [routes[names.index(name)] for name in selected]
            backend = make_backend(
                ref.path,
                count,
                serial=mode == "serial",
                controlled_joints=selected,
                effort_limits=limits,
            )
            assert backend.get_actuator_names() == tuple(selected)
            audits = [
                NativeComparison(backend, r.scene, r.articulated, r.rigids, env=i, contacts=True)
                for i, r in enumerate(references)
            ]
            initial_controllers = [[controller_state(a) for a in r.articulated] for r in references]
            for i, (r, audit) in enumerate(zip(references, audits, strict=True)):
                assert native_settings(backend._worlds[i]) == native_settings(r.scene)
                audit.seed_reference()
                r.scene.step(0)
            suffix = "" if mode == "serial" else "_batch"
            run.record(
                "structure" + suffix,
                True,
                {
                    "execution": mode,
                    "num_envs": count,
                    "articulations": len(references[0].articulated),
                    "controls": selected,
                },
            )

            def advance(commands):
                backend.step(commands)
                for i, r in enumerate(references):
                    apply_native_control(
                        r.articulated, routes, np.clip(commands[i], -limits, limits)
                    )
                    r.step()
                    audits[i].sample()

            def reset():
                backend.reset()
                for r, audit in zip(references, audits, strict=True):
                    r.restore()
                    audit.seed_reference()
                    r.scene.step(0)

            for column in range(len(selected)):
                reset()
                commands = np.zeros((count, len(selected)))
                commands[:, column] = limits[column] * 2
                advance(commands)
            reset()
            for tick in range(steps):
                phase = tick * 0.013 + np.arange(len(selected))
                commands = np.asarray([0.005 * np.sin(phase + i) for i in range(count)])
                advance(commands)
            errors = {key: max(a.errors[key] for a in audits) for key in audits[0].errors}
            run.record(
                "trajectory_equivalence" + suffix,
                all(a.passed() for a in audits),
                {
                    "physics_steps": steps,
                    **errors,
                    "reference_contact_force_steps": sum(a.contact_steps for a in audits),
                    "contact_force_peak": max(a.contact_peak for a in audits),
                },
            )
            before = backend.get_state()
            backend.set_state(np.arange(count), **before)
            roundtrip = max(maximum(backend.get_state()[k] - v) for k, v in before.items())
            backend.reset(np.array([0]))
            for key, value in before.items():
                np.testing.assert_allclose(backend.get_state()[key][1:], value[1:], atol=TOL_SCENE)
            reset()
            reset_error = max(
                maximum(backend.get_state()["qpos"] - backend.get_default_qpos()),
                maximum(backend.get_state()["qvel"] - backend.get_init_qvel()),
            )
            for i, actors in enumerate(native_actors(backend, j) for j in range(count)):
                assert [controller_state(a) for a in actors] == initial_controllers[i]
            advance(np.zeros((count, len(selected))))
            run.record(
                "reset" + suffix,
                roundtrip <= 4e-5 and reset_error <= TOL_SCENE,
                {"state_roundtrip_dev": roundtrip, "reset_error": reset_error},
            )
            run.record(
                "controller_preservation" + suffix, True, {"controllers": initial_controllers[0]}
            )
        finally:
            for reference in references:
                reference.close()
            if backend is not None:
                backend.close()
    for _ in range(2):
        reopened = make_backend(ref.path, 1, serial=True, controlled_joints=[])
        try:
            reopened.step(np.zeros((1, 0)), 3)
        finally:
            reopened.close()
    run.record("cleanup_recreation", True, {"cycles": 2})


PREFAB_DT = 0.001
PREFAB_LIMITS = np.array([87.0] * 4 + [12.0] * 3)
PREFAB_BOT = "bots/arms/fr3_v2/fr3_v2.superdex_bot"
PREFAB_SPHERE = "prefabs/sphere/sphere.mochi_prefab"
PREFAB_BOARD = "prefabs/nine_hole_peg_test/nine_hole_peg_test.mochi_prefab"


def prefab_contact_fixture(root: Path, directory: Path):
    """Generate wrapper files only; preserve all original asset bytes."""
    from unisim.scene import SceneCfg

    child = directory / "objects.mochi_prefab"
    child.write_text(
        json.dumps(
            {
                "prefabs": [
                    {
                        "name": "probe",
                        "path": str(root / PREFAB_SPHERE),
                        "translation": [-0.09, 0, 0.23],
                    },
                    {
                        "name": "board",
                        "path": str(root / PREFAB_BOARD),
                        "translation": [0.55, 0, 0],
                    },
                ]
            }
        )
    )
    assembly = directory / "assembly.mochi_prefab"
    assembly.write_text(
        json.dumps(
            {
                "prefabs": [
                    {
                        "name": "assembly",
                        "path": "./objects.mochi_prefab",
                        "translation": [0, 0, 2.532],
                        "rotation": [0, 0, np.sin(0.2), np.cos(0.2)],
                    }
                ]
            }
        )
    )
    return SceneCfg(str(root / PREFAB_BOT), fragment_files=[str(assembly)])


def prefab_contact_backend(scene, mode: str, count: int = 2):
    from unisim import create_backend

    return create_backend(
        "superdex",
        scene,
        count,
        PREFAB_DT,
        superdex_execution_mode=mode,
        superdex_effort_limits=PREFAB_LIMITS,
    )


def prefab_contact_control(q, v, target):
    return np.clip(80 * (target - q[..., :7]) - 8 * v[..., :7], -PREFAB_LIMITS, PREFAB_LIMITS)


def check_prefab_contact(root: Path, directory: Path, steps: int = 1000) -> dict:
    import superdex.physics as p
    import superdex.robotics as r

    scene_cfg = prefab_contact_fixture(root, directory)
    result = {}
    for mode in ("serial", "batch"):
        b = prefab_contact_backend(scene_cfg, mode)
        direct = None
        bot = None
        try:
            direct = p.create_scene("stage5_direct_reference")
            direct.set_gravity([0, 0, -9.81])
            owner = r.create_context()
            cfg = r.load_bot_prefab_from_file(scene_cfg.model_file)
            bot = r.create_bot(direct, cfg, owner)
            robot = bot.get_articulated_actor()
            loaded = p.prefab.load_from_file(scene_cfg.fragment_files[0], str(root))
            objects = list(p.prefab.add_to_scene(loaded, direct).actors)
            assert [a.get_name() for a in objects] == [item.name for item in b.model.rigids]
            assert len(objects) == 11  # One sphere, one static board, nine dynamic pegs.
            assert sum(a.is_static() for a in objects) == 1
            assert direct.get_num_actors() == b._worlds[0].get_num_actors()
            inventory = []
            for actor in objects:
                pose = actor.get_root_transform()
                inventory.append(
                    {
                        "name": actor.get_name(),
                        "static": actor.is_static(),
                        "collision_shape_hash": str(actor.get_reference_shape().get_hash()),
                        "initial_world_xyz": np.asarray(pose.translation).tolist(),
                        "initial_world_wxyz": np.asarray(pose.rotation)[[3, 0, 1, 2]].tolist(),
                        "mass": None if actor.is_static() else float(actor.get_mass()),
                    }
                )
            robot_handles = set(robot.get_nested_link_actors())
            probe = next(a for a in objects if a.get_name().endswith("/Sphere"))
            probe_index = next(i for i, a in enumerate(objects) if a == probe)
            adapter_handles = set(b._actors[0].get_nested_link_actors())
            probe.register_query(p.QueryType.CONTACT_POINTS)
            b._rigids[0][probe_index].register_query(p.QueryType.CONTACT_POINTS)
            initial = b.get_state()
            target = initial["qpos"][0, :7].copy()
            dtype = initial["qpos"].dtype
            q, v = np.empty(7, dtype=dtype), np.empty(7, dtype=dtype)
            peak_error = 0.0
            contact_steps = 0
            for _ in range(steps):
                state = b.get_state()
                commands = prefab_contact_control(state["qpos"], state["qvel"], target)
                robot.get_articulated_pose(q)
                robot.get_articulated_joint_velocities(v)
                robot.set_external_forces_on_dofs(
                    np.arange(7, dtype=np.int32), prefab_contact_control(q, v, target).astype(dtype)
                )
                b.step(commands)
                direct.step(PREFAB_DT)
                assert direct.get_solver_stats().convergence_status != p.ConvergenceStatus.DIVERGED
                robot.get_articulated_pose(q)
                robot.get_articulated_joint_velocities(v)
                current = b.get_state()
                deviations = [
                    np.max(abs(current["qpos"][0, :7] - q)),
                    np.max(abs(current["qvel"][0, :7] - v)),
                ]
                for item, actor in zip(b.model.rigids, objects, strict=True):
                    body = np.array([item.body_id])
                    pose = actor.get_root_transform()
                    deviations.extend(
                        [
                            np.max(abs(b.get_body_pos_w(body)[0, 0] - pose.translation)),
                            np.max(
                                abs(
                                    b.get_body_quat_w(body)[0, 0]
                                    - np.asarray(pose.rotation)[[3, 0, 1, 2]]
                                )
                            ),
                            np.max(
                                abs(b.get_body_ang_vel_w(body)[0, 0] - actor.get_angular_velocity())
                            ),
                        ]
                    )
                peak_error = max(peak_error, float(max(deviations)))
                points = probe.get_contact_points_world()
                found = any(
                    pt.distance <= 1e-4
                    and (pt.actor_a in robot_handles or pt.actor_b in robot_handles)
                    for pt in points
                )
                adapter_found = any(
                    pt.distance <= 1e-4
                    and (pt.actor_a in adapter_handles or pt.actor_b in adapter_handles)
                    for pt in b._rigids[0][probe_index].get_contact_points_world()
                )
                assert found == adapter_found
                if found:
                    assert adapter_found
                    contact_steps += 1
            assert peak_error < 3e-5, peak_error
            assert contact_steps > 0, "sphere never contacted the robot"
            moved = b.get_state()
            assert np.max(abs(moved["qpos"][:, 7:] - initial["qpos"][:, 7:])) > 0.01
            b.reset(np.array([0]))
            for key in ("qpos", "qvel"):
                np.testing.assert_allclose(b.get_state()[key][0], initial[key][0], atol=2e-6)
                np.testing.assert_array_equal(b.get_state()[key][1], moved[key][1])
            # Give every dynamic object a nontrivial pose and twist, including COM offsets.
            roundtrip = {key: value[[0]].copy() for key, value in moved.items()}
            for item in b.model.rigids:
                if item.qpos_index is None:
                    continue
                qi, vi = item.qpos_index, item.qvel_index
                roundtrip["qpos"][0, qi : qi + 3] += [0.2, -0.1, 0.3]
                roundtrip["qpos"][0, qi + 3 : qi + 7] = [np.cos(0.2), np.sin(0.2), 0, 0]
                roundtrip["qvel"][0, vi : vi + 6] = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]
            b.set_state(np.array([0]), roundtrip["qpos"], roundtrip["qvel"])
            for key in ("qpos", "qvel"):
                np.testing.assert_allclose(b.get_state()[key][0], roundtrip[key][0], atol=3e-6)
                np.testing.assert_array_equal(b.get_state()[key][1], moved[key][1])
            # A selected scene can advance independently; refresh checks no cross-scene ownership.
            unchanged = b.get_state()
            b._worlds[0].step(PREFAB_DT)
            b._refresh(np.array([0]))
            for key in unchanged:
                np.testing.assert_array_equal(b.get_state()[key][1], unchanged[key][1])
            b.reset()
            for key in initial:
                np.testing.assert_allclose(b.get_state()[key], initial[key], atol=2e-6)
            result[mode] = {
                "steps": steps,
                "max_adapter_sdk_error": peak_error,
                "robot_object_contact_steps": contact_steps,
                "scene_actor_count": direct.get_num_actors(),
                "rigid_bodies": [item.name for item in b.model.rigids],
                "rigid_inventory": inventory,
                "nq": b.model.nq,
                "nv": b.model.nv,
                "whole_scene_reset": "passed",
                "selective_reset": "passed",
                "state_roundtrip": "passed",
                "environment_isolation": "passed",
            }
        finally:
            if bot is not None:
                r.destroy_bot(direct, bot)
            if direct is not None:
                p.destroy_scene(direct)
            b.close()
            b.close()
        reopened = prefab_contact_backend(scene_cfg, mode)
        reopened.close()
        result[mode]["cleanup_recreation"] = "passed"
    return result


def check_composition(
    bot_path: Path, fragments: list[Path], run: CheckRun, steps: int = TRAJECTORY_STEPS
) -> None:
    """Compare complete robot/rigid state in both qualified execution modes."""
    from unisim import create_backend
    from unisim.scene import SceneCfg

    ref = ModelRef(bot_path, _kind_of(bot_path), bot_path.name)
    modes = ["serial", "batch"]
    for mode in modes:
        reference = BotReference(bot_path)
        backend = None
        try:
            efforts = np.asarray(resolve_efforts(ref, reference))
            objects = []
            for fragment in fragments:
                loaded = reference.p.prefab.load_from_file(str(fragment), str(_root_of(fragment)))
                objects.extend(reference.p.prefab.add_to_scene(loaded, reference.scene).actors)
            backend = create_backend(
                "superdex",
                SceneCfg(str(bot_path), fragment_files=[str(f) for f in fragments]),
                1 if mode == "serial" else 2,
                SIM_DT,
                superdex_execution_mode=mode,
                superdex_num_workers=0,
                superdex_effort_limits=efforts,
            )
            if backend.model.serial_only:
                modes[:] = ["serial"]
            expected_names = [a.get_name() for a in objects]
            assert expected_names == [item.name for item in backend.model.rigids]
            assert reference.scene.get_num_actors() == backend._worlds[0].get_num_actors()
            assert tuple(backend.get_actuator_names()) == tuple(reference.axis_columns)
            for left, right in zip(backend._rigids[0], objects, strict=True):
                assert left.is_static() == right.is_static()
                # SDK shape hashes identify handles, not geometry across scene loads.
                meshes = [
                    reference.p.get_shape_surface_mesh(actor.get_reference_shape())
                    for actor in (left, right)
                ]
                np.testing.assert_array_equal(meshes[0].coordinates, meshes[1].coordinates)
                np.testing.assert_array_equal(meshes[0].connectivity, meshes[1].connectivity)
                if not right.is_static():
                    np.testing.assert_allclose(left.get_mass(), right.get_mass(), atol=1e-6)
            audit = NativeComparison(backend, reference.scene, [reference.actor], objects)
            audit.seed_reference()
            initial = backend.get_state()
            n = backend.num_actuators
            for tick in range(steps):
                command = np.minimum(efforts, 0.005) * np.sin(tick * 0.013 + np.arange(n))
                backend.step(np.tile(command, (backend.num_envs, 1)))
                reference.step(command)
                audit.sample()
                if backend.num_envs == 2:
                    for value in backend.get_state().values():
                        np.testing.assert_allclose(value[0], value[1], atol=TOL_SCENE)
            suffix = "" if mode == "serial" else "_batch"
            run.record("structure" + suffix, True, {"rigid_names": expected_names})
            run.record(
                "trajectory_equivalence" + suffix,
                audit.passed(TOL_SCENE),
                {"physics_steps": steps, **audit.errors},
            )
            moved = backend.get_state()
            backend.set_state(np.array([0]), moved["qpos"][[0]], moved["qvel"][[0]])
            for key in moved:
                np.testing.assert_allclose(backend.get_state()[key], moved[key], atol=TOL_SCENE)
            backend.reset(np.array([0]))
            for key in initial:
                np.testing.assert_allclose(
                    backend.get_state()[key][0], initial[key][0], atol=TOL_SCENE
                )
                if backend.num_envs == 2:
                    np.testing.assert_array_equal(backend.get_state()[key][1], moved[key][1])
            backend.reset()
            for key in initial:
                np.testing.assert_allclose(backend.get_state()[key], initial[key], atol=TOL_SCENE)
            run.record("reset" + suffix, True, {"roundtrip": True, "selective": True, "full": True})
        finally:
            reference.close()
            if backend is not None:
                backend.close()
                backend.close()


# --------------------------------------------------------------------- #
# Controller qualification (opt-in)
# --------------------------------------------------------------------- #


def load_controller_config(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    base = {"type_name": str(data["type_name"]), "param_args": "", "init_args": ""}
    for key in ("param_args", "init_args"):
        value = data.get(key, "")
        if value and not value.lstrip().startswith("{"):
            resolved = (path.parent / value).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"controller {key} file not found: {resolved}")
            base[key] = str(resolved)
        else:
            base[key] = str(value)
    return base


def load_absolute_target(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_vector(value, size, label):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite with shape ({size},)")
    return array


def _absolute_transform(p, value, label):
    if not isinstance(value, dict) or set(value) != {"translation", "rotation_xyzw"}:
        raise ValueError(f"{label} requires translation and rotation_xyzw")
    translation = _finite_vector(value["translation"], 3, f"{label} translation")
    rotation = _finite_vector(value["rotation_xyzw"], 4, f"{label} rotation_xyzw")
    if not np.isclose(np.linalg.norm(rotation), 1, atol=1e-5):
        raise ValueError(f"{label} rotation_xyzw must be normalized")
    return p.TransformRT(translation=translation, rotation=rotation)


def absolute_target(reference, spec):
    """Build one SDK target from absolute, controller-specific JSON values."""
    if not isinstance(spec, dict) or spec.get("type_name") != reference.kind:
        raise ValueError("absolute target type_name must match the configured controller")
    kind, r = reference.kind, reference.r
    if kind == "BASIC_JSC_PD":
        if set(spec) != {"type_name", "target_pose"}:
            raise ValueError("JSC absolute target requires only target_pose")
        pose = _finite_vector(spec["target_pose"], reference.nv, "JSC target_pose")
        return r.ControllerBasicJscPdTarget(target_pose=pose.astype(reference.dtype))
    if kind == "BASIC_OSC_PD":
        if set(spec) != {"type_name", "root_from_target_ee"}:
            raise ValueError("OSC absolute target requires only root_from_target_ee")
        pose = _absolute_transform(
            reference.p, spec["root_from_target_ee"], "OSC root_from_target_ee"
        )
        return r.ControllerBasicOscPdTarget(root_from_target_ee=pose)
    if kind == "MOCHI_ARTICULATED_POSE":
        if set(spec) != {"type_name", "world_from_root", "pose_dofs"}:
            raise ValueError(
                "articulated-pose absolute target requires world_from_root and pose_dofs"
            )
        target = r.ControllerMochiArticulatedPoseTarget()
        target.world_from_root = _absolute_transform(
            reference.p, spec["world_from_root"], "pose world_from_root"
        )
        target.pose_dofs = _finite_vector(spec["pose_dofs"], reference.n, "pose pose_dofs").astype(
            reference.dtype
        )
        return target
    raise NotImplementedError(f"unsupported absolute target controller {kind!r}")


def shared_absolute_target(kind, spec, info):
    """Convert the existing CLI JSON format into shared values, without SDK imports."""
    from unisim import ArticulationPoseTarget, CartesianTarget, JointTarget

    if not isinstance(spec, dict) or spec.get("type_name") != kind:
        raise ValueError("absolute target type_name must match the configured controller")

    def pose(value):
        if not isinstance(value, dict) or set(value) != {"translation", "rotation_xyzw"}:
            raise ValueError("pose requires translation and rotation_xyzw")
        xyz = _finite_vector(value["translation"], 3, "translation")
        xyzw = _finite_vector(value["rotation_xyzw"], 4, "rotation")
        if not np.isclose(np.linalg.norm(xyzw), 1, atol=1e-5):
            raise ValueError("rotation must be normalized")
        return np.r_[xyz, xyzw[[3, 0, 1, 2]]]

    if kind == "BASIC_JSC_PD":
        if set(spec) != {"type_name", "target_pose"}:
            raise ValueError("JSC absolute target requires only target_pose")
        size = len(info.articulations[0].qvel_indices)
        values = _finite_vector(spec["target_pose"], size, "target_pose")
        return JointTarget(values[list(info.coordinate_qvel_indices)])
    if kind == "BASIC_OSC_PD":
        if set(spec) != {"type_name", "root_from_target_ee"}:
            raise ValueError("OSC absolute target requires only root_from_target_ee")
        return CartesianTarget(pose(spec["root_from_target_ee"]))
    if kind == "MOCHI_ARTICULATED_POSE":
        if set(spec) != {"type_name", "world_from_root", "pose_dofs"}:
            raise ValueError("pose target requires world_from_root and pose_dofs")
        return ArticulationPoseTarget(
            pose(spec["world_from_root"]),
            joint_positions=_finite_vector(
                spec["pose_dofs"], len(info.coordinate_names), "pose_dofs"
            ),
        )
    raise NotImplementedError(f"unsupported controller {kind!r}")


def check_controller(
    ref: ModelRef,
    config: dict[str, str],
    target: dict[str, Any],
    run: CheckRun,
    steps: int = SWEEP_CTRL_STEPS,
) -> None:
    """Compare shared targets with native controllers at every physics substep."""
    reference, backend = None, None
    try:
        reference = BotReference(ref.path)
        efforts = resolve_efforts(ref, reference)
        backend = make_backend(ref.path, 1, serial=True, effort_limits=efforts)
        audit = NativeComparison(backend, reference.scene, [reference.actor])
        audit.seed_reference()
        reference.configure(config)
        backend.configure_controller(**config)
        native_target = absolute_target(reference, target)
        shared_target = shared_absolute_target(
            config["type_name"], target, backend.get_model_info()
        )
        for _ in range(steps * DECIMATION):
            backend.step_controller([shared_target])
            reference.step_controller(native_target, np.asarray(efforts))
            audit.sample()
        run.record(
            "controller_trajectory",
            audit.passed(TOL_TRAJECTORY),
            {
                "controller": config["type_name"],
                "physics_steps": steps * DECIMATION,
                **audit.errors,
            },
        )
        cameras = check_cameras(backend, reference.bot)
        run.record("cameras", cameras["pose_max_dev"] <= TOL_RESET, cameras)
        before = backend.get_state()
        backend.set_state(np.array([0]), **before)
        roundtrip = max(maximum(backend.get_state()[k] - v) for k, v in before.items())
        backend.reset()
        reset_error = max(
            maximum(backend.get_state()["qpos"] - backend.get_default_qpos()),
            maximum(backend.get_state()["qvel"] - backend.get_init_qvel()),
        )
        run.record(
            "reset",
            roundtrip <= TOL_RESET and reset_error <= TOL_RESET,
            {"roundtrip": roundtrip, "reset": reset_error},
        )
        backend.clear_controller()
        run.record(
            "controller_cleanup",
            not backend._controllers,
            {"controllers_remaining": len(backend._controllers)},
        )
    finally:
        if reference is not None:
            reference.close()
        if backend is not None:
            backend.close()


# --------------------------------------------------------------------- #
# Worker protocol and subprocess isolation
# --------------------------------------------------------------------- #


@dataclass
class WorkerTask:
    label: str
    path: str
    kind: str
    steps: int | None = None
    controller_config: str | None = None
    absolute_target: str | None = None
    fixtures_dir: str | None = None
    prefab_contact_root: str | None = None
    composition: list[str] | None = None


def run_worker(task_path: Path, result_path: Path) -> int:
    """Run exactly one model in this process and write its result record."""
    task = json.loads(task_path.read_text(encoding="utf-8"))
    worker = WorkerTask(**task)
    if worker.steps is None:
        worker.steps = (
            SWEEP_CTRL_STEPS * DECIMATION
            if worker.kind in {"bot", "archive"}
            and not worker.composition
            and not worker.prefab_contact_root
            else TRAJECTORY_STEPS
        )
    ref = ModelRef(Path(worker.path), worker.kind, worker.label)
    run = CheckRun(worker.label, worker.kind)
    status = STATUS_PASSED
    limitation = None
    diagnostic = None
    started = time.perf_counter()
    runtime_held = False
    try:
        limitation = classify_static(ref)
        if limitation is not None:
            status = STATUS_BLOCKED if worker.kind != "urdf" else STATUS_UNSUPPORTED
            print(f"    blocked: {limitation.diagnostic}")
        else:
            # A direct SDK reference scene shares the adapter's process-global
            # runtime; hold one reference here so scene creation succeeds
            # before the first backend exists and stays alive to the end.
            from unisim.backend.superdex.runtime import (
                acquire_runtime,
                load_superdex_dependencies,
                release_runtime,
            )

            physics, _robotics = load_superdex_dependencies()
            acquire_runtime(physics)
            runtime_held = True
            if worker.prefab_contact_root:
                with tempfile.TemporaryDirectory(prefix="unisim-contact-") as directory:
                    checks = check_prefab_contact(
                        Path(worker.prefab_contact_root), Path(directory), steps=worker.steps
                    )
                for mode, details in checks.items():
                    run.record("prefab_contact_" + mode, True, details)
            elif worker.controller_config:
                config = load_controller_config(Path(worker.controller_config))
                target = load_absolute_target(Path(worker.absolute_target))
                check_controller(ref, config, target, run, steps=max(1, worker.steps // DECIMATION))
            elif worker.composition:
                check_composition(
                    Path(worker.path),
                    [Path(p) for p in worker.composition],
                    run,
                    steps=worker.steps,
                )
            elif worker.kind in {"bot", "archive"}:
                check_bot(ref, run, steps=worker.steps)
            elif worker.kind in {"scene", "prefab"}:
                check_scene(ref, run, steps=worker.steps)
            else:
                status = STATUS_UNSUPPORTED
                print(f"    unsupported input format: {ref.suffix}")
        if run.failures:
            status = STATUS_FAILED
    except NotImplementedError as exc:
        status = STATUS_BLOCKED
        if limitation is None:
            limitation = Limitation(
                path=_display_path(ref),
                format=ref.suffix,
                unsupported_feature="adapter profile rejection",
                diagnostic=str(exc)[:300],
                blocker_class=ADAPTER_RESTRICTION,
                checked=CHECKED_RUNTIME,
                enabling_capability="extended audited profile support",
            )
        print(f"    blocked: {exc}")
    except ImportError as exc:
        status = STATUS_UNVERIFIED
        diagnostic = f"optional dependency unavailable: {exc}"
        print(f"    unverified: {diagnostic}")
    except Exception as exc:  # noqa: BLE001 — recorded, never hides later models
        status = STATUS_FAILED
        diagnostic = f"{type(exc).__name__}: {exc}"
        print(f"    failed: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        if runtime_held:
            try:
                from unisim.backend.superdex.runtime import release_runtime

                release_runtime(physics)
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass
    result = {
        "model": _display_path(ref),
        "path": str(ref.path),
        "kind": worker.kind,
        "status": status,
        "checks": run.checks,
        "failed_checks": run.failures,
        "limitation": limitation.record() if limitation else None,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "sim_dt": SIM_DT,
        "diagnostic": diagnostic,
        "parameters": task,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if status == STATUS_PASSED else 1


def run_model_subprocess(worker_task: WorkerTask, timeout: int = SUBPROCESS_TIMEOUT) -> dict:
    """Run one model in an isolated subprocess; native crashes stay isolated."""
    with tempfile.TemporaryDirectory(prefix="unisim-compare-") as directory:
        task_path = Path(directory) / "task.json"
        result_path = Path(directory) / "result.json"
        task_path.write_text(json.dumps(worker_task.__dict__), encoding="utf-8")
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(task_path),
                    "--worker-result",
                    str(result_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(REPOSITORY_ROOT),
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            if result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if completed.returncode not in (0, 1):
                    result["status"] = STATUS_CRASHED
                    result["returncode"] = completed.returncode
                result["worker_output_tail"] = output[-4000:]
                return result
            return {
                "model": worker_task.label,
                "path": worker_task.path,
                "kind": worker_task.kind,
                "status": STATUS_CRASHED,
                "checks": {},
                "failed_checks": [],
                "limitation": None,
                "returncode": completed.returncode,
                "diagnostic": output[-4000:],
            }
        except subprocess.TimeoutExpired as exc:
            tail = ((exc.stdout or b"") + (exc.stderr or b"")).decode(errors="replace")[-2000:]
            return {
                "model": worker_task.label,
                "path": worker_task.path,
                "kind": worker_task.kind,
                "status": STATUS_TIMEOUT,
                "checks": {},
                "failed_checks": [],
                "limitation": None,
                "diagnostic": f"subprocess timeout after {timeout}s",
                "worker_output_tail": tail,
            }


# --------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------- #


def print_results(results: list[dict], elapsed: float) -> None:
    print(f"\n== Results ({len(results)} models, {elapsed:.0f}s)")
    width = max([len(r["model"]) for r in results] + [10])
    for result in results:
        marker = {
            STATUS_PASSED: "pass",
            STATUS_FAILED: "FAIL",
            STATUS_BLOCKED: "blocked",
            STATUS_UNSUPPORTED: "unsupported",
            STATUS_UNVERIFIED: "unverified",
            STATUS_CRASHED: "CRASH",
            STATUS_TIMEOUT: "TIMEOUT",
        }[result["status"]]
        note = ""
        if result["status"] == STATUS_PASSED:
            note = f"{len(result['checks'])} checks"
        elif result.get("failed_checks"):
            note = ", ".join(result["failed_checks"][:4])
        elif result.get("limitation"):
            note = result["limitation"]["unsupported_feature"]
        elif result.get("diagnostic"):
            note = str(result["diagnostic"])[:60]
        print(f"  {result['model']:{width}}  {marker:12}  {note}")
        if result["status"] in {STATUS_FAILED, STATUS_CRASHED, STATUS_TIMEOUT}:
            detail = result.get("worker_output_tail") or result.get("diagnostic", "")
            if detail:
                print("\n".join("      " + line for line in detail.splitlines()[-15:]))
    limitations = [r["limitation"] for r in results if r.get("limitation")]
    if limitations:
        print(f"\n== Current limitations ({len(limitations)} unsupported models)")
        for item in limitations:
            print(f"  {item['model']}")
            print(f"      feature:   {item['unsupported_feature']}")
            print(f"      blocker:   {item['blocker_class']} (checked: {item['checked']})")
            print(f"      needs:     {item['enabling_capability']}")


def write_reports(out: Path, results: list[dict], refs: list[ModelRef]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = code_fingerprint()
    fingerprint["model_sha256"] = {
        str(ref.path): hashlib.sha256(ref.path.read_bytes()).hexdigest()
        for ref in refs
        if ref.path.is_file()
    }
    summary = {
        "tool": "scripts/superdex_compare.py",
        "sim_dt": SIM_DT,
        "results": results,
        "limitation_records": [r["limitation"] for r in results if r.get("limitation")],
        "provenance": fingerprint,
        "counts": {
            status: sum(1 for r in results if r["status"] == status)
            for status in {r["status"] for r in results}
        },
    }
    (out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"reports: {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    models = parser.add_mutually_exclusive_group()
    models.add_argument(
        "model_paths",
        nargs="*",
        type=Path,
        help="individual native model paths (bots, archives, scenes, prefabs)",
    )
    models.add_argument(
        "--all",
        action="store_true",
        help="discover and compare every native model under the local roots",
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=None,
        help="roots for --all (default: assets/superdex[ assets/superdex-physics])",
    )
    parser.add_argument(
        "--match", type=str, default=None, help="substring filter on discovered model paths"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="trajectory steps (default: bots/controllers 1040, scenes/compositions 1000)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=SUBPROCESS_TIMEOUT,
        help="per-model subprocess timeout in seconds",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write JSON results into this (ignored) directory; e.g. results/superdex",
    )
    parser.add_argument(
        "--controller-config",
        type=Path,
        default=None,
        help="SDK controller JSON (type_name/param_args/init_args)",
    )
    parser.add_argument(
        "--absolute-target",
        type=Path,
        default=None,
        help="absolute controller target JSON for --controller-config",
    )
    parser.add_argument(
        "--composition",
        nargs="*",
        type=Path,
        default=None,
        help="rigid prefab fragments to compose with a single bot model",
    )
    parser.add_argument(
        "--fixtures",
        action="store_true",
        help="also compare the synthetic fixture set (temporary directory)",
    )
    parser.add_argument("--worker", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker is not None:
        return run_worker(args.worker, args.worker_result or args.worker.with_name("result.json"))

    if args.steps is not None and args.steps < DECIMATION:
        parser.error("--steps must be at least 8")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if bool(args.controller_config) != bool(args.absolute_target):
        parser.error("--controller-config and --absolute-target must be supplied together")
    if args.controller_config or args.composition:
        if len(args.model_paths) != 1 or args.fixtures or args.all:
            parser.error("controller/composition options require exactly one bot or archive")
    if args.controller_config and args.composition:
        parser.error("controller comparison with fragments is not qualified; use the viewer")

    started = time.perf_counter()
    refs: list[ModelRef] = []
    if args.all:
        roots = [r.resolve() for r in (args.roots or default_roots())]
        refs.extend(discover_models(roots))
    if args.model_paths:
        for path in args.model_paths:
            try:
                resolved = resolve_model_path(path)
            except FileNotFoundError as exc:
                parser.error(str(exc))
            kind = _kind_of(resolved)
            if kind in {"unknown", "urdf"}:
                print(f"error: unsupported direct-model format: {resolved}", file=sys.stderr)
                print(
                    "  supported: .superdex_bot, .superdex_bot_archive, .mochi_scene, "
                    ".mochi_prefab, audited .xml MJCF; .urdf is not loadable "
                    "(SDK URDF loader drops primitive geometry)",
                    file=sys.stderr,
                )
                return 2
            refs.append(ModelRef(resolved, kind, resolved.name))
    if not refs and not args.fixtures:
        parser.error("pass model paths, --all, or --fixtures")

    tasks = []
    for ref in refs:
        if (args.controller_config or args.composition) and ref.kind not in {"bot", "archive"}:
            parser.error("controller/composition options require a bot or archive")
        if args.match and args.match not in str(ref.path):
            continue
        task = WorkerTask(
            label=ref.label,
            path=str(ref.path),
            kind=ref.kind,
            steps=args.steps,
            controller_config=str(args.controller_config.resolve())
            if args.controller_config
            else None,
            absolute_target=str(args.absolute_target.resolve()) if args.absolute_target else None,
            composition=[str(resolve_model_path(p)) for p in args.composition]
            if args.composition
            else None,
        )
        if args.controller_config:
            task.kind = "bot"
        if args.composition:
            task.kind = "bot"
        tasks.append(task)

    fixture_owner = None
    if args.fixtures:
        fixture_owner = tempfile.TemporaryDirectory(prefix="unisim-compare-fixtures-")
        fixtures_dir = Path(fixture_owner.name)
        generated = synthetic_fixtures(fixtures_dir)
        for root in default_roots():
            if (root / PREFAB_BOT).is_file() and not args.match:
                tasks.append(
                    WorkerTask(
                        label="synthetic:prefab_contact",
                        path=str(root / PREFAB_BOT),
                        kind="bot",
                        steps=args.steps,
                        prefab_contact_root=str(root),
                    )
                )
        for name, path in sorted(generated.items()):
            if args.match and args.match not in name:
                continue
            tasks.append(
                WorkerTask(
                    label=f"synthetic:{name}", path=str(path), kind=_kind_of(path), steps=args.steps
                )
            )

    results = []
    for task in tasks:
        print(f"== {task.label}", flush=True)
        results.append(run_model_subprocess(task, timeout=args.timeout))

    elapsed = time.perf_counter() - started
    print_results(results, elapsed)
    if args.out is not None:
        write_reports(args.out, results, refs)

    if fixture_owner is not None:
        fixture_owner.cleanup()
    if not results:
        parser.error("no models matched the selection")
    unexpected = sum(
        result["status"] != STATUS_PASSED
        and not (
            result["status"] in {STATUS_BLOCKED, STATUS_UNSUPPORTED}
            and result.get("limitation")
            and result["limitation"]["checked"] == CHECKED_STATIC
        )
        for result in results
    )
    if unexpected:
        print(f"error: {unexpected} model(s) failed, crashed or timed out", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
