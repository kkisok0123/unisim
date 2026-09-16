#!/usr/bin/env python3
"""Numerical and lifecycle qualification of stage-3 candidate bots (stage 3).

One parameterized command runs the stage-2B FR3 checks for every registered
candidate (native fixed-base bots and recipe compositions) against the local
asset copy. For each bot, a direct SuperDex SDK scene in the same process is
driven with identical inputs, so adapter translation is separated from
asset/SDK behavior:

* structure — body/joint inventories, default pose, dynamic-link masses, world
  link transforms, world AABBs and authored joint ranges against the compiled
  SDK actor and authored prefab;
* recipe accounting — for recipe bots, every base/attachment dependency
  resolves inside the verified bundle and the SDK-compiled inventories account
  for each referenced bot;
* control — per-joint ordering (a single-joint torque dominates only that
  joint) and effort-limit clipping;
* trajectories — batch and serial execution match the direct SDK rollout under
  the recorded PD sweep profile (also the >=1,000-step stability evidence);
* contact — recovery from a pose beyond an authored joint range, where the
  bot's geometry admits a representative self-contact;
* reset — exact whole-scene/selective restore plus state round trips;
* isolation — each environment of a two-environment backend reproduces the
  matching single-environment backend exactly;
* lifecycle — repeated create/step/reset/close cycles.

Bots with known later-capability blockers are probed for precise evidence and
recorded as blocked with their exact defect; they are not qualified. The run
writes one JSON report per bot plus a compatibility table under
``docs/superdex-bots-qualification/`` and exits nonzero when a registered
candidate fails a check. Requires the optional SuperDex runtime; no viewer or
graphical session is needed:

    export SUPERDEX_ASSETS_PATH="$PWD/assets/superdex"
    uv run scripts/superdex_bot_qualify.py                     # all candidates
    uv run scripts/superdex_bot_qualify.py --bots openarm_v20,googly_eyes
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from superdex_bot_profiles import (  # noqa: E402
    BLOCKED_KEYS,
    PROFILES,
    BotProfile,
    ResolvedProfile,
    resolve_gains,
    sweep_amplitudes,
)

from unisim.backend.superdex.assets import resolve_reference, verify_asset_bundle  # noqa: E402

INVENTORY_REPORT = REPOSITORY_ROOT / "docs" / "superdex-assets-inventory.json"

SIM_DT = 0.002
DECIMATION = 8  # 160 Hz control on 500 Hz physics

# Length of the bounded PD sweep in control steps; 130 * 8 = 1040 physics
# steps, satisfying the >=1000-step stability requirement.
SWEEP_CTRL_STEPS = 130
REACHABLE_FRACTION = 0.6
# How far the penetrating contact pose sits beyond the authored lower limit.
CONTACT_PENETRATION = 0.8
CONTACT_STEPS = 400

# Tolerances: adapter and direct SDK integrate the same native code path with
# identical inputs, so comparisons are exact up to float noise; the small
# nonzero bounds make translation regressions fail loudly.
TOL_STRUCTURE = 1e-6
TOL_TRAJECTORY = 1e-7
TOL_RESET = 1e-6


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


def _load_prefab(bot_path: Path):
    """Load one bot prefab through the SDK, silencing its stdout chatter.

    Prefab parsing does not require the process runtime; the SDK prints
    verbose joint dumps to stdout on recipe compilation, which is noise here.
    """
    import superdex.robotics as robotics

    with contextlib.redirect_stdout(io.StringIO()):
        return robotics.load_bot_prefab_from_file(str(bot_path))


def authored_joint_metadata(cfg) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Extract (names, projected ranges, efforts, armature) from a prefab."""
    import superdex.physics as physics

    jt = physics.ArticulatedJointType
    names, ranges, efforts, armature = [], [], [], []
    for joint in cfg.joints:
        if joint.type in (jt.HARD, jt.FREE):
            continue
        names.append(str(joint.name))
        if joint.type == jt.REVOLUTE and joint.min_limit is not None:
            axis = np.asarray(joint.axis, dtype=float)
            axis = axis / np.linalg.norm(axis)
            ranges.append(
                [float(np.dot(joint.min_limit, axis)), float(np.dot(joint.max_limit, axis))]
            )
        else:
            ranges.append([-math.inf, math.inf])
        efforts.append(float(joint.effort_limit))
        armature.append(float(joint.inertia or 0))
    return tuple(names), np.asarray(ranges), np.asarray(efforts), np.asarray(armature)


def _recipe_references(bot_path: Path) -> list[tuple[str, str]]:
    """Return (kind, reference) for every base/attachment path of a recipe."""
    data = json.loads(bot_path.read_text(encoding="utf-8"))
    references: list[tuple[str, str]] = []
    if "base" in data:
        references.append(("base", str(data["base"])))
    for modification in data.get("modifications") or []:
        if not isinstance(modification, dict):
            continue
        for kind in ("AttachBot", "ReplaceLinkWithBot"):
            if kind in modification and isinstance(modification[kind], dict):
                references.append((kind, str(modification[kind].get("path", ""))))
    return references


class Reference:
    """A direct SuperDex SDK scene of one bot, driven with identical inputs.

    The native runtime is process-global and reference-counted by the adapter,
    so the reference scene lives in the same process as the backends under
    test: a backend must exist (holding the runtime) before construction, and
    this scene must be destroyed before the last backend closes so teardown
    never runs after the adapter has shut the shared runtime down.
    """

    def __init__(self, bot_path: Path, dt: float) -> None:
        import superdex.physics as physics
        import superdex.robotics as robotics

        self.p = physics
        self.cfg = _load_prefab(bot_path)
        self.scene = physics.create_scene("bot_qualification_reference")
        self.scene.set_gravity([0, 0, -9.81])
        self.context = robotics.create_context()
        self.bot = robotics.create_bot(self.scene, self.cfg, self.context)
        self.actor = self.bot.get_articulated_actor()
        self.links = [self.scene.get_actor(h) for h in actor_links(self.bot)]
        self.names, self.ranges, self.efforts, self.armature = authored_joint_metadata(self.cfg)
        self.n = len(self.names)
        self.dofs = np.arange(self.n, dtype=np.int32)
        self.dt = dt

    def qvel(self) -> tuple[np.ndarray, np.ndarray]:
        q, v = np.empty(self.n, dtype=np.float32), np.empty(self.n, dtype=np.float32)
        self.actor.get_articulated_pose(q)
        self.actor.get_articulated_joint_velocities(v)
        return q, v

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Mirror SuperDexBackend.set_state's native sequence exactly."""
        self.actor.set_articulated_pose_from_joints(np.asarray(qpos, dtype=np.float32))
        self.actor.set_articulated_joint_velocities(np.asarray(qvel, dtype=np.float32))
        self.actor.set_external_forces_on_dofs(self.dofs, np.zeros(self.n, dtype=np.float32))
        self.scene.step(0)

    def step(self, torque: np.ndarray, nsteps: int = 1) -> None:
        """Re-apply the torque before every substep, matching adapter semantics."""
        for _ in range(nsteps):
            self.actor.set_external_forces_on_dofs(
                self.dofs, np.asarray(torque, dtype=np.float32)
            )
            self.scene.step(self.dt)

    def aabb(self, link_index: int) -> np.ndarray:
        box = self.links[link_index].get_aabb_world()
        return np.asarray([box.min, box.max], dtype=np.float64)

    def link_pos(self, link_index: int) -> np.ndarray:
        return np.asarray(
            self.links[link_index].get_root_transform().translation, dtype=np.float64
        )

    def register_contact_queries(self) -> list:
        """Register point queries on the first non-static links after the root."""
        probes = []
        for link in self.links[2 : min(9, len(self.links))]:
            if link.is_static():
                continue
            link.register_query(self.p.QueryType.CONTACT_POINTS)
            probes.append(link)
        return probes

    def contact_point_count(self, probes: list) -> int:
        return sum(len(link.get_contact_points_world()) for link in probes)

    def close(self) -> None:
        import superdex.robotics as robotics

        robotics.destroy_bot(self.scene, self.bot)
        self.p.destroy_scene(self.scene)
        self.bot = None
        self.scene = None


def actor_links(bot) -> list:
    actor = bot.get_articulated_actor()
    return list(actor.get_nested_link_actors())


class QualificationRun:
    """The stage-2B check set, parameterized over one bot and its profile."""

    def __init__(self, profile: BotProfile, bot_path: Path) -> None:
        from unisim import create_backend
        from unisim.scene import SceneCfg

        self.create_backend = create_backend
        self.SceneCfg = SceneCfg
        self.profile = profile
        self.bot_path = bot_path
        self.report: dict = {
            "bot": profile.relpath,
            "key": profile.key,
            "sim_dt": SIM_DT,
            "decimation": DECIMATION,
            "actuator_order": [],
            "effort_limits_nm": None,
            "pd_gains": None,
            "sweep_amplitudes_rad": None,
            "tolerances": {
                "structure": TOL_STRUCTURE,
                "trajectory": TOL_TRAJECTORY,
                "reset": TOL_RESET,
            },
            "checks": {},
        }
        self.failures: list[str] = []

    def make_backend(self, resolved: ResolvedProfile, num_envs: int, serial: bool = False):
        kwargs: dict = {"superdex_effort_limits": list(resolved.effort_limits)}
        if serial:
            kwargs.update(superdex_execution_mode="serial", superdex_num_workers=0)
        return self.create_backend(
            "superdex", self.SceneCfg(str(self.bot_path)), num_envs, SIM_DT, **kwargs
        )

    def record(self, name: str, passed: bool, evidence: dict) -> None:
        self.report["checks"][name] = {"passed": bool(passed), **evidence}
        print(
            f"    [{'PASS' if passed else 'FAIL'}] {name}"
            + ("" if passed else f" — {evidence}")
        )
        if not passed:
            self.failures.append(name)

    # ------------------------------------------------------------------ #
    # Controls                                                           #
    # ------------------------------------------------------------------ #

    def sweep_target(self, q0: np.ndarray, amplitudes: np.ndarray, ctrl_index: int) -> np.ndarray:
        t = ctrl_index * SIM_DT * DECIMATION
        phase = np.linspace(0.0, 3.0 * math.pi / 2.0, len(q0))
        period = self.profile.sweep_period
        return q0 + amplitudes * np.sin(2.0 * math.pi * t / period + phase)

    @staticmethod
    def pd_torque(
        resolved: ResolvedProfile, target: np.ndarray, q: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        kp, kd = np.asarray(resolved.kp), np.asarray(resolved.kd)
        effort = np.asarray(resolved.effort_limits)
        return np.clip(kp * (target - q) - kd * v, -effort, effort)

    # ------------------------------------------------------------------ #
    # Checks                                                             #
    # ------------------------------------------------------------------ #

    def check_recipe(self, assets_root: Path) -> None:
        """Every recipe dependency resolves inside the verified bundle."""
        references = _recipe_references(self.bot_path)
        if not references:
            self.record("recipe_accounting", True, {"recipe": False})
            return
        edges = []
        all_resolved = True
        for kind, reference in references:
            resolved_path, rule = resolve_reference(self.bot_path.parent, reference, assets_root)
            ok = resolved_path.is_file()
            all_resolved = all_resolved and ok
            edges.append(
                {
                    "kind": kind,
                    "reference": reference,
                    "resolution_rule": rule,
                    "resolved": ok,
                    "target": (
                        resolved_path.relative_to(assets_root).as_posix()
                        if ok
                        else str(resolved_path)
                    ),
                }
            )
        self.record("recipe_accounting", all_resolved, {"recipe": True, "edges": edges})

    def check_structure(self, backend, reference: Reference) -> None:
        model = backend.model
        n = reference.n
        authored_links = tuple(str(link.name) for link in reference.cfg.links)
        bodies_ok = tuple(model.body_names)[1:] == authored_links
        joints = reference.names
        joints_ok = tuple(backend.get_actuator_names()) == joints
        ctrl_order_ok = tuple(backend.get_actuator_joint_names()) == joints
        joint_dofs_ok = tuple(model.joint_names) == joints

        q0 = backend.get_default_qpos()
        ref_q0, _ = reference.qvel()
        q0_dev = float(np.max(np.abs(q0 - ref_q0))) if n else 0.0

        # Authored mass metadata vs compiled SDK link masses. Adapter bodies
        # are [world, <links>]; SDK links are the same links. Static links
        # report no SDK mass while the adapter keeps the authored value.
        masses = backend.get_body_mass()
        mass_max = 0.0
        for body_id in range(1, len(model.body_names)):
            ref_link = reference.links[body_id - 1]
            if ref_link.is_static():
                continue
            mass_max = max(mass_max, abs(float(masses[body_id]) - float(ref_link.get_mass())))

        body_ids = np.arange(len(model.body_names))
        pos_w = backend.get_body_pos_w(body_ids)[0]
        ref_pos = np.array([reference.link_pos(i) for i in range(len(reference.links))])
        pos_dev = float(np.max(np.abs(pos_w[1:] - ref_pos)))

        aabb_dev = 0.0
        for body_id in range(1, len(model.body_names)):
            adapter_actor = backend._actors[0]
            handle = adapter_actor.get_nested_link_actors()[model.body_link_indices[body_id]]
            adapter_link = backend._worlds[0].get_actor(handle)
            box = adapter_link.get_aabb_world()
            adapter_box = np.asarray([box.min, box.max], dtype=np.float64)
            aabb_dev = max(
                aabb_dev, float(np.max(np.abs(adapter_box - reference.aabb(body_id - 1))))
            )

        ranges_dev = float(np.max(np.abs(backend.get_joint_range() - reference.ranges)))

        self.record(
            "structure",
            bodies_ok
            and joints_ok
            and ctrl_order_ok
            and joint_dofs_ok
            and q0_dev <= TOL_STRUCTURE
            and mass_max <= TOL_STRUCTURE
            and pos_dev <= TOL_STRUCTURE
            and aabb_dev <= TOL_STRUCTURE
            and ranges_dev <= TOL_STRUCTURE,
            {
                "body_inventory_ok": bodies_ok,
                "joint_inventory_ok": joints_ok and joint_dofs_ok,
                "control_order_ok": ctrl_order_ok,
                "default_qpos_max_dev_rad": q0_dev,
                "dynamic_body_mass_max_dev_kg": mass_max,
                "body_pos_max_dev_m": pos_dev,
                "aabb_max_dev_m": aabb_dev,
                "joint_range_max_dev_rad": ranges_dev,
            },
        )

    def check_control(self, resolved: ResolvedProfile) -> None:
        """Control wiring and effort clipping.

        The stage-2B FR3 check asserted the torqued joint is the most affected
        DoF, which holds for FR3's inertia shape but not universally: light
        distal links respond more to proximal torques than to their own. The
        stage-3 generalization compares the adapter's per-joint velocity
        response matrix (torque on joint i, response on every joint, minus a
        zero-torque baseline) against the direct SDK matrix under identical
        probes — exact translation — plus a nonzero diagonal, which proves
        each control column reaches its own joint.
        """
        n = resolved.n
        probe = min(5.0, float(np.min(resolved.effort_limits)) * 0.5)

        def adapter_matrix() -> np.ndarray:
            backend = self.make_backend(resolved, 1)
            try:
                backend.reset()
                backend.step(np.zeros((1, n)), 5)
                base = backend.get_dof_vel()[0].copy()
                matrix = np.zeros((n, n))
                for i in range(n):
                    backend.reset()
                    torque = np.zeros((1, n))
                    torque[0, i] = probe
                    backend.step(torque, 5)
                    matrix[i] = backend.get_dof_vel()[0] - base
                return matrix
            finally:
                backend.close()

        def reference_matrix(holder) -> np.ndarray:
            reference = Reference(self.bot_path, SIM_DT)
            try:
                q0 = holder.get_default_qpos().copy()
                reference.set_state(q0, np.zeros(n))
                reference.step(np.zeros(n), 5)
                _, base = reference.qvel()
                matrix = np.zeros((n, n))
                for i in range(n):
                    reference.set_state(q0, np.zeros(n))
                    torque = np.zeros(n)
                    torque[i] = probe
                    reference.step(torque, 5)
                    _, velocity = reference.qvel()
                    matrix[i] = velocity - base
                return matrix
            finally:
                reference.close()

        holder = self.make_backend(resolved, 1)  # keeps the shared runtime alive
        try:
            adapter = adapter_matrix()
            reference = reference_matrix(holder)
        finally:
            holder.close()
        wiring_ok = bool(np.all(np.diag(adapter) != 0.0))
        matrix_dev = float(np.max(np.abs(adapter - reference)))

        # Effort clipping: a strongly saturated command reproduces per-joint
        # capped dynamics exactly.
        limits = np.asarray(resolved.effort_limits)
        saturated = self.make_backend(resolved, 1)
        capped = self.make_backend(resolved, 1)
        try:
            saturated.step(np.full((1, n), 1e4), 200)
            capped.step(limits[None, :], 200)
            clip_ok = bool(
                np.array_equal(saturated.get_state()["qpos"], capped.get_state()["qpos"])
            )
            clip_dev = float(
                np.max(np.abs(saturated.get_state()["qpos"] - capped.get_state()["qpos"]))
            )
        finally:
            saturated.close()
            capped.close()

        self.record(
            "control",
            wiring_ok and matrix_dev <= TOL_TRAJECTORY and clip_ok,
            {
                "probe_torque_nm": probe,
                "control_columns_reach_own_joint": wiring_ok,
                "response_matrix_vs_sdk_max_dev_rad_s": matrix_dev,
                "response_matrix_diagonal": [
                    round(float(v), 6) for v in np.diag(adapter)
                ],
                "effort_clip_exact": clip_ok,
                "effort_clip_max_dev_rad": clip_dev,
            },
        )

    def check_trajectory_equivalence(
        self, backend, serial_backend, reference: Reference, resolved: ResolvedProfile
    ) -> None:
        n = reference.n
        q0 = backend.get_default_qpos().copy()
        amplitudes = np.asarray(
            sweep_amplitudes(self.profile, q0, backend.get_joint_range(), REACHABLE_FRACTION)
        )
        backend.reset()
        serial_backend.reset()
        reference.set_state(q0, np.zeros(n))

        batch_dev = 0.0
        serial_dev = 0.0
        max_qvel = 0.0
        for k in range(SWEEP_CTRL_STEPS):
            target = self.sweep_target(q0, amplitudes, k)
            tau_b = self.pd_torque(
                resolved, target, backend.get_dof_pos()[0], backend.get_dof_vel()[0]
            )
            tau_s = self.pd_torque(
                resolved,
                target,
                serial_backend.get_dof_pos()[0],
                serial_backend.get_dof_vel()[0],
            )
            q_ref, v_ref = reference.qvel()
            tau_r = self.pd_torque(resolved, target, q_ref, v_ref)
            backend.step(np.repeat(tau_b[None, :], backend.num_envs, axis=0), DECIMATION)
            serial_backend.step(tau_s[None, :], DECIMATION)
            reference.step(tau_r, DECIMATION)
            q_ref, v_ref = reference.qvel()
            batch_dev = max(batch_dev, float(np.max(np.abs(backend.get_dof_pos()[0] - q_ref))))
            serial_dev = max(
                serial_dev, float(np.max(np.abs(serial_backend.get_dof_pos()[0] - q_ref)))
            )
            max_qvel = max(max_qvel, float(np.max(np.abs(v_ref))))

        state = backend.get_state()
        finite = bool(np.isfinite(state["qpos"]).all() and np.isfinite(state["qvel"]).all())
        steps = SWEEP_CTRL_STEPS * DECIMATION
        bound = self.profile.max_qvel_bound
        qvel_ok = bound is None or max_qvel < bound

        self.record(
            "trajectory_equivalence",
            batch_dev <= TOL_TRAJECTORY
            and serial_dev <= TOL_TRAJECTORY
            and finite
            and qvel_ok
            and steps >= 1000,
            {
                "physics_steps": steps,
                "sweep_amplitudes_rad": [round(float(a), 4) for a in amplitudes],
                "batch_vs_sdk_max_dev_rad": batch_dev,
                "serial_vs_sdk_max_dev_rad": serial_dev,
                "reference_max_abs_qvel_rad_s": max_qvel,
                "max_abs_qvel_bound_rad_s": bound,
                "finite_state": finite,
            },
        )

    def check_contact_recovery(
        self, backend, resolved: ResolvedProfile
    ) -> None:
        """Recovery from a pose beyond an authored joint range.

        Uses a fresh reference scene: registering CONTACT_POINTS queries on a
        scene whose earlier rollouts produced deep self-collisions trips a
        native MOCHI assertion in the debug SDK build when it later steps a
        penetrating pose (an SDK behavior recorded in the report, not an
        adapter defect — the adapter's own scenes never assert here). A fresh
        scene whose history is only the penetrating probe is stable.
        """
        if self.profile.contact_joint is None:
            self.record(
                "contact_recovery",
                True,
                {"skipped": "no representative self-contact geometry"},
            )
            return
        n = len(backend.get_actuator_names())
        joint = self.profile.contact_joint
        if joint >= n:
            self.record("contact_recovery", True, {"skipped": f"joint index {joint} out of range"})
            return
        reference = Reference(self.bot_path, SIM_DT)
        try:
            lo = float(reference.ranges[joint][0])
            if not math.isfinite(lo):
                self.record(
                    "contact_recovery",
                    True,
                    {"skipped": f"joint {joint} has no finite lower limit"},
                )
                return
            probes = reference.register_contact_queries()
            q0 = backend.get_default_qpos().copy()
            q_pen = q0.copy()
            q_pen[joint] = lo - CONTACT_PENETRATION

            backend.set_state(np.array([0]), q_pen[None, :], np.zeros((1, n)))
            reference.set_state(q_pen, np.zeros(n))
            set_agreement = float(
                np.max(np.abs(backend.get_state()["qpos"][0] - reference.qvel()[0]))
            )

            zero = np.zeros((backend.num_envs, n))
            contact_steps = 0
            max_dev = 0.0
            for _ in range(CONTACT_STEPS):
                backend.step(zero, 1)
                reference.step(np.zeros(n), 1)
                q_ref, _ = reference.qvel()
                max_dev = max(
                    max_dev, float(np.max(np.abs(backend.get_state()["qpos"][0] - q_ref)))
                )
                if reference.contact_point_count(probes) > 0:
                    contact_steps += 1
        finally:
            reference.close()

        self.record(
            "contact_recovery",
            set_agreement <= TOL_TRAJECTORY
            and contact_steps > 0
            and max_dev <= TOL_TRAJECTORY,
            {
                "penetrating_joint": reference.names[joint],
                "penetrating_qpos_rad": round(float(q_pen[joint]), 4),
                "set_state_agreement_rad": set_agreement,
                "reference_contact_steps": contact_steps,
                "adapter_vs_sdk_max_dev_rad": max_dev,
            },
        )

    def check_reset(
        self, backend, reference: Reference, resolved: ResolvedProfile
    ) -> None:
        n = reference.n
        q0 = backend.get_default_qpos().copy()
        probe = float(np.min(np.asarray(resolved.effort_limits))) * 0.5

        backend.reset()
        backend.step(np.full((backend.num_envs, n), probe), 100)
        backend.reset()
        full_dev = float(np.max(np.abs(backend.get_state()["qpos"] - q0)))
        reference.set_state(np.asarray(q0, dtype=np.float32), np.zeros(n))
        ref_q0, _ = reference.qvel()
        default_dev = float(np.max(np.abs(backend.get_state()["qpos"][0] - ref_q0)))

        backend.reset()
        backend.step(np.full((backend.num_envs, n), probe), 100)
        before = backend.get_state()
        backend.reset(np.array([0]))
        after = backend.get_state()
        selective_dev = float(np.max(np.abs(after["qpos"][0] - q0)))
        env1_untouched = bool(
            np.array_equal(after["qpos"][1], before["qpos"][1])
            and np.array_equal(after["qvel"][1], before["qvel"][1])
        )

        # State round trip through set_state/get_state with a bounded ramp.
        q_arb = q0 + np.linspace(0.1, -0.1, n if n else 0)
        v_arb = np.linspace(0.05, -0.05, n if n else 0)
        backend.set_state(np.array([1]), q_arb[None, :], v_arb[None, :])
        rb = backend.get_state()
        roundtrip_q = float(np.max(np.abs(rb["qpos"][1] - q_arb)))
        roundtrip_v = float(np.max(np.abs(rb["qvel"][1] - v_arb)))

        self.record(
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

    def check_isolation(self, backend, resolved: ResolvedProfile) -> None:
        n = resolved.n
        q0 = backend.get_default_qpos().copy()
        amplitudes = np.asarray(
            sweep_amplitudes(self.profile, q0, backend.get_joint_range(), REACHABLE_FRACTION)
        )
        single = self.make_backend(resolved, 1)

        def replay(env_index: int, hold: bool) -> float:
            backend.reset()
            single.reset()
            deviation = 0.0
            for k in range(60):
                target = self.sweep_target(q0, amplitudes, k)
                own_target = q0 if hold else target
                tau_own = self.pd_torque(
                    resolved,
                    own_target,
                    backend.get_dof_pos()[env_index],
                    backend.get_dof_vel()[env_index],
                )
                commands = np.zeros((backend.num_envs, n))
                commands[env_index] = tau_own
                other = 1 - env_index
                other_target = q0 if not hold else target
                commands[other] = self.pd_torque(
                    resolved,
                    other_target,
                    backend.get_dof_pos()[other],
                    backend.get_dof_vel()[other],
                )
                backend.step(commands, DECIMATION)
                tau_single = self.pd_torque(
                    resolved, own_target, single.get_dof_pos()[0], single.get_dof_vel()[0]
                )
                single.step(tau_single[None, :], DECIMATION)
                deviation = max(
                    deviation,
                    float(
                        np.max(np.abs(backend.get_dof_pos()[env_index] - single.get_dof_pos()[0]))
                    ),
                )
            return deviation

        try:
            dev0 = replay(env_index=0, hold=False)
            dev1 = replay(env_index=1, hold=True)
        finally:
            single.close()

        self.record(
            "isolation",
            dev0 <= TOL_TRAJECTORY and dev1 <= TOL_TRAJECTORY,
            {
                "env0_vs_single_max_dev_rad": dev0,
                "env1_vs_single_max_dev_rad": dev1,
            },
        )

    def check_lifecycle(self, resolved: ResolvedProfile) -> None:
        n = resolved.n
        deviations = []
        for _ in range(3):
            backend = self.make_backend(resolved, 1)
            try:
                backend.step(np.zeros((1, n)), 10)
                backend.reset()
                deviations.append(
                    float(
                        np.max(
                            np.abs(backend.get_state()["qpos"] - backend.get_default_qpos())
                        )
                    )
                )
            finally:
                backend.close()
        self.record(
            "lifecycle",
            all(dev <= TOL_RESET for dev in deviations),
            {"cycles": len(deviations), "reset_deviations_rad": deviations},
        )


def probe_blocked(bot_path: Path, key: str, assets_root: Path) -> dict:
    """Probe one blocked bot for its precise compiled defect.

    Prefab parsing needs no runtime. The adapter probe uses serial mode so a
    hypothetical 0-DoF articulation is not rejected by the batch executor
    before its native features are inspected; a batch-mode rejection is then
    recorded as a separate blocker (the qualification runner itself runs the
    default batch mode).
    """
    import superdex.physics as physics

    entry: dict = {"key": key, "bot": bot_path.relative_to(assets_root).as_posix()}
    load: dict = {}
    try:
        cfg = _load_prefab(bot_path)
        jt = physics.ArticulatedJointType
        names, ranges, efforts, armature = authored_joint_metadata(cfg)
        unsupported = sorted(
            {
                str(j.type).split(".")[-1]
                for j in list(cfg.joints)[1:]
                if j.type not in (jt.HARD, jt.REVOLUTE, jt.PRISMATIC)
            }
        )
        load = {
            "root_joint": str(cfg.joints[0].type).split(".")[-1],
            "cycles": len(cfg.cycles),
            "components": sum(
                len(link.sensors) + len(link.actuators) for link in cfg.links
            ),
            "unsupported_joint_types": unsupported,
            "active_joints": len(names),
        }
    except Exception as exc:  # noqa: BLE001 — recorded as evidence
        load = {"error": f"{type(exc).__name__}: {exc}"}
    entry["compiled"] = load

    def construct(serial: bool) -> str | None:
        try:
            from unisim import create_backend
            from unisim.scene import SceneCfg

            kwargs: dict = {}
            if serial:
                kwargs.update(superdex_execution_mode="serial", superdex_num_workers=0)
            backend = create_backend("superdex", SceneCfg(str(bot_path)), 1, SIM_DT, **kwargs)
            backend.close()
            return None
        except Exception as exc:  # noqa: BLE001 — recorded as evidence
            return f"{type(exc).__name__}: {str(exc)[:300]}"

    serial_error = construct(serial=True)
    batch_error = None if serial_error is not None else construct(serial=False)
    entry["adapter_serial_error"] = serial_error
    entry["adapter_batch_error"] = batch_error

    blockers: list[str] = []
    if isinstance(load, dict) and "error" not in load:
        if load["root_joint"] not in ("HARD", "FREE"):
            blockers.append(f"unsupported root ({load['root_joint']} root)")
        if load["cycles"]:
            blockers.append(f"mechanical cycles ({load['cycles']})")
        if load["components"]:
            blockers.append(f"actuator/sensor components ({load['components']})")
        if load["unsupported_joint_types"]:
            blockers.append(
                "unsupported joint types: " + ", ".join(load["unsupported_joint_types"])
            )
    if not blockers and serial_error is not None:
        blockers.append(f"adapter rejection: {serial_error}")
    if not blockers and batch_error is not None:
        # Loads in serial mode but the default batch mode rejects it (e.g. the
        # 0-DoF torso: SceneBatchExecutor requires articulated actors with
        # DoFs); qualification runs the default batch mode, so it is blocked.
        blockers.append(f"batch-executor rejection: {batch_error}")
    entry["blockers"] = blockers
    return entry


def _sdk_precision() -> str:
    import superdex.physics as physics

    return "float64" if physics.uses_double_precision() else "float32"


def run_one(profile: BotProfile, assets_root: Path, out: Path, tree_digest: str | None) -> int:
    print(f"== {profile.key}: {profile.relpath}")
    bot_path = assets_root / profile.relpath
    if not bot_path.is_file():
        print(f"    error: fixture missing: {bot_path}", file=sys.stderr)
        return 1

    # Prefab metadata (no runtime needed) validates the profile first.
    cfg = _load_prefab(bot_path)
    names, ranges, efforts, armature = authored_joint_metadata(cfg)
    try:
        resolved = resolve_gains(profile, names, armature, efforts)
    except ValueError as exc:
        print(f"    error: {exc}", file=sys.stderr)
        return 1

    run = QualificationRun(profile, bot_path)
    # The first backend acquires the shared runtime; the reference scene is
    # created afterwards and destroyed before the last backend closes.
    backend = run.make_backend(resolved, 2)
    reference = None
    serial_backend = None
    started = time.perf_counter()
    try:
        reference = Reference(bot_path, SIM_DT)
        run.report["actuator_order"] = list(backend.get_actuator_names())
        run.report["effort_limits_nm"] = list(resolved.effort_limits)
        run.report["pd_gains"] = {
            "kp": list(resolved.kp),
            "kd": list(resolved.kd),
            "dof_armature": list(resolved.dof_armature),
            "armature_floor": profile.armature_floor,
        }
        run.report["sweep_amplitudes_rad"] = [
            round(float(a), 4)
            for a in sweep_amplitudes(profile, backend.get_default_qpos(),
                                      backend.get_joint_range(), REACHABLE_FRACTION)
        ]
        serial_backend = run.make_backend(resolved, 1, serial=True)
        run.check_recipe(assets_root)
        run.check_structure(backend, reference)
        run.check_control(resolved)
        run.check_trajectory_equivalence(backend, serial_backend, reference, resolved)
        run.check_contact_recovery(backend, resolved)
        run.check_reset(backend, reference, resolved)
        run.check_isolation(backend, resolved)
        run.check_lifecycle(resolved)
    finally:
        if reference is not None:
            reference.close()
        if serial_backend is not None:
            serial_backend.close()
        backend.close()
        run.report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        run.report["provenance"] = {
            "code_commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "asset_root": str(assets_root),
            "tree_digest": tree_digest,
            "sdk_precision": _sdk_precision(),
            "gravity": [0, 0, -9.81],
            "visual_verification": "separate (scripts/superdex_bot_viewer.py --bot <key>)",
        }
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{profile.key}.json").write_text(
            json.dumps(run.report, indent=2) + "\n", encoding="utf-8"
        )
    if run.failures:
        print(f"    FAILED: {', '.join(run.failures)}", file=sys.stderr)
        return 1
    print(f"    all {len(run.report['checks'])} checks passed ({run.report['elapsed_seconds']}s)")
    return 0


def write_compatibility_table(out: Path, qualified: list[dict], blocked: list[dict]) -> None:
    lines = [
        "# Stage-3 bot compatibility table",
        "",
        "Generated by `scripts/superdex_bot_qualify.py` from the qualification",
        "reports in this directory. Qualified means every check passed with the",
        "adapter matched against a direct SuperDex SDK scene driven with",
        "identical inputs in the same process. Visual verification is separate",
        "(`scripts/superdex_bot_viewer.py --bot <key>`).",
        "",
        "## Qualified candidates",
        "",
        "| Bot | Joints | Checks passed | Effort limits | Report |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for entry in qualified:
        limits = entry.get("effort_limits_nm") or []
        shape = (
            f"{min(limits):g}..{max(limits):g} N·m over {len(limits)} joints"
            if limits
            else "authored"
        )
        lines.append(
            f"| `{entry['key']}` — `{entry['bot']}` | {len(entry.get('actuator_order', []))} "
            f"| {len(entry['checks'])}/{len(entry['checks'])} | {shape} | `{entry['key']}.json` |"
        )
    lines += [
        "",
        "## Blocked candidates (precise blockers)",
        "",
        "| Bot | Compiled evidence | Adapter result | Blockers |",
        "| --- | --- | --- | --- |",
    ]
    for entry in blocked:
        compiled = entry.get("compiled") or {}
        evidence = (
            ", ".join(
                f"{field}={compiled[field]}"
                for field in ("root_joint", "cycles", "components", "unsupported_joint_types")
                if compiled.get(field)
            )
            or compiled.get("error", "load error")
        )
        if entry.get("adapter_serial_error"):
            adapter = entry["adapter_serial_error"]
        elif entry.get("adapter_batch_error"):
            adapter = f"loads (serial); batch: {entry['adapter_batch_error']}"
        else:
            adapter = "loads (serial + batch)"
        blockers = "; ".join(entry["blockers"]) or "unexpectedly loads"
        lines.append(
            f"| `{entry['key']}` — `{entry['bot']}` | {evidence} | {adapter} | {blockers} |"
        )
    lines.append("")
    (out / "compatibility-table.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--assets",
        type=str,
        default=None,
        help="asset bundle root (default $SUPERDEX_ASSETS_PATH or assets/superdex)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPOSITORY_ROOT / "docs" / "superdex-bots-qualification",
        help="directory for the qualification reports",
    )
    parser.add_argument(
        "--bots",
        type=str,
        default=None,
        help="comma-separated profile keys (default: all registered candidates)",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="skip the recorded-manifest tree verification",
    )
    parser.add_argument(
        "--skip-blocked-probes",
        action="store_true",
        help="skip probing blocked candidates for precise evidence",
    )
    args = parser.parse_args(argv)

    root = resolve_assets_root(args.assets)
    tree_digest = None
    if args.skip_verification:
        print("asset verification: skipped")
    else:
        started = time.perf_counter()
        summary = verify_asset_bundle(root, INVENTORY_REPORT)
        tree_digest = summary["tree_digest"]
        print(
            f"asset verification: {summary['file_count']} files, "
            f"{summary['total_bytes'] / (1024 * 1024):.1f} MiB, tree "
            f"{tree_digest[:12]} ({time.perf_counter() - started:.1f}s)"
        )

    if args.bots:
        keys = [k.strip() for k in args.bots.split(",") if k.strip()]
        unknown = [k for k in keys if k not in PROFILES]
        if unknown:
            parser.error(f"unknown bot keys: {unknown} (available: {sorted(PROFILES)})")
    else:
        keys = sorted(PROFILES)

    overall = 0
    qualified: list[dict] = []
    try:
        for key in keys:
            code = run_one(PROFILES[key], root, args.out, tree_digest)
            overall |= code
            report_path = args.out / f"{key}.json"
            if code == 0 and report_path.is_file():
                qualified.append(json.loads(report_path.read_text(encoding="utf-8")))
        if not args.skip_blocked_probes:
            print("== blocked candidates (probing precise blockers)")
            blocked_entries = [
                probe_blocked(root / relpath, key, root)
                for key, relpath in sorted(BLOCKED_KEYS.items())
            ]
            for entry in blocked_entries:
                summary = "; ".join(entry["blockers"]) or "loads cleanly"
                print(f"    {entry['key']}: {summary}")
            (args.out / "blocked-probes.json").write_text(
                json.dumps(blocked_entries, indent=2) + "\n", encoding="utf-8"
            )
        else:
            blocked_entries = []
        write_compatibility_table(args.out, qualified, blocked_entries)
    finally:
        # The adapter shuts the process runtime down when its last backend
        # closes; nothing to release here.
        pass

    if overall:
        print("error: one or more candidates failed qualification", file=sys.stderr)
        return 1
    print(f"reports: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
