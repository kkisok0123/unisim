#!/usr/bin/env python3
"""Numerical and lifecycle qualification of the unchanged FR3 (stage 2B).

One standalone command loads ``bots/arms/fr3_v2/fr3_v2.superdex_bot`` from the
repository-local asset copy, verifies it against the recorded inventory, and
runs the physics and lifecycle checks that stage 2A's visual demonstration did
not cover. Every adapter-side result is compared against a direct SuperDex SDK
scene driven with identical inputs in the same process, so adapter translation
is distinguished from asset/SDK behavior:

* structure — body/joint inventories, default pose, masses, world link
  transforms and world AABBs at the default pose, authored joint ranges;
* control — per-joint ordering (a single-joint torque dominates only that
  joint), effort-limit clipping (saturated commands reproduce capped dynamics
  exactly), bounded PD sweep;
* trajectories — batch and serial execution each match a direct SDK rollout
  under the same PD profile, including self-contact recovery from a pose that
  penetrates an authored joint range;
* stability — 1,000+ physics steps under the recorded profile with finite
  state and no divergence;
* reset — exact whole-scene and selective restore plus a state round trip;
* isolation — each environment of a two-environment backend reproduces the
  matching single-environment backend exactly while the other environment
  runs different commands;
* lifecycle — repeated create/step/reset/close cycles.

The run writes a JSON report with per-check evidence and provenance (code
commit, asset tree digest, SDK precision, timestep, control profile,
tolerances) under ``docs/superdex-fr3-qualification/`` and exits nonzero when
any check fails. Requires the optional SuperDex runtime; no viewer or
graphical session is needed:

    export SUPERDEX_ASSETS_PATH="$PWD/assets/superdex"
    uv run scripts/superdex_fr3_qualify.py
"""

from __future__ import annotations

import argparse
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

from unisim.backend.superdex.assets import verify_asset_bundle  # noqa: E402

BOT_RELPATH = "bots/arms/fr3_v2/fr3_v2.superdex_bot"
INVENTORY_REPORT = REPOSITORY_ROOT / "docs" / "superdex-assets-inventory.json"

SIM_DT = 0.002
DECIMATION = 8  # 160 Hz control on 500 Hz physics, the stage-2A profile

# Explicit stage-2B control profile: joint-position targets in radians, a
# pre-step PD converter producing motor torques clipped to explicit effort
# limits (N·m). Actuator order follows the authored single-DoF joints
# fr3_joint1..fr3_joint7. Research profile, not verified FR3 hardware ratings.
EFFORT_LIMITS = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
KP = [400.0, 400.0, 400.0, 400.0, 20.0, 20.0, 20.0]
# Authored joint inertia (dof_armature) of fr3_v2.superdex_bot; kd is
# critically damped against it.
JOINT_INERTIA = [0.1095493957400322] * 4 + [0.11079134792089462] * 3
KD = [2.0 * math.sqrt(kp * inertia) for kp, inertia in zip(KP, JOINT_INERTIA)]

# Bounded PD sweep around the default pose (rad), identical to stage 2A.
SWEEP_AMPLITUDES = [0.35, 0.25, 0.35, 0.30, 0.25, 0.25, 0.35]
SWEEP_PERIOD = 2.0
SWEEP_CTRL_STEPS = 130  # 130 * 8 = 1040 physics steps (stability evidence)

# Authored joint ranges of fr3_v2.superdex_bot are re-derived from the asset
# at run time (Reference.authored_joint_ranges); nothing is transcribed here.

# A pose beyond joint 2's authored range used to start self-contact recovery:
# qpos[1] = -2.6 rad < limit -1.784 rad. set_state must store it unclamped;
# contact then pushes the arm out over the following steps.
PENETRATING_QPOS = {1: -2.6}

# Numerical tolerances. With identical scene inputs the adapter and the direct
# SDK rollout integrate the same native code path, so comparisons are exact up
# to float noise; the small nonzero bounds make any translation regression
# fail loudly instead of hiding inside a tolerance.
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


def pd_torque(target: np.ndarray, q: np.ndarray, v: np.ndarray) -> np.ndarray:
    kp, kd, effort = (np.asarray(x) for x in (KP, KD, EFFORT_LIMITS))
    return np.clip(kp * (target - q) - kd * v, -effort, effort)


def sweep_target(q0: np.ndarray, ctrl_index: int) -> np.ndarray:
    t = ctrl_index * SIM_DT * DECIMATION
    phase = np.linspace(0.0, 3.0 * math.pi / 2.0, len(q0))
    return q0 + np.asarray(SWEEP_AMPLITUDES) * np.sin(2.0 * math.pi * t / SWEEP_PERIOD + phase)


class Reference:
    """A direct SuperDex SDK FR3 scene driven with identical inputs.

    The native runtime is process-global and reference-counted by the adapter,
    so the reference scene lives in the same process as the backends under
    test. It must be destroyed before the last backend closes so scene
    teardown never runs after the adapter has shut the shared runtime down.
    """

    def __init__(self, bot_path: Path, dt: float) -> None:
        import superdex.physics as physics
        import superdex.robotics as robotics

        self.p = physics
        self.cfg = robotics.load_bot_prefab_from_file(str(bot_path))
        self.scene = physics.create_scene("fr3_qualification_reference")
        self.scene.set_gravity([0, 0, -9.81])
        self.context = robotics.create_context()
        self.bot = robotics.create_bot(self.scene, self.cfg, self.context)
        self.actor = self.bot.get_articulated_actor()
        self.links = [self.scene.get_actor(h) for h in self.actor.get_nested_link_actors()]
        self.dofs = np.arange(7, dtype=np.int32)
        self.dt = dt

    def authored_joint_ranges(self) -> np.ndarray:
        """Projected authored joint limits along each joint axis (rad)."""
        rows = []
        for joint in self.cfg.joints:
            if joint.type == self.p.ArticulatedJointType.REVOLUTE:
                axis = np.asarray(joint.axis, dtype=float)
                axis = axis / np.linalg.norm(axis)
                rows.append(
                    [
                        float(np.dot(joint.min_limit, axis)),
                        float(np.dot(joint.max_limit, axis)),
                    ]
                )
        return np.asarray(rows)

    def qvel(self) -> tuple[np.ndarray, np.ndarray]:
        q, v = np.empty(7, dtype=np.float32), np.empty(7, dtype=np.float32)
        self.actor.get_articulated_pose(q)
        self.actor.get_articulated_joint_velocities(v)
        return q, v

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Mirror SuperDexBackend.set_state's native sequence exactly."""
        self.actor.set_articulated_pose_from_joints(np.asarray(qpos, dtype=np.float32))
        self.actor.set_articulated_joint_velocities(np.asarray(qvel, dtype=np.float32))
        self.actor.set_external_forces_on_dofs(self.dofs, np.zeros(7, dtype=np.float32))
        self.scene.step(0)

    def step(self, torque: np.ndarray, nsteps: int = 1) -> None:
        """Re-apply the torque before every substep, matching adapter semantics."""
        for _ in range(nsteps):
            self.actor.set_external_forces_on_dofs(self.dofs, np.asarray(torque, dtype=np.float32))
            self.scene.step(self.dt)

    def aabb(self, link_index: int) -> np.ndarray:
        box = self.links[link_index].get_aabb_world()
        return np.asarray([box.min, box.max], dtype=np.float64)

    def link_pos(self, link_index: int) -> np.ndarray:
        return np.asarray(self.links[link_index].get_root_transform().translation, dtype=np.float64)

    def register_contact_queries(self) -> None:
        """Register point queries on every link that supports contact sampling."""
        for link in self.links[2:9]:
            link.register_query(self.p.QueryType.CONTACT_POINTS)

    def contact_point_count(self) -> int:
        return sum(len(self.links[i].get_contact_points_world()) for i in range(2, 9))

    def close(self) -> None:
        import superdex.robotics as robotics

        robotics.destroy_bot(self.scene, self.bot)
        self.p.destroy_scene(self.scene)
        self.bot = None
        self.scene = None


def _adapter_link(backend, env: int, link_index: int):
    actor = backend._actors[env]
    handle = actor.get_nested_link_actors()[link_index]
    return backend._worlds[env].get_actor(handle)


def _aabb_of(link) -> np.ndarray:
    box = link.get_aabb_world()
    return np.asarray([box.min, box.max], dtype=np.float64)


class QualificationRun:
    def __init__(self, bot_path: Path) -> None:
        from unisim import create_backend
        from unisim.scene import SceneCfg

        self.create_backend = create_backend
        self.SceneCfg = SceneCfg
        self.bot_path = bot_path
        self.report: dict = {
            "bot": BOT_RELPATH,
            "sim_dt": SIM_DT,
            "decimation": DECIMATION,
            "actuator_order": [],
            "effort_limits_nm": EFFORT_LIMITS,
            "pd_gains": {"kp": KP, "kd": KD, "joint_inertia": JOINT_INERTIA},
            "sweep_amplitudes_rad": SWEEP_AMPLITUDES,
            "penetrating_qpos_rad": PENETRATING_QPOS,
            "tolerances": {
                "structure": TOL_STRUCTURE,
                "trajectory": TOL_TRAJECTORY,
                "reset": TOL_RESET,
            },
            "checks": {},
        }
        self.failures: list[str] = []

    def make_backend(self, num_envs: int, serial: bool = False):
        kwargs: dict = {"base_name": "fr3_link0", "superdex_effort_limits": EFFORT_LIMITS}
        if serial:
            kwargs.update(superdex_execution_mode="serial", superdex_num_workers=0)
        return self.create_backend(
            "superdex", self.SceneCfg(str(self.bot_path)), num_envs, SIM_DT, **kwargs
        )

    def record(self, name: str, passed: bool, evidence: dict) -> None:
        self.report["checks"][name] = {"passed": bool(passed), **evidence}
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + ("" if passed else f" — {evidence}"))
        if not passed:
            self.failures.append(name)

    # ------------------------------------------------------------------ #
    # Structure                                                          #
    # ------------------------------------------------------------------ #

    def check_structure(self, backend, reference: Reference) -> None:
        model = backend.model
        expected_bodies = ("world", "base", *(f"fr3_link{i}" for i in range(9)))
        bodies_ok = tuple(model.body_names) == expected_bodies
        expected_joints = tuple(f"fr3_joint{i}" for i in range(1, 8))
        joints_ok = backend.get_actuator_names() == expected_joints
        ctrl_order_ok = tuple(backend.get_actuator_joint_names()) == expected_joints
        joint_dofs_ok = tuple(model.joint_names) == expected_joints

        q0 = backend.get_default_qpos()
        ref_q0, _ = reference.qvel()
        q0_dev = float(np.max(np.abs(q0 - ref_q0)))

        # Authored mass metadata vs compiled SDK link masses. fr3_link0 is
        # world-welded (static): the SDK reports no mass while the adapter
        # keeps the authored 2.3966 kg by design, so compare dynamic links
        # only. Adapter bodies are [world, base, fr3_link0..8]; SDK links are
        # [base, fr3_link0..8].
        masses = backend.get_body_mass()
        mass_max = 0.0
        for body_id in range(2, len(model.body_names)):
            ref_link = reference.links[body_id - 1]
            if ref_link.is_static():
                continue
            mass_max = max(mass_max, abs(float(masses[body_id]) - float(ref_link.get_mass())))

        body_ids = np.arange(len(model.body_names))
        pos_w = backend.get_body_pos_w(body_ids)[0]
        ref_pos = np.array([reference.link_pos(i) for i in range(len(reference.links))])
        pos_dev = float(np.max(np.abs(pos_w[1:] - ref_pos)))

        aabb_dev = 0.0
        for body_id in range(2, len(model.body_names)):
            adapter_box = _aabb_of(_adapter_link(backend, 0, model.body_link_indices[body_id]))
            deviation = float(np.max(np.abs(adapter_box - reference.aabb(body_id - 1))))
            aabb_dev = max(aabb_dev, deviation)

        ranges_dev = float(
            np.max(np.abs(backend.get_joint_range() - reference.authored_joint_ranges()))
        )

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

    # ------------------------------------------------------------------ #
    # Control                                                            #
    # ------------------------------------------------------------------ #

    def check_control(self, backend) -> None:
        # Per-joint ordering by differencing: for each joint i, compare a
        # rollout with 5 N·m on joint i against an identical zero-torque
        # rollout. The velocity difference isolates control column i; the
        # torqued joint must be the most affected DoF (gravity coupling moves
        # other joints in both rollouts and cancels).
        ordering_deviations = []
        ordering_ok = True
        for joint_index in range(7):
            baseline = self.make_backend(1)
            driven = self.make_backend(1)
            try:
                baseline.step(np.zeros((1, 7)), 5)
                torque = np.zeros((1, 7))
                torque[0, joint_index] = 5.0
                driven.step(torque, 5)
                difference = driven.get_dof_vel()[0] - baseline.get_dof_vel()[0]
            finally:
                baseline.close()
                driven.close()
            top = int(np.argmax(np.abs(difference)))
            ordering_ok = ordering_ok and top == joint_index
            ordering_deviations.append(
                round(float(difference[joint_index]), 6)
            )
        # Effort clipping: a strongly saturated command must reproduce the
        # dynamics of commanding exactly each joint's effort limit.
        saturated = self.make_backend(1)
        capped = self.make_backend(1)
        try:
            limits = np.asarray(EFFORT_LIMITS)
            saturated.step(np.full((1, 7), 1e4), 200)
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
            ordering_ok and clip_ok,
            {
                "single_joint_dominant_per_joint": ordering_ok,
                "per_joint_velocity_deltas_rad_s": ordering_deviations,
                "effort_clip_exact": clip_ok,
                "effort_clip_max_dev_rad": clip_dev,
            },
        )

    # ------------------------------------------------------------------ #
    # Trajectories (also the >=1000-step stability evidence)              #
    # ------------------------------------------------------------------ #

    def check_trajectory_equivalence(
        self, backend, serial_backend, reference: Reference
    ) -> None:
        q0 = backend.get_default_qpos().copy()
        backend.reset()
        serial_backend.reset()
        reference.set_state(q0, np.zeros(7))

        batch_dev = 0.0
        serial_dev = 0.0
        max_qvel = 0.0
        for k in range(SWEEP_CTRL_STEPS):
            target = sweep_target(q0, k)
            tau_b = pd_torque(target, backend.get_dof_pos()[0], backend.get_dof_vel()[0])
            tau_s = pd_torque(
                target, serial_backend.get_dof_pos()[0], serial_backend.get_dof_vel()[0]
            )
            q_ref, v_ref = reference.qvel()
            tau_r = pd_torque(target, q_ref, v_ref)
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

        self.record(
            "trajectory_equivalence",
            batch_dev <= TOL_TRAJECTORY
            and serial_dev <= TOL_TRAJECTORY
            and finite
            and max_qvel < 10.0
            and steps >= 1000,
            {
                "physics_steps": steps,
                "batch_vs_sdk_max_dev_rad": batch_dev,
                "serial_vs_sdk_max_dev_rad": serial_dev,
                "reference_max_abs_qvel_rad_s": max_qvel,
                "finite_state": finite,
            },
        )

    def check_contact_recovery(self, backend, reference: Reference) -> None:
        reference.register_contact_queries()
        q0 = backend.get_default_qpos().copy()
        q_pen = q0.copy()
        for index, value in PENETRATING_QPOS.items():
            q_pen[index] = value

        backend.set_state(np.array([0]), q_pen[None, :], np.zeros((1, 7)))
        reference.set_state(q_pen, np.zeros(7))
        set_agreement = float(
            np.max(np.abs(backend.get_state()["qpos"][0] - reference.qvel()[0]))
        )

        zero = np.zeros((backend.num_envs, 7))
        contact_steps = 0
        max_dev = 0.0
        for _ in range(400):
            backend.step(zero, 1)
            reference.step(np.zeros(7), 1)
            q_ref, _ = reference.qvel()
            max_dev = max(max_dev, float(np.max(np.abs(backend.get_state()["qpos"][0] - q_ref))))
            if reference.contact_point_count() > 0:
                contact_steps += 1

        self.record(
            "contact_recovery",
            set_agreement <= TOL_TRAJECTORY and contact_steps > 0 and max_dev <= TOL_TRAJECTORY,
            {
                "penetrating_qpos_rad": PENETRATING_QPOS,
                "set_state_agreement_rad": set_agreement,
                "reference_contact_steps": contact_steps,
                "adapter_vs_sdk_max_dev_rad": max_dev,
            },
        )

    # ------------------------------------------------------------------ #
    # Reset                                                              #
    # ------------------------------------------------------------------ #

    def check_reset(self, backend, reference: Reference) -> None:
        q0 = backend.get_default_qpos().copy()

        # Disturb, then whole-scene reset must restore the authored default
        # pose exactly (and therefore also match the untouched SDK default).
        backend.reset()
        backend.step(np.full((backend.num_envs, 7), 3.0), 100)
        backend.reset()
        full_dev = float(np.max(np.abs(backend.get_state()["qpos"] - q0)))
        reference.set_state(np.asarray(q0, dtype=np.float32), np.zeros(7))
        ref_q0, _ = reference.qvel()
        default_dev = float(np.max(np.abs(backend.get_state()["qpos"][0] - ref_q0)))

        # Selective reset: env0 only; env1's state must stay untouched.
        backend.reset()
        backend.step(np.full((backend.num_envs, 7), 3.0), 100)
        before = backend.get_state()
        backend.reset(np.array([0]))
        after = backend.get_state()
        selective_dev = float(np.max(np.abs(after["qpos"][0] - q0)))
        env1_untouched = bool(
            np.array_equal(after["qpos"][1], before["qpos"][1])
            and np.array_equal(after["qvel"][1], before["qvel"][1])
        )

        # State round trip through set_state/get_state.
        q_arb = q0 + np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1])
        v_arb = np.array([0.05, -0.04, 0.03, -0.02, 0.01, -0.01, 0.2])
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

    # ------------------------------------------------------------------ #
    # Isolation                                                          #
    # ------------------------------------------------------------------ #

    def check_isolation(self, backend) -> None:
        q0 = backend.get_default_qpos().copy()
        single = self.make_backend(1)

        def replay(env_index: int, hold: bool) -> float:
            """Drive env `env_index` while the other sweeps/holds; replay the
            same command stream in a fresh single-env backend and return the
            maximum joint deviation."""
            backend.reset()
            single.reset()
            deviation = 0.0
            for k in range(60):
                target = sweep_target(q0, k)
                own_target = q0 if hold else target
                tau_own = pd_torque(
                    own_target, backend.get_dof_pos()[env_index], backend.get_dof_vel()[env_index]
                )
                commands = np.zeros((backend.num_envs, 7))
                commands[env_index] = tau_own
                # The other environment receives its own live PD commands, so
                # both environments are always actively driven.
                other = 1 - env_index
                other_target = q0 if not hold else target
                commands[other] = pd_torque(
                    other_target, backend.get_dof_pos()[other], backend.get_dof_vel()[other]
                )
                backend.step(commands, DECIMATION)
                tau_single = pd_torque(
                    own_target, single.get_dof_pos()[0], single.get_dof_vel()[0]
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

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def check_lifecycle(self) -> None:
        deviations = []
        for _ in range(3):
            backend = self.make_backend(1)
            try:
                backend.step(np.zeros((1, 7)), 10)
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
        default=REPOSITORY_ROOT / "docs" / "superdex-fr3-qualification",
        help="directory for the qualification report",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="skip the recorded-manifest tree verification",
    )
    args = parser.parse_args(argv)

    root = resolve_assets_root(args.assets)
    bot_path = root / BOT_RELPATH
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

    run = QualificationRun(bot_path)
    backend = run.make_backend(2)
    serial_backend = None
    reference = Reference(bot_path, SIM_DT)
    started = time.perf_counter()
    try:
        run.report["actuator_order"] = list(backend.get_actuator_names())
        serial_backend = run.make_backend(1, serial=True)
        run.check_structure(backend, reference)
        run.check_control(backend)
        run.check_trajectory_equivalence(backend, serial_backend, reference)
        run.check_contact_recovery(backend, reference)
        run.check_reset(backend, reference)
        run.check_isolation(backend)
        run.check_lifecycle()
    finally:
        # The reference scene must be destroyed before the backends release
        # the shared process runtime.
        reference.close()
        if serial_backend is not None:
            serial_backend.close()
        backend.close()
        run.report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        run.report["provenance"] = {
            "code_commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "asset_root": str(root),
            "tree_digest": tree_digest,
            "sdk_precision": "float64" if reference.p.uses_double_precision() else "float32",
            "gravity": [0, 0, -9.81],
            "visual_verification": "separate (stage 2A, docs/superdex-fr3-viewer/)",
        }
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.json").write_text(
            json.dumps(run.report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"report: {args.out / 'report.json'}")

    if run.failures:
        print("error: qualification failed: " + ", ".join(run.failures), file=sys.stderr)
        return 1
    print(f"all checks passed ({len(run.report['checks'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
