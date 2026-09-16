#!/usr/bin/env python3
"""Visualize a qualified robot through UniSim's SuperDex adapter.

The command loads the selected profile from the repository-local asset copy,
verifies it against the recorded inventory, and opens the native Polyscope
viewer through the adapter's ``run_playback`` path. The default profile remains
``fr3_v2``. Fixed-base and floating profiles are registered in
``scripts/superdex_bot_profiles.py``. The demonstration runs three observable
phases:

1. Initial pose: hold the authored default pose so the geometry, scale and
   link alignment can be inspected.
2. Bounded movement: a phase-shifted sine sweep around the default pose,
   bounded per joint by the authored joint ranges.
3. Reset: ``backend.reset()`` restores the authored default pose in view, and
   the restored joint state is checked numerically.

Each profile supplies explicit per-asset control and camera settings. Floating
profiles additionally demonstrate bounded root translation and rotation.

Close the window to exit; the viewer and backend are released in ``finally``
blocks, including on errors. Requires the optional SuperDex runtime, Polyscope
>= 2.5.0 and a graphical session (or ``--frames N`` for an offscreen smoke
run). Asset setup:

    export SUPERDEX_ASSETS_PATH="$PWD/assets/superdex"
    uv run scripts/superdex_bot_viewer.py                     # fr3_v2
    uv run scripts/superdex_bot_viewer.py --bot openarm_v20_wuji
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from superdex_bot_profiles import (  # noqa: E402
    DEFAULT_KEY,
    FLOATING_PROFILES,
    PROFILES,
    resolve_gains,
    sweep_amplitudes,
)

from unisim.backend.superdex.assets import verify_asset_bundle  # noqa: E402

PROFILES = {**PROFILES, **FLOATING_PROFILES}

INVENTORY_REPORT = REPOSITORY_ROOT / "docs" / "superdex-assets-inventory.json"

SIM_DT = 0.002
CTRL_DT = 1.0 / 60.0
DECIMATION = int(round(CTRL_DT / SIM_DT))

# Phase schedule in wall-clock demo seconds; the movement amplitude per joint
# comes from the bot profile bounded by the authored reachable range.
HOLD_SECONDS = 2.0
MOVE_SECONDS = 6.0
SETTLE_SECONDS = 1.0
MOVE_PERIOD = 2.0
# Fraction of the reachable range around the default pose used when bounding
# the requested sweep amplitude (kept identical to the qualification runner).
REACHABLE_FRACTION = 0.6


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


def _authored_joint_metadata(bot_path: Path):
    """(names, ranges, efforts, armature) of a prefab's active joints.

    Prefab parsing needs no process runtime; the SDK's verbose joint dumps on
    recipe compilation are silenced.
    """
    import contextlib
    import io

    import numpy as np
    import superdex.physics as physics
    import superdex.robotics as robotics

    with contextlib.redirect_stdout(io.StringIO()):
        cfg = robotics.load_bot_prefab_from_file(str(bot_path))
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
    return tuple(names), ranges, efforts, armature


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bot",
        type=str,
        default=DEFAULT_KEY,
        choices=sorted(PROFILES),
        help=f"profile key (default {DEFAULT_KEY}; see scripts/superdex_bot_profiles.py)",
    )
    parser.add_argument(
        "--assets",
        type=str,
        default=None,
        help="asset bundle root (default $SUPERDEX_ASSETS_PATH or assets/superdex)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory for the run report "
        "(default docs/superdex-bots-viewer/<key> or docs/superdex-fr3-viewer for fr3_v2)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="run offscreen for N rendered frames instead of opening a window",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="skip the recorded-manifest tree verification",
    )
    args = parser.parse_args(argv)

    profile = PROFILES[args.bot]
    root = resolve_assets_root(args.assets)
    bot_path = root / profile.relpath
    if args.out is None:
        args.out = (
            REPOSITORY_ROOT / "docs" / "superdex-fr3-viewer"
            if args.bot == "fr3_v2"
            else REPOSITORY_ROOT / "docs" / "superdex-bots-viewer" / args.bot
        )
    args.out = args.out.expanduser().resolve()
    if args.skip_verification:
        print("asset verification: skipped")
    else:
        started = time.perf_counter()
        summary = verify_asset_bundle(root, INVENTORY_REPORT)
        print(
            f"asset verification: {summary['file_count']} files, "
            f"{summary['total_bytes'] / (1024 * 1024):.1f} MiB, tree "
            f"{summary['tree_digest'][:12]} ({time.perf_counter() - started:.1f}s)"
        )

    from unisim import create_backend
    from unisim.scene import SceneCfg

    args.out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    # Resolve the control profile from the authored prefab before any backend
    # exists (prefab parsing needs no process runtime): explicit effort limits
    # and critically damped kd from the authored armature.
    names, _, efforts, armature = _authored_joint_metadata(bot_path)
    try:
        resolved = resolve_gains(profile, names, armature, efforts)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    backend = create_backend(
        "superdex",
        SceneCfg(str(bot_path)),
        1,
        SIM_DT,
        superdex_execution_mode="serial",
        superdex_num_workers=0,
        superdex_effort_limits=list(resolved.effort_limits),
    )

    default_state = backend.get_default_qpos()
    q0 = backend.get_dof_pos()[0].copy()
    floating = backend.model.floating
    body_ids = np.arange(1, len(backend.model.body_names))
    # Apply gravity compensation at each link's COM through the public API.
    # This viewer profile keeps a free hand in frame; qualification uses gravity.
    support = -backend.get_body_mass()[body_ids, None] * backend.get_gravity()[None, :]
    joint_ranges = backend.get_joint_range()
    # Movement stays inside the authored joint ranges around the default pose.
    amplitudes = np.asarray(sweep_amplitudes(profile, q0, joint_ranges, REACHABLE_FRACTION))
    if floating:
        # A joint authored at its limit needs a one-sided sweep. Targets are
        # clipped below instead of assigning that joint zero amplitude.
        amplitudes = np.minimum(profile.sweep_amplitude, .3 * np.diff(joint_ranges)[:, 0])
    kp, kd, effort = (
        np.asarray(v, dtype=float) for v in (resolved.kp, resolved.kd, resolved.effort_limits)
    )

    def position_control(backend, ctrl):
        q = backend.get_dof_pos()[0]
        v = backend.get_dof_vel()[0]
        torque = np.clip(kp * (ctrl[0] - q) - kd * v, -effort, effort)
        return torque[None, :]

    backend.set_pre_step_control(position_control)

    # Phase machine driven from the run_playback render loop. One callback
    # call advances DECIMATION physics substeps (sim_dt each), matching the
    # viewer's 60 Hz pacing.
    phase = "initial-pose"
    phase_time = 0.0
    report = {
        "asset_root": str(root),
        "bot": profile.relpath,
        "bot_key": profile.key,
        "tree_verified": not args.skip_verification,
        "sim_dt": SIM_DT,
        "ctrl_dt": CTRL_DT,
        "decimation": DECIMATION,
        "actuator_order": list(backend.get_actuator_names()),
        "effort_limits_nm": list(resolved.effort_limits),
        "pd_gains": {"kp": list(resolved.kp), "kd": list(resolved.kd)},
        "move_amplitudes_rad": [round(float(a), 4) for a in amplitudes],
        "floating_root": floating,
        "root_profile": "per-link gravity compensation; movement starts with world linear "
                        "velocity [0.015,0,0.005] m/s and body angular [0,0.04,0] rad/s",
        "phases": {},
    }
    movement_peak = 0.0
    root_movement_peak = 0.0
    reset_error = None

    def enter_phase(name: str) -> bool:
        """Run one-time phase entry; True exactly on each phase's first callback."""
        nonlocal phase, phase_time, reset_error
        if phase != name:
            return False
        if phase_time > 0.0:
            return False
        if name == "initial-pose":
            if profile.camera_direction is not None:
                from superdex.physics.viewer.backend import polyscope as ps

                boxes = [link.get_aabb_world() for link in backend._links[0]]
                lo = np.min([box.min for box in boxes], axis=0)
                hi = np.max([box.max for box in boxes], axis=0)
                center = (lo + hi) / 2
                direction = np.asarray(profile.camera_direction)
                eye = center + direction / np.linalg.norm(direction) * np.linalg.norm(hi - lo) * 2
                ps.look_at(eye, center)
            print(f"[{name}] holding authored default pose (inspect geometry)")
        elif name == "bounded-movement":
            if floating:
                state = backend.get_state()
                state["qvel"][0, :3] = [0.015, 0, 0.005]
                state["qvel"][0, 3:6] = [0, 0.04, 0]
                backend.set_state(np.array([0]), state["qpos"], state["qvel"])
            print(f"[{name}] sine sweep, amplitudes {[round(float(a), 3) for a in amplitudes]} rad")
        elif name == "reset":
            backend.reset()
            reset_error = float(np.max(np.abs(backend.get_state()["qpos"][0] - default_state)))
            report["phases"]["reset"] = {"restored_qpos_max_error_rad": reset_error}
            print(
                f"[{name}] backend.reset() restored default pose "
                f"(max qpos error {reset_error:.2e} rad)"
            )
        return True

    def demo_step(obs):
        nonlocal phase, phase_time, movement_peak, root_movement_peak
        target = q0.copy()
        enter_phase(phase)
        if phase == "initial-pose":
            if phase_time >= HOLD_SECONDS:
                phase, phase_time = "bounded-movement", -CTRL_DT
        elif phase == "bounded-movement":
            target = q0 + amplitudes * np.sin(
                2.0 * math.pi * phase_time / MOVE_PERIOD
                + np.linspace(0.0, 3.0 * math.pi / 2.0, len(q0))
            )
            if floating:
                target = np.clip(target, joint_ranges[:, 0], joint_ranges[:, 1])
                root_movement_peak = max(root_movement_peak, float(np.linalg.norm(
                    backend.get_state()["qpos"][0, :3] - default_state[:3]
                )))
            movement_peak = max(movement_peak, float(np.max(np.abs(backend.get_dof_pos()[0] - q0))))
            if phase_time >= MOVE_SECONDS:
                report["phases"]["bounded-movement"] = {
                    "peak_abs_deviation_rad": movement_peak,
                    "peak_root_displacement_m": root_movement_peak,
                }
                phase, phase_time = "reset", -CTRL_DT
        elif phase == "reset":
            if phase_time >= SETTLE_SECONDS:
                settled = float(np.max(np.abs(backend.get_dof_pos()[0] - q0)))
                report["phases"]["reset"]["settled_max_deviation_rad"] = settled
                print(
                    f"[done] reset verified (settle deviation {settled:.3f} rad under PD hold); "
                    "close the window to exit"
                )
                phase, phase_time = "done", -CTRL_DT
        if floating:
            backend.apply_body_force(body_ids, support[None], np.zeros_like(support)[None])
        backend.step(target[None, :], DECIMATION)
        phase_time += CTRL_DT
        return obs

    frames = args.frames
    started = time.perf_counter()
    interrupted = False
    try:
        backend.run_playback(
            env=None,
            initialize=lambda: None,
            step=demo_step,
            num_steps=frames,
            headless=frames is not None,
            record_video=False,
        )
    except KeyboardInterrupt:
        # Ctrl-C must still release the viewer and backend via the finally
        # blocks below; treat it as a clean interactive exit.
        interrupted = True
    finally:
        backend.cleanup_scene_assets()
        report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        report["viewer"] = "offscreen" if frames is not None else "interactive"
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if interrupted:
        print("interrupted: viewer and backend released")
        return 0 if reset_error is not None else 1
    print(f"report: {args.out / 'report.json'}")
    if reset_error is None:
        print("error: the reset phase never ran (closed too early?)", file=sys.stderr)
        return 1
    if reset_error > 1e-6:
        print(f"error: reset did not restore the default pose ({reset_error:.2e})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
