#!/usr/bin/env python3
"""Visualize the unchanged FR3 through UniSim's SuperDex adapter (stage 2A).

One standalone command loads ``bots/arms/fr3_v2/fr3_v2.superdex_bot`` from the
repository-local asset copy, verifies it against the recorded inventory, and
opens the native Polyscope viewer through the adapter's ``run_playback``
interactive path (serial execution, one environment, no recording). The
demonstration runs three distinct, observable phases:

1. Initial pose: hold the authored default pose so the geometry, scale and
   link alignment can be inspected.
2. Bounded movement: a phase-shifted sine sweep around the default pose,
   bounded per joint by the authored joint ranges.
3. Reset: ``backend.reset()`` restores the authored default pose in view, and
   the restored joint state is checked numerically.

Stage 3 adds ``--bot <key>`` to reuse the same demonstration for any bot
registered in ``scripts/superdex_bot_profiles.py`` (fixed-base arms and
recipe compositions), each with its explicit per-asset control profile.

Close the window to exit; the viewer and backend are released in ``finally``
blocks, including on errors. Requires the optional SuperDex runtime, Polyscope
>= 2.5.0 and a graphical session (or ``--frames N`` for an offscreen smoke
run). Asset setup:

    export SUPERDEX_ASSETS_PATH="$PWD/assets/superdex"
    uv run scripts/superdex_fr3_viewer.py                     # fr3_v2
    uv run scripts/superdex_fr3_viewer.py --bot openarm_v20_wuji
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
import zlib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from superdex_bot_profiles import (  # noqa: E402
    DEFAULT_KEY,
    PROFILES,
    resolve_gains,
    sweep_amplitudes,
)

from unisim.backend.superdex.assets import verify_asset_bundle  # noqa: E402

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


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    body = tag + data
    return len(data).to_bytes(4, "big") + body + (zlib.crc32(body) & 0xFFFFFFFF).to_bytes(4, "big")


def write_png(path: Path, rgb: "object") -> None:
    """Write one (H, W, 3|4) uint8 screenshot buffer as an RGB PNG, stdlib only."""
    import numpy as np

    frame = np.flipud(np.ascontiguousarray(rgb))
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(f"unsupported screenshot buffer shape {frame.shape}")
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    height, width, _ = frame.shape
    raw = b"".join(b"\x00" + frame[row].tobytes() for row in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, 6))
        + _png_chunk(b"IEND", b"")
    )


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
        if joint.type == jt.HARD:
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
        help="directory for captures and the run report "
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

    q0 = backend.get_default_qpos()
    joint_ranges = backend.get_joint_range()
    # Movement stays inside the authored joint ranges around the default pose.
    amplitudes = np.asarray(sweep_amplitudes(profile, q0, joint_ranges, REACHABLE_FRACTION))
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
        "phases": {},
    }
    captures: list[tuple[str, Path]] = []
    pending_capture: list[str] = []

    def grab_pending() -> None:
        if not pending_capture:
            return
        from superdex.physics.viewer.backend import polyscope as ps

        name = pending_capture.pop(0)
        path = args.out / f"{profile.key}_{name}.png"
        write_png(path, ps.screenshot_to_buffer())
        captures.append((name, path))

    def mark_capture(name: str) -> None:
        # The buffer holds the last rendered frame; capture one callback later.
        if pending_capture:
            grab_pending()
        pending_capture.append(name)

    movement_peak = 0.0
    reset_error = None
    movement_captured = False

    def enter_phase(name: str) -> bool:
        """Run one-time phase entry; True exactly on each phase's first callback."""
        nonlocal phase, phase_time, reset_error
        if phase != name:
            return False
        if phase_time > 0.0:
            return False
        if name == "initial-pose":
            print(f"[{name}] holding authored default pose (inspect geometry)")
            mark_capture("initial_pose")
        elif name == "bounded-movement":
            print(f"[{name}] sine sweep, amplitudes {[round(float(a), 3) for a in amplitudes]} rad")
        elif name == "reset":
            backend.reset()
            reset_error = float(np.max(np.abs(backend.get_state()["qpos"][0] - q0)))
            report["phases"]["reset"] = {"restored_qpos_max_error_rad": reset_error}
            print(
                f"[{name}] backend.reset() restored default pose "
                f"(max qpos error {reset_error:.2e} rad)"
            )
            mark_capture("reset")
        return True

    def demo_step(obs):
        nonlocal phase, phase_time, movement_peak, movement_captured
        grab_pending()
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
            movement_peak = max(movement_peak, float(np.max(np.abs(backend.get_dof_pos()[0] - q0))))
            if not movement_captured and phase_time >= MOVE_SECONDS / 2.0:
                mark_capture("movement")
                movement_captured = True
            if phase_time >= MOVE_SECONDS:
                report["phases"]["bounded-movement"] = {
                    "peak_abs_deviation_rad": movement_peak,
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
        try:
            grab_pending()
        except Exception:
            pass
        backend.cleanup_scene_assets()
        report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        report["captures"] = [str(path.relative_to(REPOSITORY_ROOT)) for _, path in captures]
        report["viewer"] = "offscreen" if frames is not None else "interactive"
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if interrupted:
        print("interrupted: viewer and backend released")
        return 0 if reset_error is not None else 1
    print(f"report: {args.out / 'report.json'}")
    for name, path in captures:
        print(f"capture[{name}]: {path}")
    if reset_error is None:
        print("error: the reset phase never ran (closed too early?)", file=sys.stderr)
        return 1
    if reset_error > 1e-6:
        print(f"error: reset did not restore the default pose ({reset_error:.2e})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
