#!/usr/bin/env python3
"""Qualify unchanged Cart Pole/Half Cheetah scenes, or inspect them in the native viewer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from superdex_bot_qualify import INVENTORY_REPORT, _git, resolve_assets_root  # noqa: E402

from unisim import create_backend  # noqa: E402
from unisim.backend.superdex.assets import verify_asset_bundle  # noqa: E402
from unisim.backend.superdex.scenes import native_settings  # noqa: E402
from unisim.scene import SceneCfg  # noqa: E402

DT = 0.002
TOLERANCE = 3e-5
PROFILES = {
    "cart_pole": {"joints": ["Cart"], "limits": [3.0], "indices": [0]},
    "half_cheetah": {
        "joints": ["BackThigh", "BackShin", "BackFoot", "FrontThigh", "FrontShin", "FrontFoot"],
        "limits": [120.0, 90.0, 60.0, 120.0, 60.0, 30.0], "indices": [3, 4, 5, 6, 7, 8],
    },
}


def scene_path(root, key):
    return root / "benchmarks" / key / f"{key}.mochi_scene"


def backend(root, key, mode, count=2):
    profile = PROFILES[key]
    return create_backend(
        "superdex", SceneCfg(str(scene_path(root, key))), count, DT,
        superdex_execution_mode=mode, superdex_controlled_joints=profile["joints"],
        superdex_effort_limits=profile["limits"],
    )


def inventory(world):
    actors = []
    world.for_each_actor(lambda a: actors.append((a.get_name(), str(a.get_type()))))
    return sorted(actors)


def controller_state(p, actor, dtype):
    if not actor.has_articulated_pose_controller():
        return None
    count = len(actor.get_nested_link_actors())
    params = p.PoseControllerParams()
    for k in ("joint_tracking", "link_pos_tracking", "link_rot_tracking"):
        getattr(params, k).resize(count)
    actor.get_articulated_pose_controller_params(params)
    target = np.empty(actor.get_num_dofs(), dtype=dtype)
    actor.get_articulated_target_pose(target)
    return {"joint_tracking": [[v.stiffness, v.damping] for v in params.joint_tracking],
            "target": target.tolist()}


def qualify(root, key, steps=1000):
    import superdex.physics as p

    results = {}
    for mode in ("serial", "batch"):
        b = backend(root, key, mode)
        references, snapshots, actors = [], [], []
        try:
            dtype = b.get_state()["qpos"].dtype
            profile = PROFILES[key]
            limits = np.asarray(profile["limits"], dtype=dtype)
            indices = np.asarray(profile["indices"], dtype=np.int32)
            for i in range(2):
                world = p.create_scene(f"stage7_reference_{i}")
                references.append(world)
                loaded = p.prefab.load_from_file(str(scene_path(root, key)), str(root))
                result = p.prefab.add_to_scene(loaded, world)
                actors.append(next(a for a in result.actors
                                   if a.get_type() == p.ActorType.ARTICULATED))
                snapshots.append(world.capture_state())
                world.step(0)
                assert inventory(world) == inventory(b._worlds[i])
                assert native_settings(world) == native_settings(b._worlds[i])
            initial_controller = controller_state(p, actors[0], dtype)
            errors = {"qpos": 0.0, "qvel": 0.0, "body_position": 0.0}

            def compare():
                state = b.get_state()
                for i, actor in enumerate(actors):
                    for field, method in (("qpos", "get_articulated_pose"),
                                          ("qvel", "get_articulated_joint_velocities")):
                        values = np.empty(b.model.nv, dtype=dtype)
                        getattr(actor, method)(values)
                        error = float(np.max(abs(state[field][i] - values)))
                        errors[field] = max(error, errors[field])
                        np.testing.assert_allclose(state[field][i], values,
                                                   atol=TOLERANCE, rtol=TOLERANCE)
                    native_pos = np.asarray([
                        references[i].get_actor(h).get_root_transform().translation
                        for h in actor.get_nested_link_actors()
                    ])
                    positions = b.get_body_pos_w(np.arange(1, len(b.model.body_names)))[i]
                    errors["body_position"] = max(
                        errors["body_position"], float(np.max(abs(positions - native_pos)))
                    )
                    np.testing.assert_allclose(positions, native_pos,
                                               atol=TOLERANCE, rtol=TOLERANCE)
                    assert controller_state(p, b._actors[i], dtype) == initial_controller

            def reset_references():
                for i, world in enumerate(references):
                    world.restore_state(snapshots[i], release_immediately=False)
                    actors[i].set_external_forces_on_dofs(
                        np.arange(b.model.nv, dtype=np.int32), np.zeros(b.model.nv, dtype=dtype)
                    )
                    world.step(0)

            def advance(ctrl):
                b.step(ctrl)
                for i, world in enumerate(references):
                    actors[i].set_external_forces_on_dofs(
                        indices, np.clip(ctrl[i], -limits, limits)
                    )
                    world.step(DT)
                compare()

            compare()
            # Independent per-input probes catch permutations and accidentally driven passive DoFs.
            for joint in range(len(limits)):
                b.reset()
                reset_references()
                ctrl = np.zeros((2, len(limits)), dtype=dtype)
                ctrl[:, joint] = limits[joint] * 2  # Exercise clipping as well as action routing.
                advance(ctrl)
            b.reset()
            reset_references()
            for step in range(steps):
                phase = step * DT * 5 + np.arange(len(limits)) * 0.7
                ctrl = np.asarray([np.sin(phase), np.cos(phase)], dtype=dtype) * limits * 0.2
                advance(ctrl)
            moved = b.get_state()
            b.set_state(np.arange(2), moved["qpos"], moved["qvel"])
            for field in ("qpos", "qvel"):
                np.testing.assert_allclose(b.get_state()[field], moved[field],
                                           atol=TOLERANCE, rtol=TOLERANCE)
            other = {k: v[1].copy() for k, v in b.get_state().items()}
            b.reset(np.array([0]))
            for field in other:
                np.testing.assert_array_equal(b.get_state()[field][1], other[field])
            b.reset()
            reset_references()
            compare()
            # Replaying after moved-state/reset exercises retained rest-spring targets.
            advance(np.zeros((2, len(limits)), dtype=dtype))
            results[mode] = {
                "status": "passed", "steps": steps, "num_envs": 2,
                "max_errors": errors, "actor_inventory": inventory(b._worlds[0]),
                "effective_settings": native_settings(b._worlds[0]),
                "controller": initial_controller,
                "checks": ["initial-state", "controls", "effort-clipping", "trajectory",
                           "body-positions", "state-roundtrip", "whole-scene-reset",
                           "selective-reset", "environment-isolation", "controller-preservation"],
            }
        finally:
            for i, world in enumerate(references):
                if i < len(snapshots):
                    world.release_state(snapshots[i])
                p.destroy_scene(world)
            # Release prefab shape ownership before the runtime is shut down.
            loaded = None
            b.close()
            b.close()
        reopened = backend(root, key, mode)
        reopened.close()
        results[mode]["checks"].append("cleanup-recreation")
    return results


def view(root, key, frames):
    b = backend(root, key, "serial", 1)
    frame, error = 0, None

    def advance(obs):
        nonlocal frame, error
        if 60 <= frame < 180:
            ctrl = np.sin((frame - 60) * 0.04 + np.arange(b.num_actuators))
            b.step((ctrl * np.asarray(PROFILES[key]["limits"]) * 0.1)[None], nsteps=4)
        if frame == 180:
            b.reset()
            error = float(np.max(abs(b.get_state()["qpos"][0] - b.get_default_qpos())))
        frame += 1
        return obs

    try:
        b.run_playback(
            env=SimpleNamespace(cfg=SimpleNamespace(ctrl_dt=1 / 60)),
            initialize=lambda: None, step=advance, num_steps=frames,
            headless=frames is not None, record_video=False,
        )
    finally:
        b.close()
    if frame <= 180 or error is None or error > TOLERANCE:
        raise RuntimeError("scene viewer did not complete initial pose, motion and reset")
    return {"status": "passed", "frames": frame, "reset_max_error": error,
            "phases": ["initial-pose", "controlled-motion", "whole-scene-reset"],
            "visual_inspection": "manual inspection outstanding; renderer smoke only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets")
    parser.add_argument("--scenes", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.frames is not None and (not args.viewer or args.frames < 181):
        parser.error("--frames requires --viewer and at least 181 frames")
    root = resolve_assets_root(args.assets)
    os.environ["SUPERDEX_ASSETS_PATH"] = str(root)
    verified = verify_asset_bundle(root, INVENTORY_REPORT)
    import superdex.physics as p

    report = {
        "code_commit": _git("rev-parse", "HEAD"), "working_tree": _git("status", "--short"),
        "python": platform.python_version(), "platform": platform.platform(),
        "sdk": importlib.metadata.version("superdex-physics-uni"),
        "precision": "float64" if p.uses_double_precision() else "float32",
        "dt": DT, "tolerance": TOLERANCE, "profiles": PROFILES,
        "asset_verification": verified,
        "manifest_sha256": hashlib.sha256(INVENTORY_REPORT.read_bytes()).hexdigest(),
        "source_sha256": {str(v.relative_to(ROOT)): hashlib.sha256(v.read_bytes()).hexdigest()
                          for v in [*sorted((ROOT / "src/unisim/backend/superdex").glob("*.py")),
                                    ROOT / "src/unisim/factory.py", Path(__file__).resolve()]},
        "scene_sha256": {key: hashlib.sha256(scene_path(root, key).read_bytes()).hexdigest()
                         for key in args.scenes},
        "results": {},
    }
    for key in args.scenes:
        try:
            report["results"][key] = (view(root, key, args.frames) if args.viewer
                                      else qualify(root, key))
        except Exception as exc:
            report["results"][key] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    out = args.out or ROOT / "docs/superdex-scene-qualification" / (
        "viewer.json" if args.viewer else "report.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["results"], indent=2))
    if any(v.get("status") == "failed" for v in report["results"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
