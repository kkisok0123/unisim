#!/usr/bin/env python3
"""Qualify native rigid assets from local roots against independent SDK scenes.

Each asset runs in a subprocess so native SDK failures cannot hide later results.
No source checkout is read. Supply independently copied physics assets with
--physics-root; --root defaults to the repository-local original bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from superdex_rigid_fixtures import generate  # noqa: E402


def maximum(value):
    return float(np.max(np.abs(value), initial=0))


def state(actor, dtype):
    q = np.empty(actor.get_num_dofs(), dtype)
    v = np.empty_like(q)
    actor.get_articulated_pose(q)
    actor.get_articulated_joint_velocities(v)
    return q, v


def pose(actor):
    t = actor.get_root_transform()
    return np.r_[np.asarray(t.translation), np.asarray(t.rotation)]


def native_controls(p, actors):
    """Independent action routing from compiled SDK joint sizes and offsets."""
    names, routes = [], []
    for ai, actor in enumerate(actors):
        info = actor.get_articulated_shape_info()
        prefix = f"{actor.get_name() or f'articulation_{ai}'}/" if len(actors) > 1 else ""
        for ji in range(len(actor.get_nested_link_actors())):
            if info.joint_types[ji] in (p.ArticulatedJointType.HARD, p.ArticulatedJointType.FREE):
                continue
            size = info.dof_info[ji].get_size()
            name = str(info.joint_names[ji]) or f"joint_{ji}"
            for axis in range(size):
                names.append(prefix + name + (f"/{'xyz'[axis]}" if size == 3 else ""))
                routes.append((ai, info.dof_info[ji].offset + axis))
    return names, routes


def qualify(path: Path, root: Path, steps=1000):
    from superdex import physics as p
    from superdex import robotics as r

    from unisim import create_backend
    from unisim.backend.superdex.runtime import acquire_runtime, release_runtime
    from unisim.scene import SceneCfg

    is_bot = path.suffix in {".superdex_bot", ".superdex_bot_archive"}
    backend = None
    world = None
    bot = None
    context = None
    previous_root = os.environ.get("SUPERDEX_ASSETS_PATH")
    os.environ["SUPERDEX_ASSETS_PATH"] = str(root)
    acquire_runtime(p)
    try:
        world = p.create_scene("stage8_direct_sdk")
        if is_bot:
            cfg = r.load_bot_prefab_from_file(str(path))
            # Classify custom components before spawning native plugin objects.
            for link in cfg.links:
                if len(link.actuators) or any(s.type != "SENSOR_CAMERA" for s in link.sensors):
                    return {"status": "deferred", "reason": "authored custom component"}
            context = r.create_context()
            bot = r.create_bot(world, cfg, context)
            actors, rigids = [bot.get_articulated_actor()], []
            world.set_gravity([0, 0, -9.81])
        else:
            # Recognize deferred dynamics without loading their native FEM solver.
            data = json.loads(path.read_text())
            if set(data.get("actors", {})) - {"articulated", "rigid", "comment"}:
                return {"status": "deferred", "reason": "authored deformable actors"}
            cfg = p.prefab.load_from_file(str(path), str(root))
            result = p.prefab.add_to_scene(cfg, world)
            actors = [a for a in result.actors if a.get_type() == p.ActorType.ARTICULATED]
            rigids = [a for a in result.actors if a.get_type() == p.ActorType.RIGID]
        names, routes = native_controls(p, actors)
        options = {"superdex_execution_mode": "serial", "superdex_effort_limits": 1.0}
        if not is_bot:
            options["superdex_controlled_joints"] = names
        backend = create_backend("superdex", SceneCfg(str(path)), 1, 0.002, **options)
        # White-box SDK audit: native contact samples and compiled actor inventories
        # have no shared query equivalent. Application stepping/state stay public.
        actual = (
            list(backend._actors[0].actors)
            if hasattr(backend._actors[0], "actors")
            else [backend._actors[0]]
        )
        dtype = backend.get_state()["qpos"].dtype
        assert backend.get_actuator_names() == tuple(names)
        np.testing.assert_allclose(backend.get_gravity(), world.get_gravity(), atol=1e-6)
        assert len(actual) == len(actors) and len(backend._rigids[0]) == len(rigids)
        links = [world.get_actor(h) for a in actors for h in a.get_nested_link_actors()]
        all_reference = [*links, *rigids]
        all_actual = [*backend._links[0], *backend._rigids[0]]
        contact_pairs = []
        # Some static/collision-disabled links have no contact sample points.
        for a, ref in zip([*all_actual, *actual], [*all_reference, *actors], strict=True):
            registered = []
            for item in (a, ref):
                try:
                    item.register_query(p.QueryType.TOTAL_CONTACT_FORCE)
                    registered.append(True)
                except Exception as exc:
                    if not any(
                        reason in str(exc)
                        for reason in ("contact sample points", "far SDF evaluation")
                    ):
                        raise
                    registered.append(False)
            assert registered[0] == registered[1]
            if registered[0]:
                contact_pairs.append((a, ref))
        q_initial = [state(a, dtype) for a in actors]
        # Verify authored initial state first, then use the same representable native
        # seed on both sides. Quaternion/frame round trips can round float32 by one ULP.
        for a, ref in zip(actual, actors, strict=True):
            qa, va = state(a, dtype)
            qr, vr = state(ref, dtype)
            np.testing.assert_allclose(qa, qr, atol=2e-6)
            np.testing.assert_allclose(va, vr, atol=2e-6)
            ref.set_articulated_pose_from_joints(qa)
            ref.set_articulated_joint_velocities(va)
        for a, ref in zip(backend._rigids[0], rigids, strict=True):
            np.testing.assert_allclose(pose(a), pose(ref), atol=2e-6)
            if not ref.is_static():
                ref.set_root_transform(a.get_root_transform())
                ref.set_velocity(a.get_linear_velocity(), a.get_angular_velocity())
        world.step(0)
        deviation = body_dev = contact_dev = 0.0
        action_response = np.zeros(len(names))
        commands = np.zeros((1, len(names)), dtype=dtype)
        controlled_before = np.concatenate([v for _, v in q_initial]) if actors else np.empty(0)
        contact_peak = 0.0
        for tick in range(steps):
            # Conservative bounded torque, independent of adapter state and joint layout.
            commands[0] = 0.005 * np.sin(tick * 0.013 + np.arange(len(names)))
            backend.step(commands)
            forces = [np.zeros(a.get_num_dofs(), dtype=dtype) for a in actors]
            for column, (ai, dof) in enumerate(routes):
                forces[ai][dof] = commands[0, column]
            for actor, force in zip(actors, forces, strict=True):
                actor.set_external_forces_on_dofs(np.arange(len(force), dtype=np.int32), force)
            world.step(0.002)
            for a, ref in zip(actual, actors, strict=True):
                qa, va = state(a, dtype)
                qr, vr = state(ref, dtype)
                deviation = max(deviation, maximum(qa - qr), maximum(va - vr))
                if not all(np.isfinite(x).all() for x in (qa, va, qr, vr)):
                    raise RuntimeError("non-finite native trajectory")
            for a, ref in zip(all_actual, all_reference, strict=True):
                body_dev = max(body_dev, maximum(pose(a) - pose(ref)))
            for a, ref in contact_pairs:
                fa = np.asarray(a.get_contact_force_world())
                fr = np.asarray(ref.get_contact_force_world())
                contact_dev = max(contact_dev, maximum(fa - fr))
                contact_peak = max(contact_peak, maximum(fr))
        assert deviation < 2e-5, f"native trajectory deviation {deviation}"
        assert body_dev < 2e-5, f"body transform deviation {body_dev}"
        assert contact_dev < 2e-4, f"contact force deviation {contact_dev}"
        for column, (ai, dof) in enumerate(routes):
            action_response[column] = abs(state(actual[ai], dtype)[1][dof])
        camera_deviation = 0.0
        if bot is not None:
            assert len(bot.get_sensor_handles()) == len(backend.get_camera_names())
            for handle in bot.get_sensor_handles():
                camera = bot.get_sensor(handle)
                name = camera.get_name()
                native_params = camera.get_params()
                for key, value in backend.get_camera_parameters(name).items():
                    expected = getattr(native_params, key)
                    if isinstance(value, str):
                        assert value == expected
                    else:
                        np.testing.assert_array_equal(value, np.asarray(expected))
                t = camera.get_world_transform()
                expected_pose = np.r_[
                    np.asarray(t.translation), np.asarray(t.rotation)[[3, 0, 1, 2]]
                ]
                camera_deviation = max(
                    camera_deviation, maximum(backend.get_camera_poses(name)[0] - expected_pose)
                )
            assert camera_deviation < 2e-5
        before = backend.get_state()
        backend.set_state(np.array([0]), **before)
        round_trip = max(maximum(before[k] - backend.get_state()[k]) for k in before)
        assert round_trip < 4e-5, f"state round trip {round_trip}"
        for layout in backend.get_model_info().articulations:
            if layout.floating:
                qi = layout.qpos_indices[0]
                np.testing.assert_allclose(
                    backend.get_state()["qpos"][0, qi : qi + 3],
                    backend.get_body_pos_w([layout.root_body_id])[0, 0],
                    atol=2e-5,
                )
        backend.reset()
        reset_error = max(
            maximum(backend.get_state()["qpos"][0] - backend.get_default_qpos()),
            maximum(backend.get_state()["qvel"][0] - backend.get_init_qvel()),
        )
        assert reset_error < 2e-5
        backend.close()
        backend = create_backend("superdex", SceneCfg(str(path)), 1, 0.002, **options)
        backend.step(np.zeros((1, len(names))))
        return {
            "status": "passed",
            "steps": steps,
            "sim_dt": 0.002,
            "articulations": len(actors),
            "rigid_actors": len(rigids),
            "controls": len(names),
            "native_dofs": [a.get_num_dofs() for a in actors],
            "native_state_max_deviation": deviation,
            "body_pose_max_deviation": body_dev,
            "contact_query_pairs": len(contact_pairs),
            "contact_force_max_deviation": contact_dev,
            "contact_force_peak": contact_peak,
            "state_round_trip_error": round_trip,
            "reset_error": reset_error,
            "cleanup_recreation": True,
            "control_routing_matches_sdk": True,
            "camera_count": len(backend.get_camera_names()),
            "camera_pose_max_deviation": camera_deviation,
            "passive": not bool(len(names)),
            "initial_velocity_size": len(controlled_before),
        }
    finally:
        if backend is not None:
            backend.close()
        if bot is not None:
            r.destroy_bot(world, bot)
        if world is not None:
            p.destroy_scene(world)
        context = None
        release_runtime(p)
        if previous_root is None:
            os.environ.pop("SUPERDEX_ASSETS_PATH", None)
        else:
            os.environ["SUPERDEX_ASSETS_PATH"] = previous_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY / "assets/superdex")
    parser.add_argument("--physics-root", type=Path, default=REPOSITORY / "assets/superdex-physics")
    parser.add_argument(
        "--out", type=Path, default=REPOSITORY / "docs/superdex-rigid-qualification"
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--match", default="")
    args = parser.parse_args()
    if args.worker:
        try:
            result = qualify(args.worker.resolve(), args.root.resolve(), args.steps)
        except Exception as exc:
            result = {"status": "failed", "error_type": type(exc).__name__, "reason": str(exc)}
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        return int(result["status"] == "failed")
    if args.steps < 1000:
        parser.error("qualification requires at least 1000 steps")
    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for label, root in (("bundle", args.root.resolve()), ("physics", args.physics_root.resolve())):
        if not root.is_dir():
            parser.error(f"missing local asset root: {root}")
        for path in sorted(root.rglob("*")):
            if path.suffix in {".superdex_bot", ".mochi_scene", ".mochi_prefab"}:
                entries.append((f"{label}/{path.relative_to(root)}", path, root))
    with tempfile.TemporaryDirectory(prefix="superdex-rigid-fixtures-") as directory:
        synthetic_root = Path(directory)
        entries += [
            (f"synthetic/{key}", path, synthetic_root)
            for key, path in generate(synthetic_root).items()
        ]
        reports = []
        for name, path, root in entries:
            if args.match and args.match not in name:
                continue
            output = args.out / (name.replace("/", "__") + ".json")
            output.unlink(missing_ok=True)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(path),
                    "--root",
                    str(root),
                    "--out",
                    str(output),
                    "--steps",
                    str(args.steps),
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            report = (
                json.loads(output.read_text())
                if output.exists()
                else {
                    "status": "failed",
                    "reason": "native subprocess terminated",
                    "returncode": proc.returncode,
                }
            )
            report.update(asset=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if proc.stderr or proc.returncode:
                report["diagnostic"] = (proc.stdout + proc.stderr)[-4000:]
            output.write_text(json.dumps(report, indent=2) + "\n")
            reports.append(report)
            print(name, report["status"], report.get("reason", ""), flush=True)
    from superdex import physics, robotics

    native = {}
    for label, module in (("physics", physics), ("robotics", robotics)):
        binary = Path(module._extension.__file__)
        native[label] = {
            "path": str(binary),
            "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        }
    sources = [
        *sorted((REPOSITORY / "src/unisim/backend/superdex").glob("*.py")),
        REPOSITORY / "src/unisim/backend/base.py",
        REPOSITORY / "src/unisim/backend/api_types.py",
        REPOSITORY / "src/unisim/contract.py",
        REPOSITORY / "src/unisim/__init__.py",
        Path(__file__).resolve(),
        REPOSITORY / "scripts/superdex_rigid_fixtures.py",
        REPOSITORY / "scripts/superdex_component_qualify.py",
        REPOSITORY / "scripts/superdex_rigid_viewer.py",
        REPOSITORY / "scripts/superdex_bot_viewer.py",
    ]
    source_hashes = {
        str(p.relative_to(REPOSITORY)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    manifests = {}
    for label, root in (("bundle", args.root.resolve()), ("physics", args.physics_root.resolve())):
        manifests[label] = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }
    (args.out / "fixture-manifest.json").write_text(json.dumps(manifests, indent=2) + "\n")
    summary = {
        "sdk_binaries": native,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip(),
        "working_tree_modified": True,
        "implementation_sha256": source_hashes,
        "sdk_version": importlib.metadata.version("superdex-physics-uni"),
        "platform": platform.platform(),
        "execution": "serial",
        "num_envs": 1,
        "results": reports,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return int(any(r["status"] == "failed" for r in reports))


if __name__ == "__main__":
    raise SystemExit(main())
