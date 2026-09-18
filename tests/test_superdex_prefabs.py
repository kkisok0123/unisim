"""SDK-free prefab audits, synthetic runtime regressions and opt-in asset qualification."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from unisim.backend.superdex.scenes import audit_prefab
from unisim.scene import SceneCfg


def write(path, data):
    path.write_text(json.dumps(data))
    return path


def test_nested_audit_counts_instances_and_detects_cycles(tmp_path):
    (tmp_path / "shape.h5").touch()
    child = write(
        tmp_path / "child.mochi_prefab",
        {
            "actors": {"rigid": [{"shape": "./shape.h5"}]},
        },
    )
    parent = write(
        tmp_path / "parent.mochi_prefab",
        {
            "prefabs": [
                {"path": "./child.mochi_prefab", "name": "a"},
                {"path": "./child.mochi_prefab", "name": "b"},
            ]
        },
    )
    assert audit_prefab(parent, tmp_path) == 2
    write(child, {"prefabs": [{"path": "./parent.mochi_prefab"}]})
    with pytest.raises(ValueError, match="cyclic"):
        audit_prefab(parent, tmp_path)


@pytest.mark.parametrize(
    "data",
    [
        {"controllers": [{}]},
        {"constraints": {}},
        {"scene": {"gravity": [0, 0, 0]}},
        {"actors": {"soft": []}},
        {"actors": {"articulated": [{}]}},
        {"actors": {"rigid": [{"sensors": [], "shape": "./missing.h5"}]}},
        {"prefabs": [{"path": "./missing.mochi_prefab", "sensor": {}}]},
        {"contactFilter": {}},
    ],
)
def test_unsupported_prefab_content_rejected_before_sdk(tmp_path, data):
    path = write(tmp_path / "fixture.mochi_prefab", data)
    with pytest.raises(NotImplementedError, match="superdex prefab"):
        audit_prefab(path, tmp_path)


def test_missing_dependencies_and_wrong_fragment_format(tmp_path):
    path = write(
        tmp_path / "fixture.mochi_prefab",
        {
            "actors": {"rigid": [{"shape": "./missing.h5"}]},
        },
    )
    with pytest.raises(FileNotFoundError, match="missing dependency"):
        audit_prefab(path, tmp_path)
    with pytest.raises(NotImplementedError, match=".mochi_prefab"):
        audit_prefab(tmp_path / "fragment.xml", tmp_path)


@pytest.fixture(params=["serial", "batch"])
def synthetic(request, tmp_path, monkeypatch):
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        pytest.skip("SuperDex wheels require Python 3.12 or 3.13")
    p = pytest.importorskip("superdex.physics")
    pytest.importorskip("mujoco")
    from unisim import create_backend
    from unisim.backend.superdex import materialization
    from unisim.backend.superdex.materialization import primitive_shape
    from unisim.backend.superdex.scenes import compose_rigid_prefabs

    model = tmp_path / "robot.xml"
    model.write_text("""<mujoco><option gravity="0 0 0"/><worldbody>
      <body name="root"><freejoint/><geom type="sphere" size=".05" mass="1"/>
      <body name="arm" pos="0 0 .2"><joint name="hinge"/>
      <geom type="sphere" size=".02" mass=".1"/></body></body></worldbody>
      <actuator><motor joint="hinge" ctrlrange="-1 1"/></actuator></mujoco>""")
    (tmp_path / "shape.h5").touch()
    fragment = write(
        tmp_path / "objects.mochi_prefab",
        {
            "actors": {
                "rigid": [
                    {"shape": "./shape.h5"},
                    {"shape": "./shape.h5"},
                ]
            }
        },
    )

    def load(*args):
        # Replace file loading only. Native actors/state/batch execution remain real.
        shape = primitive_shape(p, "box", [0.02] * 3, [0, 0, 0], [1, 0, 0, 0])
        return p.prefab.ScenePrefab(
            actors=p.prefab.ActorLists(
                rigid=[
                    p.prefab.RigidActorPrefab(
                        name="object",
                        shape=shape,
                        mass=1,
                        center_of_mass=[0.003, -0.002, 0.001],
                        moment_of_inertia=[0.001, 0, 0, 0.001, 0, 0.001],
                        translation=[1, 0, 1],
                        linear_velocity=[0.1, 0, 0],
                    ),
                    p.prefab.RigidActorPrefab(
                        name="support", shape=shape, is_static=True, translation=[1, 0, 0]
                    ),
                ]
            )
        )

    def materialize(physics, robotics, scene, **kwargs):
        plan = materialization._mjcf_plan(physics, model, SceneCfg(str(model)), None, False)
        return compose_rigid_prefabs(physics, plan, scene)

    monkeypatch.setattr(p.prefab, "load_from_file", load)
    monkeypatch.setattr(materialization, "materialize_model", materialize)
    b = create_backend(
        "superdex",
        SceneCfg(str(model), fragment_files=[str(fragment)]),
        2,
        0.002,
        superdex_execution_mode=request.param,
    )
    try:
        yield b
    finally:
        b.close()
        b.close()


def test_complete_state_frames_reset_and_force(synthetic):
    b = synthetic
    assert (b.model.nq, b.model.nv, b.model.robot_nq, b.model.robot_nv) == (15, 13, 8, 7)
    layout = b.get_root_state_layout("object")
    assert layout.qpos_indices == tuple(range(8, 15))
    assert layout.qvel_indices == tuple(range(7, 13))
    initial = b.get_state()
    np.testing.assert_allclose(initial["qvel"][:, 7], 0.1)
    q, v = initial["qpos"][[0]].copy(), initial["qvel"][[0]].copy()
    q[0, 8:11] = [2, 1, 3]
    q[0, 11:15] = [np.cos(0.3), 0, np.sin(0.3), 0]
    v[0, 7:] = [0.2, -0.1, 0.3, 0.4, 0.5, -0.6]
    b.set_state(np.array([0]), q, v)
    for key, expected in (("qpos", q), ("qvel", v)):
        np.testing.assert_allclose(b.get_state()[key][[0]], expected, atol=2e-6)
        np.testing.assert_array_equal(b.get_state()[key][1], initial[key][1])
    np.testing.assert_allclose(
        b.get_body_lin_vel_w(b.get_body_ids(["object"]))[0, 0], v[0, 7:10], atol=2e-6
    )
    b.reset()
    force = np.zeros((2, 1, 3))
    force[0, 0, 0] = 1
    b.apply_body_force(b.get_body_ids(["object"]), force)
    b.step(np.zeros((2, 1)), nsteps=5)
    state = b.get_state()
    np.testing.assert_allclose(state["qvel"][:, 10:13], 0, atol=1e-6)
    assert state["qvel"][0, 7] > state["qvel"][1, 7] + 0.005
    b.reset(np.array([0]))
    for key in initial:
        np.testing.assert_allclose(b.get_state()[key][0], initial[key][0], atol=2e-6)
        np.testing.assert_array_equal(b.get_state()[key][1], state[key][1])
    with pytest.raises(ValueError, match="static"):
        b.apply_body_force(b.get_body_ids(["support"]), force)
    # Validate every quaternion before mutating any selected scene.
    before = b.get_state()
    invalid = before["qpos"].copy()
    invalid[1, 11:15] = 0
    with pytest.raises(ValueError, match="rigid quaternion"):
        b.set_state(np.array([0, 1]), invalid, before["qvel"])
    for key in before:
        np.testing.assert_array_equal(b.get_state()[key], before[key])
    b.reset()
    for key in initial:
        np.testing.assert_allclose(b.get_state()[key], initial[key], atol=2e-6)


def test_local_nested_prefab_qualification(tmp_path):
    root = os.environ.get("SUPERDEX_ASSETS_PATH")
    if not root:
        pytest.skip("set SUPERDEX_ASSETS_PATH for local prefab qualification")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import superdex_compare as compare

    from unisim.backend.superdex.assets import verify_asset_bundle

    verify_asset_bundle(Path(root).resolve(), compare.INVENTORY_REPORT)
    bot = Path(root).resolve() / compare.PREFAB_BOT
    task = compare.WorkerTask(
        label="fr3_prefabs",
        path=str(bot),
        kind="bot",
        prefab_contact_root=str(Path(root).resolve()),
    )
    result = compare.run_model_subprocess(task, timeout=900)
    assert result["status"] == "passed", result.get("worker_output_tail", "")
    for mode in ("serial", "batch"):
        assert result["checks"]["prefab_contact_" + mode]["robot_object_contact_steps"] > 0


def test_partial_prefab_instantiation_cleans_up(synthetic, monkeypatch):
    import superdex.physics as p

    from unisim import create_backend

    b = synthetic
    mode = b._execution_mode
    model = Path(b.model.source_file)
    b.close()
    original = p.prefab.add_to_scene

    def fail_after_creation(cfg, world):
        result = original(cfg, world)
        if world.get_name() == "UniSim SuperDex 1":
            raise RuntimeError("injected failure after rigid creation")
        return result

    monkeypatch.setattr(p.prefab, "add_to_scene", fail_after_creation)
    scene = SceneCfg(str(model), fragment_files=[str(model.parent / "objects.mochi_prefab")])
    with pytest.raises(RuntimeError, match="injected failure"):
        create_backend("superdex", scene, 2, 0.002, superdex_execution_mode=mode)
    assert not p.is_initialized()
    monkeypatch.setattr(p.prefab, "add_to_scene", original)
    recreated = create_backend("superdex", scene, 2, 0.002, superdex_execution_mode=mode)
    recreated.close()
    assert not p.is_initialized()


def test_generic_composition_compares_rigid_geometry_and_state():
    root = os.environ.get("SUPERDEX_ASSETS_PATH")
    if not root:
        pytest.skip("set SUPERDEX_ASSETS_PATH for composition qualification")
    pytest.importorskip("superdex.physics")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import superdex_compare as compare

    root = Path(root).resolve()
    result = compare.run_model_subprocess(
        compare.WorkerTask(
            label="composition",
            path=str(root / compare.PREFAB_BOT),
            kind="bot",
            composition=[str(root / compare.PREFAB_SPHERE), str(root / compare.PREFAB_BOARD)],
        )
    )
    assert result["status"] == "passed", result.get("worker_output_tail")
    for suffix in ("", "_batch"):
        assert result["checks"]["trajectory_equivalence" + suffix]["passed"]
        assert result["checks"]["reset" + suffix]["passed"]
