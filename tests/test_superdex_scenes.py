"""SDK-free scene audits plus native synthetic and unchanged-asset qualification."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from unisim.backend.superdex.scenes import audit_scene, controlled_indices
from unisim.scene import SceneCfg


def data():
    return {"actors": {"articulated": [{
        "name": "robot",
        "joints": [{"name": "slide", "type": "Prismatic", "axis": [1, 0, 0]},
                   {"name": "fixed", "type": "Hard"}],
        "links": [{"name": "root", "mass": 1}, {"name": "tip", "parentLink": 0, "mass": 1}],
    }]}}


def write(path, value):
    path.write_text(json.dumps(value))
    return path


def test_scene_controls_require_explicit_active_joint_mapping(tmp_path):
    source = data()
    source["actors"]["articulated"][0]["joints"].append(
        {"name": "hinge", "type": "Revolute", "axis": [0, 0, 1]}
    )
    source["actors"]["articulated"][0]["links"].append({"name": "end", "parentLink": 1})
    audited = audit_scene(write(tmp_path / "test.mochi_scene", source), tmp_path)
    indices, ranges = controlled_indices(audited, ["hinge", "slide"], [2, 3])
    np.testing.assert_array_equal(indices, [1, 0])
    np.testing.assert_array_equal(ranges, [[-2, 2], [-3, 3]])
    for selected, efforts in [(None, [1]), ("slide", [1]), ([], []),
                              (["slide", "slide"], [1, 1]), (["fixed"], [1]),
                              (["missing"], [1]), (["slide"], None),
                              (["slide"], [0]), (["slide"], [np.inf]), (["slide"], [1, 2])]:
        with pytest.raises(ValueError):
            controlled_indices(audited, selected, efforts)


@pytest.mark.parametrize("patch", [
    {"scene": {"timestep": 0.01}}, {"scene": {"solver": {"typo": 3}}},
    {"scene": {"solver": {"linearSolver": {"typo": 3}}}},
    {"constraints": {}}, {"cameras": []}, {"controllers": [{"type": "custom"}]},
    {"actors": {"soft": [], "articulated": []}},
])
def test_unsupported_scene_fields_fail_before_sdk(tmp_path, patch):
    source = data()
    source.update(patch)
    with pytest.raises((NotImplementedError, ValueError), match="superdex"):
        audit_scene(write(tmp_path / "test.mochi_scene", source), tmp_path)


@pytest.mark.parametrize("change", ["multiple", "free", "components", "cycle", "bad_parent"])
def test_unsupported_articulation_content(tmp_path, change):
    source = data()
    actor = source["actors"]["articulated"][0]
    if change == "multiple":
        source["actors"]["articulated"].append(copy.deepcopy(actor))
    elif change == "free":
        actor["joints"][0]["type"] = "Free"
    elif change == "components":
        actor["links"][0]["sensors"] = [{"type": "SENSOR_CAMERA"}]
    elif change == "cycle":
        actor["cycles"] = [{}]
    else:
        actor["links"][0]["parentLink"] = 1
    with pytest.raises((NotImplementedError, ValueError)):
        audit_scene(write(tmp_path / "test.mochi_scene", source), tmp_path)


def test_nested_settings_dependencies_and_cycles(tmp_path):
    source = data()
    source["scene"] = {"gravity": [0, -3, 0]}
    source["prefabs"] = [{"path": "./nested.mochi_prefab", "name": "objects"}]
    path = write(tmp_path / "test.mochi_scene", source)
    with pytest.raises(FileNotFoundError):
        audit_scene(path, tmp_path)
    child = {"scene": {"gravity": [0, -3, 0]}}
    nested = write(tmp_path / "nested.mochi_prefab", child)
    audit_scene(path, tmp_path)
    child["scene"]["gravity"] = [0, 0, -3]
    write(nested, child)
    with pytest.raises(ValueError, match="conflicting nested setting"):
        audit_scene(path, tmp_path)
    write(nested, {"prefabs": [{"path": "./nested.mochi_prefab"}]})
    with pytest.raises(ValueError, match="cyclic"):
        audit_scene(path, tmp_path)
    source.pop("prefabs")
    source["actors"]["articulated"][0]["links"][0]["shape"] = "./missing.h5"
    with pytest.raises(FileNotFoundError):
        audit_scene(write(path, source), tmp_path)


def test_dispatch_audits_before_native_loader(tmp_path):
    from unisim.backend.superdex.materialization import materialize_model

    source = data()
    source["scene"] = {"timestep": 0.01}
    path = write(tmp_path / "test.mochi_scene", source)
    # No SDK objects are supplied: a bad scene must fail before touching either runtime.
    with pytest.raises(NotImplementedError, match="timestep"):
        materialize_model(None, None, SceneCfg(str(path)), controlled_joints=["slide"])


def sdk():
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        pytest.skip("SuperDex wheels require Python 3.12 or 3.13")
    return pytest.importorskip("superdex.physics")


@pytest.fixture
def synthetic(tmp_path):
    p = sdk()

    source = data()
    source["scene"] = {"gravity": [0, -3, 0], "solver": {"nonLinearSolver": {"maxIter": 8}}}
    source["controllers"] = [{"articulatedActor": "robot",
                              "jointTracking": [{"stiffness": 10, "damping": 1}, {}]}]
    source["actors"]["articulated"][0]["jointVelocities"] = [0.1]
    source["prefabs"] = [{"name": "objects", "path": "./objects.mochi_prefab",
                          "translation": [0, 0, 0.5]}]
    for link in source["actors"]["articulated"][0]["links"]:
        link.update(shape="./shape.obj", colliderType="None")
    path = write(tmp_path / "test.mochi_scene", source)
    (tmp_path / "shape.obj").write_text(
        "v -.1 -.1 -.1\nv .1 -.1 -.1\nv .1 .1 -.1\nv -.1 .1 -.1\n"
        "v -.1 -.1 .1\nv .1 -.1 .1\nv .1 .1 .1\nv -.1 .1 .1\n"
        "f 1 3 2\nf 1 4 3\nf 5 6 7\nf 5 7 8\nf 1 2 6\nf 1 6 5\n"
        "f 4 8 7\nf 4 7 3\nf 1 5 8\nf 1 8 4\nf 2 3 7\nf 2 7 6\n"
    )
    write(tmp_path / "objects.mochi_prefab", {"actors": {"rigid": [
        {"name": "ball", "shape": "./shape.obj", "translation": [0, 1, 0], "mass": 1},
        {"name": "support", "shape": "./shape.obj", "isStatic": True},
    ]}})

    return path, p


def make_backend(path, mode="serial"):
    from unisim import create_backend

    return create_backend("superdex", SceneCfg(str(path)), 2, 0.002,
                          superdex_execution_mode=mode, superdex_controlled_joints=["slide"],
                          superdex_effort_limits=[3])


@pytest.mark.parametrize("mode", ["serial", "batch"])
def test_synthetic_settings_objects_controller_reset(synthetic, mode):
    path, p = synthetic
    b = make_backend(path, mode)
    try:
        np.testing.assert_array_equal(b.get_gravity(), [0, -3, 0])
        assert b._worlds[0].get_solver_params().non_linear_solver.max_iter == 8
        assert b.model.nq == 8 and b.model.nv == 7
        assert b.model.body_names == ("world", "robot/root", "robot/tip",
                                      "objects/ball", "objects/support")
        np.testing.assert_allclose(b.get_init_qvel()[0], 0.1)
        with pytest.raises(NotImplementedError, match="authored scene controller"):
            b.configure_controller("BASIC_JSC_PD")
        b.step(np.zeros((2, 1)), nsteps=400)
        state = b.get_state()
        # The dynamic box settles on the authored support at y=0.2.
        assert 0.15 < state["qpos"][0, 2] < 0.25
        b.set_state(np.array([0]), state["qpos"][:1], state["qvel"][:1])
        for k in state:
            np.testing.assert_allclose(b.get_state()[k], state[k], atol=3e-5)
        b.reset(np.array([0]))
        np.testing.assert_array_equal(b.get_state()["qpos"][1], state["qpos"][1])
        for actor in b._actors:
            target = np.empty(1, dtype=b.get_state()["qpos"].dtype)
            actor.get_articulated_target_pose(target)
            np.testing.assert_array_equal(target, [0])
        b.reset()
        np.testing.assert_array_equal(b.get_state()["qpos"][0], b.get_default_qpos())
        np.testing.assert_array_equal(b.get_state()["qvel"][0], b.get_init_qvel())
    finally:
        b.close()
        b.close()
    recreated = make_backend(path, mode)
    recreated.close()


def test_partial_scene_instantiation_cleanup(synthetic, monkeypatch):
    path, p = synthetic
    original = p.prefab.add_to_scene
    calls = 0

    def fail(cfg, world):
        nonlocal calls
        result = original(cfg, world)
        calls += 1
        if calls == 3:  # metadata and first environment succeeded; second partially instantiated.
            raise RuntimeError("injected scene failure")
        return result

    monkeypatch.setattr(p.prefab, "add_to_scene", fail)
    with pytest.raises(RuntimeError, match="injected scene failure"):
        make_backend(path)
    assert not p.is_initialized()
    monkeypatch.setattr(p.prefab, "add_to_scene", original)
    b = make_backend(path)
    b.close()


def test_scene_contact_filter_and_extra_fragment(synthetic):
    path, _ = synthetic
    source = json.loads(path.read_text())
    source["contactFilter"] = {"layerContactSymmetric": [
        {"layers": ["ghost", "ghost"], "enable": False},
    ]}
    write(path, source)
    child = path.parent / "objects.mochi_prefab"
    objects = json.loads(child.read_text())
    for actor in objects["actors"]["rigid"]:
        actor["layer"] = "ghost"
    write(child, objects)
    extra = write(path.parent / "extra.mochi_prefab", {"actors": {"rigid": [
        {"name": "extra", "shape": "./shape.obj", "isStatic": True, "translation": [3, 0, 0]},
    ]}})
    from unisim import create_backend

    b = create_backend("superdex", SceneCfg(str(path), fragment_files=[str(extra)]), 1, 0.002,
                       superdex_controlled_joints=["slide"], superdex_effort_limits=[3])
    try:
        assert b.model.body_names[-1] == "extra"
        b.step(np.zeros((1, 1)), nsteps=400)
        assert b.get_state()["qpos"][0, 2] < 0.1  # Filter disables box/support contact.
        b.reset()
        np.testing.assert_array_equal(b.get_state()["qpos"][0], b.get_default_qpos())
    finally:
        b.close()


@pytest.mark.parametrize("key", ["cart_pole", "half_cheetah"])
def test_unchanged_scene_qualification(key, monkeypatch):
    sdk()
    root = os.environ.get("SUPERDEX_ASSETS_PATH")
    if not root:
        pytest.skip("set SUPERDEX_ASSETS_PATH for unchanged-scene qualification")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from superdex_scene_qualify import qualify

    result = qualify(Path(root), key)
    assert all(v["status"] == "passed" for v in result.values())
