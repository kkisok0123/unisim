"""Runtime model helpers for the MuJoCo apple benchmark."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "mujoco/scene.xml"
DT = 0.001
GAIN_SWITCH_TIME = 23.0
REQUESTED_INTEGRATOR = "discrete"
APPLE_SDF_GEOM = "apple_with_stem_collision"
APPLE_SDF_MESH = "apple_with_stem"
APPLE_COLLISION_RGBA = (0.1, 0.8, 0.1, 0.65)
SDF_TARGET_SPACING = 0.0002


def _obj_vertices(path: Path) -> np.ndarray:
    vertices = [
        [float(value) for value in line.split()[1:4]]
        for line in path.read_text(encoding="ascii").splitlines()
        if line.startswith("v ")
    ]
    if not vertices:
        raise ValueError(f"OBJ contains no vertices: {path}")
    return np.asarray(vertices, dtype=float)


def required_sdf_octree_depth(mesh_path: Path, spacing: float = SDF_TARGET_SPACING) -> int:
    """Return a depth whose cells are no wider than spacing under any rotation."""
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("SDF spacing must be finite and positive")
    extent = np.ptp(_obj_vertices(mesh_path), axis=0)
    # The diagonal bounds every axis-aligned extent after MuJoCo recenters and
    # principal-axis-aligns the mesh during compilation.
    diameter = float(np.linalg.norm(extent))
    return max(1, math.ceil(math.log2(diameter / spacing)))


def sdf_collision_metadata(model: mujoco.MjModel) -> dict[str, object]:
    """Validate and describe the compiled apple SDF grid."""
    try:
        geom_id = int(model.geom(APPLE_SDF_GEOM).id)
    except (AttributeError, KeyError) as error:
        raise ValueError(f"MuJoCo model is missing geom {APPLE_SDF_GEOM!r}") from error
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_SDF):
        raise ValueError(f"MuJoCo geom {APPLE_SDF_GEOM!r} is not an SDF")
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        raise ValueError(f"MuJoCo SDF geom {APPLE_SDF_GEOM!r} has no mesh")
    octree_start = int(model.mesh_octadr[mesh_id])
    octree_count = int(model.mesh_octnum[mesh_id])
    if octree_start < 0 or octree_count <= 0:
        raise ValueError(f"MuJoCo SDF mesh {APPLE_SDF_MESH!r} has no compiled octree")
    octree_slice = slice(octree_start, octree_start + octree_count)
    depths = np.asarray(model.oct_depth[octree_slice], dtype=int)
    maximum_depth = int(np.max(depths))
    finest = np.asarray(model.oct_aabb[octree_slice, 3:])[depths == maximum_depth] * 2
    maximum_cell_width = float(np.max(finest))
    if maximum_cell_width > SDF_TARGET_SPACING * (1 + 1e-9):
        raise ValueError(
            "Compiled MuJoCo apple SDF is coarser than SuperDex: "
            f"{maximum_cell_width:.9g} m > {SDF_TARGET_SPACING:.9g} m"
        )
    return {
        "type": "native_mesh_sdf",
        "target_spacing_m": SDF_TARGET_SPACING,
        "octree_depth": maximum_depth,
        "octree_cells": octree_count,
        "finest_cell_width_m": finest.max(axis=0).tolist(),
        "maximum_finest_cell_width_m": maximum_cell_width,
    }


def compile_physics_model(
    path: Path | str, *, discard_visual: bool = True
) -> tuple[mujoco.MjModel, dict[str, object]]:
    """Compile the native apple mesh SDF at SuperDex's 0.2 mm target spacing."""
    scene_path = Path(path).resolve()
    spec = mujoco.MjSpec.from_file(str(scene_path))
    collision = spec.geom(APPLE_SDF_GEOM)
    if collision is None:
        raise ValueError(f"MuJoCo scene is missing geom {APPLE_SDF_GEOM!r}")
    collision.rgba = list(APPLE_COLLISION_RGBA)
    mesh = spec.mesh(APPLE_SDF_MESH)
    if mesh is None:
        raise ValueError(f"MuJoCo scene is missing mesh {APPLE_SDF_MESH!r}")
    mesh_path = scene_path.parent / "meshes" / f"{APPLE_SDF_MESH}.obj"
    mesh.octree_maxdepth = required_sdf_octree_depth(mesh_path)
    spec.compiler.discardvisual = discard_visual
    model = spec.compile()
    return model, sdf_collision_metadata(model)


def _selected_integrator() -> tuple[object, str, bool]:
    """Use discrete when the installed MuJoCo build exposes it."""
    discrete = getattr(mujoco.mjtIntegrator, "mjINT_DISCRETE", None)
    if discrete is not None:
        return discrete, "discrete", True
    # MuJoCoUni 3.11.0 currently exposes implicit but not discrete. Keep the
    # fallback visible in controller metadata instead of silently claiming parity.
    return mujoco.mjtIntegrator.mjINT_IMPLICIT, "implicit", False


INTEGRATOR, INTEGRATOR_NAME, DISCRETE_AVAILABLE = _selected_integrator()


def hinge_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
        if int(model.jnt_type[index]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    )


def build_position_gains(
    joint_names: tuple[str, ...] | list[str], *, body_grasp: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return the unchanged SuperDex pose-controller gains in hinge order."""
    kp = np.zeros(len(joint_names), dtype=float)
    kv = np.zeros(len(joint_names), dtype=float)
    for index, name in enumerate(joint_names):
        if name.startswith(("l_", "r_")):
            if body_grasp and name.startswith("r_"):
                kp[index], kv[index] = 0.8, 0.01
            else:
                kp[index] = 30.0 if name.startswith(("r_index_finger", "r_thumb")) else 0.8
                kv[index] = 0.02
        elif name.startswith("arm_openarm_"):
            kp[index], kv[index] = 1000.0, 40.0
        else:
            raise ValueError(f"Unsupported robot hinge for controller gains: {name}")
    return kp, kv


def write_runtime_scene(
    path: Path | str | None = None, *, disable_contacts: bool = False
) -> Path:
    """Add native position servos without force limits or passive damping."""
    if not SCENE.exists():
        raise FileNotFoundError(
            f"Controller-free scene is missing: {SCENE}. Run mujoco_src/export.py first."
        )
    tree = ET.parse(SCENE)
    root = tree.getroot()
    if root.find("actuator") is not None:
        raise ValueError("The controller-free source scene unexpectedly contains actuators")
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    option.set("timestep", f"{DT:.17g}")
    option.set("integrator", INTEGRATOR_NAME)
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("autoreset", "disable")
    if disable_contacts:
        flag.set("contact", "disable")

    names = tuple(
        joint.attrib["name"]
        for joint in root.iter("joint")
        if joint.attrib.get("type") == "hinge"
    )
    kp, kv = build_position_gains(names)
    actuator = ET.SubElement(root, "actuator")
    for name, stiffness, damping in zip(names, kp, kv, strict=True):
        # Damping is applied once by the servo's velocity term. Passive joint
        # damping would duplicate the unchanged SuperDex controller damping.
        joint = next(joint for joint in root.iter("joint") if joint.attrib.get("name") == name)
        joint.set("damping", "0")
        ET.SubElement(
            actuator,
            "general",
            name=f"position_{name}",
            joint=name,
            ctrllimited="false",
            forcelimited="false",
            biastype="affine",
            gainprm=f"{float(stiffness):.17g}",
            biasprm=f"0 {-float(stiffness):.17g} {-float(damping):.17g}",
        )

    output = Path(path) if path is not None else ROOT / "mujoco/runtime.xml"
    output.parent.mkdir(parents=True, exist_ok=True)
    # meshdir="." is relative to the source scene, so runtime XML must live beside it.
    output = SCENE.parent / output.name
    ET.indent(root, space="  ")
    output.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    return output


def model_joint_arrays(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    hinges = [
        index
        for index in range(model.njnt)
        if int(model.jnt_type[index]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    ]
    return (
        np.asarray([model.jnt_qposadr[index] for index in hinges], dtype=np.int32),
        np.asarray([model.jnt_dofadr[index] for index in hinges], dtype=np.int32),
    )


def load_planning_model() -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray, np.ndarray]:
    spec = mujoco.MjSpec.from_file(str(SCENE))
    apple = spec.body("apple_with_stem")
    if apple is None:
        raise ValueError("Planning scene is missing apple_with_stem")
    # IK only needs the robot. Removing the apple prevents an unnecessary SDF
    # build and keeps planning independent of the collision-grid cache.
    spec.delete(apple)
    model = spec.compile()
    data = mujoco.MjData(model)
    qadr, vadr = model_joint_arrays(model)
    return model, data, qadr, vadr


def _apply_model_gains(
    model: mujoco.MjModel, names: tuple[str, ...], kp: np.ndarray, kv: np.ndarray
) -> None:
    if model.nu != len(names):
        raise ValueError(f"Expected {len(names)} position actuators, found {model.nu}")
    actual_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    )
    expected_names = tuple(f"position_{name}" for name in names)
    if actual_names != expected_names:
        raise ValueError("MuJoCo actuator order differs from planner hinge order")
    model.actuator_gainprm[:, 0] = kp
    model.actuator_biasprm[:, 1] = -kp
    model.actuator_biasprm[:, 2] = -kv
    if not (
        np.array_equal(model.actuator_gainprm[:, 0], kp)
        and np.array_equal(model.actuator_biasprm[:, 1], -kp)
        and np.array_equal(model.actuator_biasprm[:, 2], -kv)
    ):
        raise RuntimeError("MuJoCo actuator gain readback differs from the requested profile")


def set_position_gains(
    backend, names: tuple[str, ...], *, body_grasp: bool = False
) -> tuple[np.ndarray, np.ndarray, int]:
    """Update and read back gains on the host and every pool-owned model."""
    kp, kv = build_position_gains(names, body_grasp=body_grasp)
    models = [backend.model]
    pool = getattr(backend, "_pool", None)
    if pool is not None:
        models.extend(pool.get_all_models())
    unique_models = {id(model): model for model in models}
    for model in unique_models.values():
        _apply_model_gains(model, names, kp, kv)
    return kp, kv, len(unique_models)


def controller_metadata() -> dict[str, object]:
    return {
        "type": "MuJoCo affine position-servo emulation of MOCHI_ARTICULATED_POSE",
        "physics_timestep": DT,
        "control_timestep": DT,
        "steps_per_control_update": 1,
        "requested_integrator": REQUESTED_INTEGRATOR,
        "integrator": INTEGRATOR_NAME,
        "requested_integrator_available": DISCRETE_AVAILABLE,
        "actuator_control_limits": False,
        "actuator_force_limits": False,
        "passive_joint_damping": False,
        "gain_switch_time": GAIN_SWITCH_TIME,
        "normal_gains": {
            "arm": {"kp": 1000.0, "kv": 40.0},
            "right_pinch": {"kp": 30.0, "kv": 0.02},
            "other_hand": {"kp": 0.8, "kv": 0.02},
        },
        "body_grasp_right_hand_gains": {"kp": 0.8, "kv": 0.01},
    }


def build_settling_scene(path: Path | str) -> Path:
    """Build an apple/table-only scene from the exported controller-free MJCF."""

    source = ET.parse(SCENE)
    root = source.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Exported MuJoCo scene has no worldbody")
    for body in list(worldbody.findall("body")):
        if body.attrib.get("name") != "apple_with_stem":
            worldbody.remove(body)
    for light in list(worldbody.findall("light")):
        worldbody.remove(light)
    for geom in list(worldbody.findall("geom")):
        if geom.attrib.get("name") != "table":
            worldbody.remove(geom)
    for contact in list(root.findall("contact")):
        root.remove(contact)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    # Preserve meshdir="." relative to the source scene by writing alongside it.
    output = SCENE.parent / output.name
    ET.indent(root, space="  ")
    output.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    return output
