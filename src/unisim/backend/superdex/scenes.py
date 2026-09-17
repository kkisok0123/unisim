"""Audited single-articulation native scenes, without task or SDK-source dependencies."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from unisim.scene import SceneCfg, resolve_scene_fragment_path

from .geometry import rotation_matrix
from .plans import ModelPlan
from .prefabs import _reference, _root, append_rigid_metadata, audit_prefab

_JOINT_FIELDS = {
    "name", "type", "axis", "friction", "inertia", "limitDamping", "limitStiffness",
    "minLimit", "maxLimit", "parentLinkFromJoint",
}
_LINK_FIELDS = {
    "name", "parentLink", "parentJointFromLink", "shape", "renderModel", "mass", "density",
    "centerOfMass", "momentOfInertia", "hasGravity", "layer", "colliderType", "contact",
    "boundaryElementType", "boundarySubsampling", "shapeTranslation", "shapeRotation",
    "shapeScale", "renderModelTranslation", "renderModelRotation", "renderModelScale",
}
_SOLVER_FIELDS = {"integrationMethod", "linearSolver", "nonLinearSolver", "experimentalEval"}
_CONTACT_FIELDS = {
    "collidingPenaltyLengthScale", "coulombFrictionCoefficient", "distanceErrorBound",
    "frictionFalloffVel", "frictionWithColliderNormal", "maxAlignmentNormals",
    "normalViscousDampingCoefficient", "objScale", "penaltyCoefficient",
    "penaltySmoothingHalfDistance", "penaltyThresholdDefault", "penaltyThresholdExtraPadding",
    "viscousFrictionCoefficient",
}
_SOLVER_CHILDREN = {
    "linearSolver": {
        "abortIfNotSpd", "absTol", "maxIter", "normType", "preconditionerType", "relDivTol",
        "relTol", "restartSize", "solverType", "verbosity",
    },
    "nonLinearSolver": {
        "absDivTol", "absTol", "convergenceMode", "dResidualAssemblyPeriod", "explosionControl",
        "gradientDescentFallback", "lineSearchAlpha", "lineSearchMaxIter",
        "lineSearchMaxRelIncrease", "lineSearchType", "lineSearchWolfe1", "lineSearchWolfe2",
        "linearToleranceStrategy", "maxElapsedTimeSeconds", "maxIter", "psdProjMode",
        "relDivTol", "relStepTol", "relTol", "solverType", "stopIfNoImprovement", "verbosity",
    },
    "experimentalEval": {
        "consistencyResNorm", "consistencyResNormStep", "explicitNormals", "fadeFriction",
        "fittedSaturationHessian", "frictionModel", "implicitNormalForceForDissipation",
    },
}


def _fields(value: Any, allowed: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"superdex scene {label} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise NotImplementedError(f"superdex scene {label}: unsupported fields {sorted(unknown)}")


def _finite(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif isinstance(value, list):
        for child in value:
            _finite(child)
    elif isinstance(value, (int, float)) and not np.isfinite(value):
        raise ValueError("superdex scene numbers must be finite")


def _vector(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"superdex scene {label} must have {size} finite values")
    return result


def _transform(value: Any) -> None:
    _fields(value, {"translation", "rotation"}, "transform")
    if "translation" in value:
        _vector(value["translation"], 3, "translation")
    if "rotation" in value:
        q = _vector(value["rotation"], 4, "rotation")
        if not np.isclose(np.linalg.norm(q), 1, atol=1e-5):
            raise ValueError("superdex scene rotation must be a normalized xyzw quaternion")


def audit_scene(path: Path, root: Path) -> dict[str, Any]:
    """Validate the Stage 7 profile without importing or invoking the SDK.

    One articulation must be authored in the root file. Nested files are rigid
    prefabs, optionally declaring scene settings identical to the root settings.
    """
    data = json.loads(path.read_text())
    _fields(data, {"comment", "actors", "scene", "prefabs", "contactFilter", "controllers"},
            str(path))
    _finite(data)
    settings = data.get("scene", {})
    _fields(settings, {"comment", "description", "gravity", "solver"}, "settings")
    if "gravity" in settings:
        _vector(settings["gravity"], 3, "gravity")
    if "solver" in settings:
        _fields(settings["solver"], _SOLVER_FIELDS, "solver")
        for key, allowed in _SOLVER_CHILDREN.items():
            if key in settings["solver"]:
                _fields(settings["solver"][key], allowed, f"solver.{key}")
        fitted = settings["solver"].get("experimentalEval", {}).get("fittedSaturationHessian")
        if fitted is not None:
            _fields(fitted, {"constraintSaturation", "contactFriction", "jointFriction"},
                    "fittedSaturationHessian")
    actors = data.get("actors", {})
    _fields(actors, {"comment", "articulated", "rigid"}, "actors")
    if len(actors.get("articulated", [])) != 1:
        raise NotImplementedError("superdex scene requires exactly one root-file articulation")
    articulation = actors["articulated"][0]
    _fields(articulation, {"comment", "name", "joints", "links", "translation", "rotation",
                           "scale", "jointVelocities"}, "articulation")
    if not isinstance(articulation.get("name"), str) or not articulation["name"]:
        raise ValueError("superdex scene articulation requires a name")
    if articulation.get("scale", 1) != 1:
        raise NotImplementedError("superdex scene articulation scale must be 1")
    _transform({k: articulation[k] for k in ("translation", "rotation") if k in articulation})
    joints, links = articulation.get("joints", []), articulation.get("links", [])
    if not joints or len(joints) != len(links):
        raise ValueError("superdex scene requires one joint per link")
    for index, (joint, link) in enumerate(zip(joints, links, strict=True)):
        _fields(joint, _JOINT_FIELDS, "joint")
        _fields(link, _LINK_FIELDS, "link")
        if joint.get("type") not in {"Hard", "Revolute", "Prismatic"}:
            raise NotImplementedError("superdex scene supports only fixed/hinge/slide joints")
        if joint["type"] != "Hard":
            axis = _vector(joint.get("axis"), 3, "joint axis")
            if np.linalg.norm(axis) == 0:
                raise ValueError("superdex scene joint axis must be nonzero")
        if "friction" in joint:
            _fields(joint["friction"], {"viscous", "coulomb", "falloffVel", "stictionExtra",
                                        "stribeckVel"}, "joint friction")
        if "contact" in link:
            _fields(link["contact"], _CONTACT_FIELDS, "link contact")
        for key in ("minLimit", "maxLimit"):
            if key in joint:
                _vector(joint[key], 3, key)
        _transform(joint.get("parentLinkFromJoint", {}))
        _transform(link.get("parentJointFromLink", {}))
        parent = link.get("parentLink", -1)
        if not isinstance(parent, int) or parent < -1 or parent >= index:
            raise ValueError("superdex scene links must be in parent-before-child order")
        for key in ("shape", "renderModel"):
            if link.get(key):
                _reference(path, link[key], root)
    for records, label in ((joints, "joint"), (links, "link")):
        names = [record.get("name") for record in records]
        if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
            raise ValueError(f"superdex scene {label} names must be nonempty and unique")
    if "jointVelocities" in articulation:
        _vector(articulation["jointVelocities"],
                sum(j["type"] != "Hard" for j in joints), "jointVelocities")
    controllers = data.get("controllers", [])
    if len(controllers) > 1:
        raise NotImplementedError("superdex scene supports one authored pose controller")
    for controller in controllers:
        _fields(controller, {"comment", "articulatedActor", "jointTracking"}, "controller")
        if controller.get("articulatedActor") != articulation["name"]:
            raise ValueError("superdex scene controller target is not the articulation")
        tracking = controller.get("jointTracking", [])
        if len(tracking) != len(joints):
            raise ValueError("superdex scene jointTracking must have one entry per joint")
        for item in tracking:
            _fields(item, {"stiffness", "damping"}, "jointTracking")
            if any(not isinstance(v, (int, float)) or v < 0 for v in item.values()):
                raise ValueError("superdex scene controller gains must be nonnegative")
    contact_filter = data.get("contactFilter", {})
    _fields(contact_filter, {"layerContactSymmetric"}, "contactFilter")
    for item in contact_filter.get("layerContactSymmetric", []):
        _fields(item, {"enable", "layers"}, "contactFilter entry")
        if (not isinstance(item.get("enable"), bool) or len(item.get("layers", [])) != 2
                or any(not isinstance(v, str) for v in item["layers"])):
            raise ValueError("superdex scene contact filter requires two layers and enable bool")
    _audit_rigids(data, path, root, settings, ())
    return data


def _audit_rigids(data, path, root, settings, stack) -> int:
    from .prefabs import _RIGID_FIELDS

    if path in stack:
        raise ValueError(f"superdex cyclic prefab dependency: {path}")
    count = 0
    for rigid in data.get("actors", {}).get("rigid", []):
        _fields(rigid, _RIGID_FIELDS, "rigid actor")
        if "contact" in rigid:
            _fields(rigid["contact"], _CONTACT_FIELDS, "rigid contact")
        if not rigid.get("shape"):
            raise ValueError("superdex scene rigid actors require collision shapes")
        for key in ("shape", "renderModel"):
            if rigid.get(key):
                _reference(path, rigid[key], root)
        count += 1
    for nested in data.get("prefabs", []):
        _fields(nested, {"comment", "name", "path", "translation", "rotation", "scale"},
                "nested prefab")
        _transform({k: nested[k] for k in ("translation", "rotation") if k in nested})
        target = _reference(path, nested["path"], root)
        if target.suffix != ".mochi_prefab":
            raise NotImplementedError("superdex scene nested files must be rigid .mochi_prefab")
        child = json.loads(target.read_text())
        _fields(child, {"comment", "actors", "prefabs", "scene"}, "nested prefab")
        _finite(child)
        _fields(child.get("actors", {}), {"comment", "rigid"}, "nested actors")
        _fields(child.get("scene", {}), {"comment", "description", "gravity", "solver"},
                "nested settings")
        for key, value in child.get("scene", {}).items():
            if key not in {"comment", "description"} and settings.get(key) != value:
                raise ValueError(f"superdex scene conflicting nested setting {key!r}: {target}")
        count += _audit_rigids(child, target, root, settings, (*stack, path))
    return count


def controlled_indices(data, controlled_joints, effort_limits) -> tuple[np.ndarray, np.ndarray]:
    """Resolve explicit physical effort inputs, never infer controls from asset names."""
    from .materialization import _effort_ranges

    if (controlled_joints is None or isinstance(controlled_joints, (str, bytes))
            or not len(controlled_joints)
            or any(not isinstance(n, str) for n in controlled_joints)):
        raise ValueError("superdex scene requires an ordered nonempty controlled_joints list")
    if len(set(controlled_joints)) != len(controlled_joints):
        raise ValueError("superdex scene controlled_joints must be unique")
    joints = data["actors"]["articulated"][0]["joints"]
    active = [j["name"] for j in joints if j["type"] != "Hard"]
    if any(n not in active for n in controlled_joints):
        raise ValueError("superdex scene controlled_joints must name active hinge/slide joints")
    if effort_limits is None:
        raise ValueError("superdex scene requires explicit finite effort_limits")
    return np.asarray([active.index(n) for n in controlled_joints]), _effort_ranges(
        effort_limits, len(controlled_joints)
    )


def native_settings(world: Any) -> dict[str, Any]:
    """Detached effective settings suitable for reports and comparisons."""
    def value(item):
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        if hasattr(item, "name"):
            return item.name
        return {k: value(getattr(item, k)) for k in dir(item)
                if not k.startswith("_") and not callable(getattr(item, k))}

    return {"gravity": list(world.get_gravity()), "solver": value(world.get_solver_params())}


def _verify_settings(authored: dict, effective: dict) -> None:
    """Fail closed if the SDK parser ignores or changes a requested setting."""
    for key, expected in authored.items():
        if key in {"comment", "description"}:
            continue
        native_key = re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()
        actual = effective.get(native_key)
        if isinstance(expected, dict) and isinstance(actual, dict):
            _verify_settings(expected, actual)
        elif isinstance(expected, str) and isinstance(actual, str):
            if expected.replace("_", "").lower() != actual.replace("_", "").lower():
                raise ValueError(f"superdex scene SDK did not retain setting {key!r}")
        elif actual is None or not np.allclose(expected, actual, rtol=1e-6, atol=1e-8):
            raise ValueError(f"superdex scene SDK did not retain setting {key!r}")


def materialize_native_scene(p, path: Path, scene: SceneCfg, controlled_joints,
                             effort_limits) -> ModelPlan:
    root = _root(path)
    data = audit_scene(path, root)
    indices, ranges = controlled_indices(data, controlled_joints, effort_limits)
    authored = data["actors"]["articulated"][0]
    extra_paths = [resolve_scene_fragment_path(v, path).resolve() for v in scene.fragment_files]
    extra_count = sum(audit_prefab(v, _root(v)) for v in extra_paths)
    expected_rigids = _audit_rigids(data, path, root, data.get("scene", {}), ()) + extra_count
    loaded = []
    try:
        loaded.append(p.prefab.load_from_file(str(path), str(root)))
        loaded.extend(p.prefab.load_from_file(str(v), str(_root(v))) for v in extra_paths)
    except BaseException:
        loaded.clear()
        raise
    owned: dict[Any, list[Any]] = {}

    def instantiate(world):
        all_actors = []
        for cfg in loaded:
            result = p.prefab.add_to_scene(cfg, world)
            if len(result.constraints):
                raise ValueError("superdex scene created unexpected constraints")
            all_actors.extend(result.actors)
        articulations = [a for a in all_actors if a.get_type() == p.ActorType.ARTICULATED]
        rigids = [a for a in all_actors if a.get_type() == p.ActorType.RIGID]
        if len(articulations) != 1 or len(rigids) != expected_rigids:
            raise ValueError("superdex scene compiled actor inventory differs from audit")
        if len(all_actors) != len(rigids) + 1:
            raise ValueError("superdex scene created unsupported actors")
        actor = articulations[0]
        if actor.get_name() != authored["name"]:
            raise ValueError("superdex scene articulation name differs from audit")
        if actor.has_articulated_pose_controller() != bool(data.get("controllers")):
            raise ValueError("superdex scene compiled controller inventory differs from audit")
        return actor, rigids

    def spawn(world):
        actor, rigids = instantiate(world)
        owned[world.get_handle()] = rigids

        def close():
            owned.pop(world.get_handle(), None)

        return actor, close

    temp = p.create_scene("superdex_scene_metadata")
    try:
        actor, rigids = instantiate(temp)
        _verify_settings(data.get("scene", {}), native_settings(temp))
        links = [temp.get_actor(h) for h in actor.get_nested_link_actors()]
        joints = list(loaded[0].actors.articulated[0].joints)
        active = [i for i, j in enumerate(joints) if j.type != p.ArticulatedJointType.HARD]
        n = len(active)
        if actor.get_num_dofs() != n or len(links) != len(joints):
            raise ValueError("superdex scene compiled joint/link inventory differs from audit")
        dtype = np.float64 if p.uses_double_precision() else np.float32
        q0, v0 = np.empty(n, dtype=dtype), np.empty(n, dtype=dtype)
        actor.get_articulated_pose(q0)
        actor.get_articulated_joint_velocities(v0)
        names = tuple(joints[i].name for i in active)
        limits = []
        for i in active:
            joint = joints[i]
            axis = np.asarray(joint.axis, dtype=float)
            axis /= np.linalg.norm(axis)
            limits.append([-np.inf if joint.min_limit is None else np.dot(joint.min_limit, axis),
                           np.inf if joint.max_limit is None else np.dot(joint.max_limit, axis)])
        masses, coms = [0.0], [np.zeros(3)]
        for link in links:
            masses.append(0.0 if link.is_static() else link.get_mass())
            pose = link.get_root_transform()
            offset = (np.asarray(link.get_center_of_mass_transform().translation)
                      - np.asarray(pose.translation))
            coms.append(rotation_matrix(np.asarray(pose.rotation)[[3, 0, 1, 2]]).T @ offset)
        plan = ModelPlan(
            source_file=str(path), nq=n, nv=n, root_body_id=1, floating=False,
            body_names=("world", *(a.get_name() for a in links)),
            body_parent_ids=np.asarray([0, *(a.get("parentLink", -1) + 1
                                            for a in authored["links"])]),
            body_link_indices=np.asarray([-1, *range(len(links))]),
            body_mass=np.asarray(masses), body_ipos=np.asarray(coms), joint_names=names,
            joint_qpos_indices=np.arange(n), joint_qvel_indices=np.arange(n),
            joint_ranges=np.asarray(limits).reshape(n, 2),
            actuator_names=tuple(controlled_joints), actuator_joint_names=tuple(controlled_joints),
            actuator_qpos_indices=indices, actuator_qvel_indices=indices.copy(),
            actuator_ctrl_ranges=ranges, actuator_force_ranges=ranges.copy(),
            actuator_gear=np.ones(len(indices)), actuator_kp=np.zeros(len(indices)),
            actuator_kd=np.zeros(len(indices)), default_qpos=q0, default_qvel=v0,
            keyframes={}, gravity=np.asarray(temp.get_gravity()), sensors=(),
            spawn_actor=spawn, cleanup=loaded.clear,
            dof_armature=np.asarray([float(joints[i].inertia or 0) for i in active]),
            effective_scene_settings=native_settings(temp),
        )
        if actor.has_articulated_pose_controller():
            params = p.PoseControllerParams()
            params.joint_tracking.resize(len(joints))
            params.link_pos_tracking.resize(len(links))
            params.link_rot_tracking.resize(len(links))
            actor.get_articulated_pose_controller_params(params)
            target = np.empty_like(q0)
            actor.get_articulated_target_pose(target)

            def restore_controller(instance):
                instance.set_articulated_pose_controller_params(params)
                instance.reset_articulated_target_pose(target)
                instance.set_articulated_target_velocity(v0)

            plan.restore_scene_controller = restore_controller
        append_rigid_metadata(plan, rigids)
        plan.spawn_rigids = lambda world: owned[world.get_handle()]
        if len(set(plan.body_names)) != len(plan.body_names):
            raise ValueError("superdex scene body names must be unique")
        return plan
    except BaseException:
        loaded.clear()
        owned.clear()
        raise
    finally:
        p.destroy_scene(temp)
