"""Audited native scenes, rigid prefabs, validation and prefab composition."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from unisim.scene import SceneCfg, resolve_scene_fragment_path

from .materialization import _effort_ranges, rotation_matrix
from .model import (
    ArticulationGroup,
    ArticulationLayout,
    ModelPlan,
    RigidPlan,
    RootReference,
    coordinate_groups,
    coordinate_kinds,
    joint_coordinates,
)

_JOINT_FIELDS = {
    "name",
    "type",
    "axis",
    "friction",
    "inertia",
    "limitDamping",
    "limitStiffness",
    "minLimit",
    "maxLimit",
    "parentLinkFromJoint",
}
_LINK_FIELDS = {
    "name",
    "parentLink",
    "parentJointFromLink",
    "shape",
    "renderModel",
    "mass",
    "density",
    "centerOfMass",
    "momentOfInertia",
    "hasGravity",
    "layer",
    "colliderType",
    "contact",
    "boundaryElementType",
    "boundarySubsampling",
    "shapeTranslation",
    "shapeRotation",
    "shapeScale",
    "renderModelTranslation",
    "renderModelRotation",
    "renderModelScale",
}
_SOLVER_FIELDS = {"integrationMethod", "linearSolver", "nonLinearSolver", "experimentalEval"}
_CONTACT_FIELDS = {
    "collidingPenaltyLengthScale",
    "coulombFrictionCoefficient",
    "distanceErrorBound",
    "frictionFalloffVel",
    "frictionWithColliderNormal",
    "maxAlignmentNormals",
    "normalViscousDampingCoefficient",
    "objScale",
    "penaltyCoefficient",
    "penaltySmoothingHalfDistance",
    "penaltyThresholdDefault",
    "penaltyThresholdExtraPadding",
    "viscousFrictionCoefficient",
}
_SOLVER_CHILDREN = {
    "linearSolver": {
        "abortIfNotSpd",
        "absTol",
        "maxIter",
        "normType",
        "preconditionerType",
        "relDivTol",
        "relTol",
        "restartSize",
        "solverType",
        "verbosity",
    },
    "nonLinearSolver": {
        "absDivTol",
        "absTol",
        "convergenceMode",
        "dResidualAssemblyPeriod",
        "explosionControl",
        "gradientDescentFallback",
        "lineSearchAlpha",
        "lineSearchMaxIter",
        "lineSearchMaxRelIncrease",
        "lineSearchType",
        "lineSearchWolfe1",
        "lineSearchWolfe2",
        "linearToleranceStrategy",
        "maxElapsedTimeSeconds",
        "maxIter",
        "psdProjMode",
        "relDivTol",
        "relStepTol",
        "relTol",
        "solverType",
        "stopIfNoImprovement",
        "verbosity",
    },
    "experimentalEval": {
        "consistencyResNorm",
        "consistencyResNormStep",
        "explicitNormals",
        "fadeFriction",
        "fittedSaturationHessian",
        "frictionModel",
        "implicitNormalForceForDissipation",
    },
}
# Reject fields outside the audited profile before the SDK can silently ignore them.
_RIGID_FIELDS = {
    "name",
    "comment",
    "shape",
    "renderModel",
    "translation",
    "rotation",
    "scale",
    "shapeTranslation",
    "shapeRotation",
    "renderModelTranslation",
    "renderModelRotation",
    "renderModelScale",
    "mass",
    "density",
    "centerOfMass",
    "momentOfInertia",
    "isStatic",
    "hasGravity",
    "linearVelocity",
    "angularVelocity",
    "contact",
    "layer",
    "colliderType",
    "boundaryElementType",
    "boundarySubsampling",
    "sdf",
}


def _root(path: Path) -> Path:
    for directory in path.parents:
        if (directory / ".superdex_root").is_file():
            return directory
    configured = os.environ.get("SUPERDEX_ASSETS_PATH")
    return Path(configured).expanduser().resolve() if configured else path.parent


def _reference(path: Path, value: str, root: Path) -> Path:
    if value.startswith("//"):
        result = _root(path) / value[2:]
    elif value.startswith("./"):
        result = path.parent / value
    else:
        result = root / value
    result = result.resolve()
    if not result.is_file():
        raise FileNotFoundError(f"superdex prefab {path}: missing dependency {value!r}: {result}")
    return result


# --------------------------------------------------------------------- #
# Cold-path rigid prefab audit
# --------------------------------------------------------------------- #

def audit_prefab(path: Path, root: Path, stack: tuple[Path, ...] = ()) -> int:
    """Count all rigid actors and reject unsupported content in every nested file."""
    path = path.resolve()
    if path in stack:
        raise ValueError(f"superdex cyclic prefab dependency: {path}")
    if path.suffix != ".mochi_prefab":
        raise NotImplementedError("superdex native fragments must be .mochi_prefab files")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"superdex prefab must be a JSON object: {path}")
    unknown = set(data) - {"comment", "actors", "prefabs"}
    if unknown:
        raise NotImplementedError(f"superdex prefab {path}: unsupported fields {sorted(unknown)}")
    actors = data.get("actors", {})
    if set(actors) - {"comment", "rigid"}:
        raise NotImplementedError(f"superdex prefab {path}: only rigid actors are supported")
    count = 0
    for actor in actors.get("rigid", []):
        unknown = set(actor) - _RIGID_FIELDS
        if unknown:
            raise NotImplementedError(
                f"superdex prefab {path}: unsupported rigid fields {sorted(unknown)}"
            )
        if not actor.get("shape"):
            raise ValueError(
                f"superdex prefab {path}: every rigid actor requires a collision shape"
            )
        for key in ("shape", "renderModel"):
            if actor.get(key):
                _reference(path, actor[key], root)
        count += 1
    for nested in data.get("prefabs", []):
        unknown = set(nested) - {"comment", "name", "path", "translation", "rotation", "scale"}
        if unknown:
            raise NotImplementedError(
                f"superdex prefab {path}: unsupported nested fields {unknown}"
            )
        target = _reference(path, nested["path"], root)
        count += audit_prefab(target, root, (*stack, path))
    return count


# --------------------------------------------------------------------- #
# Rigid prefab composition and canonical state layout
# --------------------------------------------------------------------- #

def compose_rigid_prefabs(p: Any, plan: ModelPlan, scene: SceneCfg) -> ModelPlan:
    """Append rigid bodies and free coordinates without changing robot control indices."""
    loaded = []
    expected = 0
    for value in scene.fragment_files:
        path = resolve_scene_fragment_path(value, Path(scene.model_file)).resolve()
        root = _root(path)
        expected += audit_prefab(path, root)
        loaded.append(p.prefab.load_from_file(str(path), str(root)))
    if not expected:
        raise ValueError("superdex prefab fragments contain no rigid actors")

    def spawn(world: Any) -> list[Any]:
        actors = []
        for cfg in loaded:
            result = p.prefab.add_to_scene(cfg, world)
            if len(result.constraints):
                raise ValueError("superdex rigid prefab unexpectedly created constraints")
            actors.extend(result.actors)
        if len(actors) != expected:
            raise ValueError("superdex prefab actor inventory differs from authored inventory")
        if any(a.get_type() != p.ActorType.RIGID for a in actors):
            raise ValueError("superdex prefab created a non-rigid actor")
        return actors

    temp = p.create_scene("superdex_prefab_metadata")
    try:
        actors = spawn(temp)
        append_rigid_metadata(plan, actors)
    finally:
        p.destroy_scene(temp)
    plan.spawn_rigids = spawn
    previous_cleanup = plan.cleanup

    def cleanup() -> None:
        loaded.clear()
        previous_cleanup()

    plan.cleanup = cleanup
    return plan


def append_rigid_metadata(
    plan: ModelPlan, actors: list[Any], *, disambiguate: bool = False
) -> None:
    """Append native rigid actors using the shared canonical object state layout."""
    expected = len(actors)
    names = list(plan.body_names)
    masses, coms = list(plan.body_mass), list(plan.body_ipos)
    qpos = list(plan.default_qpos)
    qvel = list(plan.default_qvel) if plan.default_qvel is not None else [0.0] * plan.nv
    rigids = []
    for actor in actors:
        name = actor.get_name()
        if disambiguate and (not name or name in names):
            base = name or "rigid"
            suffix = len(names)
            name = f"{base}#{suffix}"
            while name in names:
                suffix += 1
                name = f"{base}#{suffix}"
        if name in names:
            raise ValueError(f"superdex duplicate body name {name!r}; name nested instances")
        body = len(names)
        names.append(name)
        pose = actor.get_root_transform()
        quat = np.asarray(pose.rotation)[[3, 0, 1, 2]]
        offset = (
            np.zeros(3)
            if actor.is_static()
            else np.asarray(actor.get_center_of_mass_transform().translation) - pose.translation
        )
        coms.append(rotation_matrix(quat).T @ offset)
        masses.append(0.0 if actor.is_static() else actor.get_mass())
        qi = vi = None
        if not actor.is_static():
            qi, vi = len(qpos), len(qvel)
            qpos.extend([*pose.translation, *quat])
            omega = np.asarray(actor.get_angular_velocity())
            qvel.extend(np.asarray(actor.get_linear_velocity()) - np.cross(omega, offset))
            qvel.extend(rotation_matrix(quat).T @ omega)
        rigids.append(RigidPlan(name, body, qi, vi))
    plan.body_names = tuple(names)
    plan.body_mass = np.asarray(masses)
    plan.body_ipos = np.asarray(coms)
    plan.body_parent_ids = np.concatenate((plan.body_parent_ids, np.zeros(expected, dtype=int)))
    plan.body_link_indices = np.concatenate((plan.body_link_indices, np.full(expected, -1)))
    plan.rigids = tuple(rigids)
    plan.default_qpos = np.asarray(qpos)
    plan.default_qvel = np.asarray(qvel)
    added = len(qvel) - plan.nv
    plan.dof_armature = np.concatenate((plan.dof_armature, np.zeros(added)))
    plan.nq, plan.nv = len(qpos), len(qvel)


# --------------------------------------------------------------------- #
# Native scene audit
# --------------------------------------------------------------------- #

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


def audit_scene(path: Path, root: Path, *, _stack=(), _settings=None, _documents=None):
    """Audit public rigid prefab schemas, including recursively authored scenes."""
    path = path.resolve()
    if path in _stack:
        raise ValueError(f"superdex cyclic prefab dependency: {path}")
    data = json.loads(path.read_text())
    _fields(
        data,
        {"comment", "actors", "scene", "prefabs", "contactFilter", "controllers", "constraints"},
        str(path),
    )
    _finite(data)
    settings = data.get("scene", {})
    _fields(settings, {"comment", "description", "gravity", "solver"}, "settings")
    if _settings is not None:
        for key, value in settings.items():
            if key not in {"comment", "description"} and _settings.get(key) != value:
                raise ValueError(f"superdex scene conflicting nested setting {key!r}: {path}")
    else:
        _settings = settings
    if "gravity" in settings:
        _vector(settings["gravity"], 3, "gravity")
    if "solver" in settings:
        _fields(settings["solver"], _SOLVER_FIELDS, "solver")
        for key, allowed in _SOLVER_CHILDREN.items():
            if key in settings["solver"]:
                _fields(settings["solver"][key], allowed, f"solver.{key}")
        fitted = settings["solver"].get("experimentalEval", {}).get("fittedSaturationHessian")
        if fitted is not None:
            _fields(
                fitted,
                {"constraintSaturation", "contactFriction", "jointFriction"},
                "fittedSaturationHessian",
            )
    actors = data.get("actors", {})
    _fields(actors, {"comment", "articulated", "rigid"}, "actors")
    for articulation in actors.get("articulated", []):
        _fields(
            articulation,
            {
                "comment",
                "name",
                "joints",
                "links",
                "translation",
                "rotation",
                "scale",
                "jointVelocities",
                "cycles",
                "skin",
            },
            "articulation",
        )
        if articulation.get("scale", 1) <= 0:
            raise ValueError("superdex scene articulation scale must be positive")
        _transform({k: articulation[k] for k in ("translation", "rotation") if k in articulation})
        joints, links = articulation.get("joints", []), articulation.get("links", [])
        if not joints or len(joints) != len(links):
            raise ValueError("superdex scene requires one joint per link")
        sizes = {"Hard": 0, "Revolute": 1, "Prismatic": 1, "Spherical": 3, "Free": 6}
        for index, (joint, link) in enumerate(zip(joints, links, strict=True)):
            _fields(joint, _JOINT_FIELDS, "joint")
            _fields(link, _LINK_FIELDS, "link")
            if joint.get("type") not in sizes or (joint["type"] == "Free" and index != 0):
                raise NotImplementedError("superdex scene unsupported tree joint type")
            if joint["type"] in {"Revolute", "Prismatic"}:
                axis = _vector(joint.get("axis"), 3, "joint axis")
                if np.linalg.norm(axis) == 0:
                    raise ValueError("superdex scene joint axis must be nonzero")
            if "friction" in joint:
                _fields(
                    joint["friction"],
                    {"viscous", "coulomb", "falloffVel", "stictionExtra", "stribeckVel"},
                    "joint friction",
                )
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
            names = [record["name"] for record in records if record.get("name")]
            if len(set(names)) != len(names):
                raise ValueError(f"superdex scene authored {label} names must be unique")
        if "jointVelocities" in articulation:
            _vector(
                articulation["jointVelocities"],
                sum(sizes[j["type"]] for j in joints),
                "jointVelocities",
            )
        for cycle in articulation.get("cycles", []):
            _fields(cycle, {"parentLink", "childLink", "jointFromChildLink", "stiffness"}, "cycle")
            for key in ("parentLink", "childLink"):
                if not isinstance(cycle.get(key), int) or not 0 <= cycle[key] < len(links):
                    raise ValueError("superdex cycle requires in-range link indices")
            _transform(cycle.get("jointFromChildLink", {}))
        if articulation.get("skin") is not None:
            skin = articulation["skin"]
            _fields(
                skin,
                {
                    "shape",
                    "renderModel",
                    "renderModelScale",
                    "renderModelRotation",
                    "renderModelTranslation",
                    "layer",
                    "contact",
                    "boundaryElementType",
                    "boundarySubsampling",
                },
                "articulated skin",
            )
            if "contact" in skin:
                _fields(skin["contact"], _CONTACT_FIELDS, "skin contact")
            for key in ("shape", "renderModel"):
                if skin.get(key):
                    _reference(path, skin[key], root)

    for rigid in actors.get("rigid", []):
        _fields(rigid, _RIGID_FIELDS, "rigid actor")
        if "contact" in rigid:
            _fields(rigid["contact"], _CONTACT_FIELDS, "rigid contact")
        if not rigid.get("shape"):
            raise ValueError("superdex scene rigid actors require collision shapes")
        _transform({k: rigid[k] for k in ("translation", "rotation") if k in rigid})
        for key in ("shape", "renderModel"):
            if rigid.get(key):
                _reference(path, rigid[key], root)
    for controller in data.get("controllers", []):
        _fields(
            controller,
            {"comment", "articulatedActor", "jointTracking", "linkPosTracking", "linkRotTracking"},
            "controller",
        )
        if not isinstance(controller.get("articulatedActor"), str):
            raise ValueError("superdex scene controller requires an articulation name")
        for key in ("jointTracking", "linkPosTracking", "linkRotTracking"):
            for item in controller.get(key, []):
                _fields(item, {"stiffness", "damping", "saturation"}, key)
                if any(
                    not isinstance(v, (int, float)) or (v == 0 if k == "saturation" else v < 0)
                    for k, v in item.items()
                ):
                    raise ValueError("superdex scene controller gains must be nonnegative")
    contact_filter = data.get("contactFilter", {})
    categories = {
        f"{kind}Contact{symmetry}": key
        for kind, key in (("actor", "actors"), ("layer", "layers"))
        for symmetry in ("Symmetric", "Asymmetric")
    }
    _fields(contact_filter, {"comment", "_comment", *categories}, "contactFilter")
    for category, key in categories.items():
        for item in contact_filter.get(category, []):
            _fields(item, {"enable", key}, "contactFilter entry")
            if (
                not isinstance(item.get("enable"), bool)
                or len(item.get(key, [])) != 2
                or any(not isinstance(v, str) for v in item[key])
            ):
                raise ValueError("superdex scene contact filter requires two names and enable bool")
    _audit_constraints(data.get("constraints", {}))
    if _documents is not None:
        _documents.append(data)
    for nested in data.get("prefabs", []):
        _fields(
            nested, {"comment", "name", "path", "translation", "rotation", "scale"}, "nested prefab"
        )
        _transform({k: nested[k] for k in ("translation", "rotation") if k in nested})
        target = _reference(path, nested["path"], root)
        if target.suffix not in {".mochi_prefab", ".mochi_scene"}:
            raise NotImplementedError("superdex nested inputs must be rigid prefab/scene files")
        audit_scene(
            target, root, _stack=(*_stack, path), _settings=_settings, _documents=_documents
        )
    return data


def _audit_constraints(constraints):
    common = {"stiffness", "damping", "saturation"}
    types = {
        "rigidPivotPosition": {"actor", "localPosition", "targetPosition"},
        "rigidPivotRotation": {"actor", "localRotation", "targetRotation"},
        "rigidPivotToRigidTarget": {"actor", "localPosition", "targetTransform"},
        "rigidSphericalJoint": {"actorA", "actorB", "localPosA", "localPosB"},
        "rigidPrismaticJoint": {"actorA", "actorB", "freeAxis", "min", "max"},
        "jointRotationRange": {
            "actorA",
            "actorB",
            "angleRangeX",
            "angleRangeY",
            "angleRangeZ",
            "refFrameRotVec",
            "rangeAroundRest",
        },
        "jointRotationTracking": {"actorA", "actorB", "refFrameRotVec"},
        "articulatedSingleDofRange": {"actor", "jointIndex", "dofIndex", "minValue", "maxValue"},
        "articulatedSingleDofTarget": {"actor", "jointIndex", "dofIndex", "targetValue"},
        "articulated3dRotationRange": {"actor", "jointIndex", "minValues", "maxValues"},
        "articulated3dRotationTarget": {"actor", "jointIndex", "target"},
    }
    _fields(constraints, {"comment", *types}, "constraints")
    for kind, allowed in types.items():
        for item in constraints.get(kind, []):
            _fields(item, allowed | common, f"constraint {kind}")


def controlled_indices(data, controlled_joints, effort_limits) -> tuple[np.ndarray, np.ndarray]:
    """Resolve explicit physical effort inputs, never infer controls from asset names."""
    if (
        controlled_joints is None
        or isinstance(controlled_joints, (str, bytes))
        or not len(controlled_joints)
        or any(not isinstance(n, str) for n in controlled_joints)
    ):
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
        return {
            k: value(getattr(item, k))
            for k in dir(item)
            if not k.startswith("_") and not callable(getattr(item, k))
        }

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


# --------------------------------------------------------------------- #
# Native scene materialization
# --------------------------------------------------------------------- #

def materialize_native_scene(
    p, path: Path, scene: SceneCfg, controlled_joints, effort_limits
) -> ModelPlan:
    """Compile public native assets with cached per-actor ownership and layouts."""
    from types import SimpleNamespace

    from unisim.utils.rotation import np_quat_apply_batched, np_quat_mul_batched

    paths = [path, *(resolve_scene_fragment_path(v, path).resolve() for v in scene.fragment_files)]
    documents = []
    data = audit_scene(path, _root(path), _documents=documents)
    for extra in paths[1:]:
        audit_scene(extra, _root(extra), _settings=data.get("scene", {}), _documents=documents)
    counts = {
        kind: sum(len(d.get("actors", {}).get(kind, [])) for d in documents)
        for kind in ("articulated", "rigid")
    }
    constraints = sum(
        len(items)
        for d in documents
        for key, items in d.get("constraints", {}).items()
        if key != "comment"
    )
    loaded = [p.prefab.load_from_file(str(v), str(_root(v))) for v in paths]
    owned = {}

    def instantiate(world):
        actors, actual_constraints = [], []
        for cfg in loaded:
            result = p.prefab.add_to_scene(cfg, world)
            actors.extend(result.actors)
            actual_constraints.extend(result.constraints)
        articulations = [a for a in actors if a.get_type() == p.ActorType.ARTICULATED]
        rigids = [a for a in actors if a.get_type() == p.ActorType.RIGID]
        if (
            len(articulations) != counts["articulated"]
            or len(rigids) != counts["rigid"]
            or len(actors) != len(articulations) + len(rigids)
            or len(actual_constraints) != constraints
        ):
            raise ValueError(
                "superdex scene compiled actor/constraint inventory differs from audit"
            )
        expected_controllers = sum(len(d.get("controllers", [])) for d in documents)
        if sum(a.has_articulated_pose_controller() for a in articulations) != expected_controllers:
            raise ValueError("superdex scene compiled controller inventory differs from audit")
        return articulations, rigids

    def spawn(world):
        actors, rigids = instantiate(world)
        owned[world.get_handle()] = rigids
        instance = actors[0] if len(actors) == 1 else ArticulationGroup(actors)
        return instance, lambda: owned.pop(world.get_handle(), None)

    temp = p.create_scene("superdex_scene_metadata")
    try:
        actors, rigids = instantiate(temp)
        _verify_settings(data.get("scene", {}), native_settings(temp))
        dtype = np.float64 if p.uses_double_precision() else np.float32
        names, parents, link_indices, masses, coms = ["world"], [0], [-1], [0.0], [np.zeros(3)]
        joint_names, joint_q, joint_v, limits, armature = [], [], [], [], []
        kinds = []
        q0, v0, layouts, controller_states = [], [], [], []
        link_start = 0
        groups = {}
        actor_names = [a.get_name() or f"articulation_{i}" for i, a in enumerate(actors)]
        for ai, actor in enumerate(actors):
            info = actor.get_articulated_shape_info()
            links = [temp.get_actor(h) for h in actor.get_nested_link_actors()]
            joints = [
                SimpleNamespace(
                    name=str(name) or f"joint_{i}",
                    type=info.joint_types[i],
                    axis=info.joint_axes[i],
                    min_limit=info.joint_min_limits[i],
                    max_limit=info.joint_max_limits[i],
                )
                for i, name in enumerate(list(info.joint_names)[: len(links)])
            ]
            coord_names, owners, ranges, indices = joint_coordinates(p, joints, actor)
            kinds.extend(coordinate_kinds(p, joints, owners))
            actor_name = actor_names[ai]
            if actor_names.count(actor_name) > 1:
                actor_name = f"{actor_name}#{ai}"
            prefix = f"{actor_name}/" if len(actors) > 1 else ""
            root_reference = None
            if joints[0].type == p.ArticulatedJointType.FREE:
                a, b = info.parent_link_from_joint[0], info.joint_from_child_link[0]
                frame = actor.get_root_transform()
                frame_q = np.asarray(frame.rotation)[[3, 0, 1, 2]]
                root_reference = RootReference(
                    np.asarray(frame.translation)
                    + np_quat_apply_batched(frame_q, np.asarray(a.translation)),
                    np_quat_mul_batched(frame_q, np.asarray(a.rotation)[[3, 0, 1, 2]]),
                    np.asarray(b.translation).copy(),
                    np.asarray(b.rotation)[[3, 0, 1, 2]],
                )
            layout = ArticulationLayout(
                actor_name,
                len(q0),
                len(v0),
                actor.get_num_dofs(),
                link_start,
                len(links),
                len(names),
                root_reference,
            )
            layouts.append(layout)
            native_q, native_v = (
                np.empty(layout.native_size, dtype),
                np.empty(layout.native_size, dtype),
            )
            actor.get_articulated_pose(native_q)
            actor.get_articulated_joint_velocities(native_v)
            q, v = layout.decode(p, native_q, native_v)
            for name, columns in coordinate_groups(coord_names, owners, joints).items():
                groups[prefix + name] = tuple(len(joint_names) + i for i in columns)
            joint_names.extend(prefix + name for name in coord_names)
            joint_q.extend(indices + len(q0) + int(layout.floating))
            joint_v.extend(indices + len(v0))
            limits.extend(ranges)
            q0.extend(q)
            v0.extend(v)
            inertia = list(actor.get_articulated_joint_inertia_params())
            for ji in range(len(links)):
                armature.extend([inertia[ji]] * info.dof_info[ji].get_size())
            for li, link in enumerate(links):
                name = (
                    f"{actor_name}/{info.link_names[li] or f'link_{li}'}"
                    if len(actors) > 1
                    else link.get_name() or f"{actor_name}/link_{li}"
                )
                if name in names:
                    name = f"{name}#{len(names)}"
                names.append(name)
                parent = int(info.parents[li])
                parents.append(0 if parent < 0 else layout.root_body_id + parent)
                link_indices.append(link_start + li)
                masses.append(0.0 if link.is_static() else link.get_mass())
                pose = link.get_root_transform()
                offset = np.asarray(link.get_center_of_mass_transform().translation) - np.asarray(
                    pose.translation
                )
                coms.append(rotation_matrix(np.asarray(pose.rotation)[[3, 0, 1, 2]]).T @ offset)
            link_start += len(links)
            controller_states.append(
                _capture_controller(p, actor, len(joints), len(links), native_q, native_v)
            )
        if controlled_joints is None:
            if joint_names:
                raise ValueError(
                    "superdex scene requires explicit controlled_joints; use [] for passive"
                )
            controlled_joints = []
        if isinstance(controlled_joints, (str, bytes)):
            raise ValueError("superdex controlled_joints must be an ordered list")
        selected = []
        for name in controlled_joints:
            if name not in groups:
                raise ValueError(f"superdex unknown controlled joint/coordinate {name!r}")
            selected.extend(groups[name])
        if len(set(selected)) != len(selected):
            raise ValueError("superdex controlled_joints must not overlap")
        if effort_limits is None and selected:
            raise ValueError("superdex scene requires explicit finite effort_limits")
        ranges = _effort_ranges([] if effort_limits is None else effort_limits, len(selected))
        selected = np.asarray(selected, dtype=int)
        jq, jv = np.asarray(joint_q, dtype=int), np.asarray(joint_v, dtype=int)
        actuator_names = tuple(joint_names[i] for i in selected)
        # Keep the existing one-articulation hinge/slide scene batch profile.
        advanced = (
            path.suffix == ".mochi_prefab"
            or len(actors) != 1
            or constraints > 0
            or not any(a.native_size for a in layouts)
            or not data.get("actors", {}).get("articulated")
            or any(
                a.get("skin") or a.get("cycles")
                for d in documents
                for a in d.get("actors", {}).get("articulated", [])
            )
            or any(
                c.get("linkPosTracking") or c.get("linkRotTracking")
                for d in documents
                for c in d.get("controllers", [])
            )
            or any(a.floating for a in layouts)
            or any(
                j.get("type") == "Spherical"
                for d in documents
                for a in d.get("actors", {}).get("articulated", [])
                for j in a["joints"]
            )
        )
        plan = ModelPlan(
            source_file=str(path),
            nq=len(q0),
            nv=len(v0),
            root_body_id=1 if len(names) > 1 else 0,
            floating=bool(layouts and layouts[0].floating),
            body_names=tuple(names),
            body_parent_ids=np.asarray(parents),
            body_link_indices=np.asarray(link_indices),
            body_mass=np.asarray(masses),
            body_ipos=np.asarray(coms),
            joint_names=tuple(joint_names),
            coordinate_kinds=tuple(kinds),
            joint_qpos_indices=jq,
            joint_qvel_indices=jv,
            joint_ranges=np.asarray(limits).reshape(-1, 2),
            actuator_names=actuator_names,
            actuator_joint_names=actuator_names,
            actuator_qpos_indices=jq[selected],
            actuator_qvel_indices=jv[selected],
            actuator_ctrl_ranges=ranges,
            actuator_force_ranges=ranges.copy(),
            actuator_gear=np.ones(len(selected)),
            actuator_kp=np.zeros(len(selected)),
            actuator_kd=np.zeros(len(selected)),
            default_qpos=np.asarray(q0),
            default_qvel=np.asarray(v0),
            keyframes={},
            gravity=np.asarray(temp.get_gravity()),
            sensors=(),
            spawn_actor=spawn,
            cleanup=loaded.clear,
            dof_armature=np.asarray(armature),
            effective_scene_settings=native_settings(temp),
            articulations=tuple(layouts),
            serial_only=bool(advanced),
            joint_coordinate_groups=groups,
        )
        if any(state is not None for state in controller_states):

            def restore(instance):
                actual = instance.actors if isinstance(instance, ArticulationGroup) else [instance]
                for actor, state in zip(actual, controller_states, strict=True):
                    if state is not None:
                        params, q, v = state
                        actor.set_articulated_pose_controller_params(params)
                        actor.reset_articulated_target_pose(q)
                        actor.set_articulated_target_velocity(v)

            plan.restore_scene_controller = restore
        append_rigid_metadata(plan, rigids, disambiguate=True)
        plan.spawn_rigids = lambda world: owned[world.get_handle()]
        if plan.root_body_id == 0 and rigids:
            plan.root_body_id = plan.rigids[0].body_id
        if len(set(plan.body_names)) != len(plan.body_names):
            raise ValueError("superdex scene body names must be unique; name nested instances")
        return plan
    except BaseException:
        loaded.clear()
        owned.clear()
        raise
    finally:
        p.destroy_scene(temp)


def _capture_controller(p, actor, nj, nl, q, v):
    if not actor.has_articulated_pose_controller():
        return None
    params = p.PoseControllerParams()
    params.joint_tracking.resize(nj)
    params.link_pos_tracking.resize(nl)
    params.link_rot_tracking.resize(nl)
    actor.get_articulated_pose_controller_params(params)
    target = np.empty_like(q)
    actor.get_articulated_target_pose(target)
    return params, target, v.copy()
