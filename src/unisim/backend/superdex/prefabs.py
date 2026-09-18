"""Cold-path rigid prefab audit, materialization and canonical state layout."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from unisim.scene import SceneCfg, resolve_scene_fragment_path

from .geometry import rotation_matrix
from .plans import ModelPlan, RigidPlan

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
