"""Cold-path model plans: layouts, root frames, coordinates and public metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from unisim.backend.api_types import BackendArticulationInfo, BackendModelInfo
from unisim.utils.rotation import np_quat_apply_batched as rotate
from unisim.utils.rotation import np_quat_apply_inverse_batched as unrotate
from unisim.utils.rotation import np_quat_conjugate_batched as conjugate
from unisim.utils.rotation import np_quat_mul_batched as multiply

# --------------------------------------------------------------------- #
# Native free-joint reference frames and canonical root-state translation
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class RootReference:
    """For authored frames A and B, the root pose is A * joint_pose * B.

    Native free velocities are expressed in A's axes at the joint origin.
    Canonical velocities are world linear velocity at the root-link origin
    and angular velocity in the root-link frame. Arrays also accept batches.
    """

    parent_pos: np.ndarray
    parent_quat: np.ndarray
    link_pos: np.ndarray
    link_quat: np.ndarray

    def to_world(self, q: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map native xyz/wxyz and joint-frame velocities to canonical state."""
        joint_quat = multiply(self.parent_quat, q[..., 3:7])
        offset = rotate(joint_quat, self.link_pos)
        world_q = np.empty_like(q)
        world_q[..., :3] = self.parent_pos + rotate(self.parent_quat, q[..., :3]) + offset
        world_q[..., 3:7] = multiply(joint_quat, self.link_quat)
        angular = rotate(self.parent_quat, v[..., 3:6])
        world_v = np.empty_like(v)
        world_v[..., :3] = rotate(self.parent_quat, v[..., :3]) + np.cross(angular, offset)
        world_v[..., 3:6] = unrotate(world_q[..., 3:7], angular)
        return world_q, world_v

    def from_world(self, q: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Invert the root-link pose and origin-velocity mapping."""
        joint_quat = multiply(q[..., 3:7], conjugate(self.link_quat))
        offset = rotate(joint_quat, self.link_pos)
        native_q = np.empty_like(q)
        native_q[..., :3] = unrotate(self.parent_quat, q[..., :3] - self.parent_pos - offset)
        native_q[..., 3:7] = multiply(conjugate(self.parent_quat), joint_quat)
        angular = rotate(q[..., 3:7], v[..., 3:6])
        native_v = np.empty_like(v)
        native_v[..., :3] = unrotate(self.parent_quat, v[..., :3] - np.cross(angular, offset))
        native_v[..., 3:6] = unrotate(self.parent_quat, angular)
        return native_q, native_v


# --------------------------------------------------------------------- #
# Native joint layouts, including three-component spherical rotation vectors
# --------------------------------------------------------------------- #

def joint_coordinates(p, joints, actor=None):
    """Return coordinate names, owning joints, limits and native DOF offsets.

    Ball positions are SDK rotation vectors; velocities and efforts use the SDK
    joint frame. Cycle closures constrain existing coordinates and add none.
    Scalar names stay unchanged. ``ball/x``, ``ball/y``, ``ball/z`` identify the
    three ball coordinates, while the bare joint name expands to all three.
    """
    names, owners, limits, offsets = [], [], [], []
    native = list(actor.get_articulated_shape_info().dof_info) if actor is not None else None
    offset = 0
    for i, joint in enumerate(joints):
        kind = joint.type
        size = {
            p.ArticulatedJointType.HARD: 0,
            p.ArticulatedJointType.FREE: 6,
            p.ArticulatedJointType.REVOLUTE: 1,
            p.ArticulatedJointType.PRISMATIC: 1,
            p.ArticulatedJointType.SPHERICAL: 3,
        }.get(kind)
        if size is None:
            raise NotImplementedError(f"superdex unsupported tree joint type: {kind}")
        if native is not None:
            info = native[i]
            if info.get_size() != size or info.offset != offset:
                raise ValueError("superdex compiled native joint layout differs from schema")
        if kind == p.ArticulatedJointType.FREE:
            if i != 0:
                raise NotImplementedError("superdex free joints must be articulation roots")
        elif size:
            axis = np.asarray(joint.axis, dtype=float)
            if size == 1:
                norm = np.linalg.norm(axis)
                if not np.isfinite(norm) or norm == 0:
                    raise ValueError("superdex scalar joint axis must be finite and nonzero")
                axis = axis / norm
            lo = np.full(size, -np.inf)
            hi = np.full(size, np.inf)
            for value, dest in ((joint.min_limit, lo), (joint.max_limit, hi)):
                if value is not None:
                    value = np.asarray(value)
                    if size == 1 and np.isfinite(value).all():
                        dest[:] = np.dot(value, axis)
                    elif size == 1:
                        relevant = value[axis != 0]
                        dest[:] = (
                            relevant[0]
                            if np.all(relevant == relevant[0])
                            else np.dot(relevant, axis[axis != 0])
                        )
                    else:
                        dest[:] = value
            for component in range(size):
                names.append(str(joint.name) if size == 1 else f"{joint.name}/{'xyz'[component]}")
                owners.append(i)
                offsets.append(offset + component)
                limits.append((lo[component], hi[component]))
        offset += size
    if actor is not None and offset != actor.get_num_dofs():
        raise ValueError("superdex compiled native DOF count differs from schema")
    if len(set(names)) != len(names):
        raise ValueError("superdex joint coordinate names must be unique")
    return tuple(names), owners, np.asarray(limits).reshape(-1, 2), np.asarray(offsets, dtype=int)


def coordinate_groups(names, owners=(), joints=()):
    """Map authored joint identities and explicit coordinates, without name inference."""
    groups = {name: (i,) for i, name in enumerate(names)}
    for owner in set(owners):
        columns = tuple(i for i, value in enumerate(owners) if value == owner)
        name = str(joints[owner].name)
        if name in groups and groups[name] != columns:
            raise ValueError("superdex ambiguous joint and coordinate names")
        groups[name] = columns
    return groups


def coordinate_kinds(p, joints, owners):
    """Public coordinate representations from native joint identity, never names."""
    kinds = {
        p.ArticulatedJointType.PRISMATIC: "translation",
        p.ArticulatedJointType.REVOLUTE: "angle",
        p.ArticulatedJointType.SPHERICAL: "rotation_vector_component",
    }
    return tuple(kinds[joints[i].type] for i in owners)


# --------------------------------------------------------------------- #
# Per-articulation ownership and native/canonical coordinate translation
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class ArticulationLayout:
    name: str
    qpos_start: int
    qvel_start: int
    native_size: int
    link_start: int
    link_count: int
    root_body_id: int
    root_reference: RootReference | None = None

    @property
    def floating(self):
        return self.root_reference is not None

    def decode(self, p, native_q, native_v):
        if not self.floating:
            return native_q.copy(), native_v.copy()
        quat = np.asarray(p.Quaternion.from_rotation_vector(native_q[3:6]))[[3, 0, 1, 2]]
        root_q, root_v = self.root_reference.to_world(np.r_[native_q[:3], quat], native_v[:6])
        return np.r_[root_q, native_q[6:]], np.r_[root_v, native_v[6:]]

    def encode(self, p, q, v):
        if not self.floating:
            return q, v
        root_q, root_v = self.root_reference.from_world(q[:7], v[:6])
        rv = np.asarray(p.Quaternion(root_q[[4, 5, 6, 3]]).to_rotation_vector())
        return np.r_[root_q[:3], rv, q[7:]], np.r_[root_v, v[6:]]


class ArticulationGroup:
    """State/force dispatch only. The backend alone advances the owning scene."""

    def __init__(self, actors):
        self.actors = tuple(actors)
        self.slices = []
        offset = 0
        for actor in actors:
            size = actor.get_num_dofs()
            self.slices.append(slice(offset, offset + size))
            offset += size
        self.size = offset

    def get_num_dofs(self):
        return self.size

    def get_nested_link_actors(self):
        return [link for actor in self.actors for link in actor.get_nested_link_actors()]

    def get_articulated_pose(self, out):
        for actor, part in zip(self.actors, self.slices, strict=True):
            actor.get_articulated_pose(out[part])

    def get_articulated_joint_velocities(self, out):
        for actor, part in zip(self.actors, self.slices, strict=True):
            actor.get_articulated_joint_velocities(out[part])

    def set_articulated_pose_from_joints(self, value):
        for actor, part in zip(self.actors, self.slices, strict=True):
            actor.set_articulated_pose_from_joints(value[part])

    def set_articulated_joint_velocities(self, value):
        for actor, part in zip(self.actors, self.slices, strict=True):
            actor.set_articulated_joint_velocities(value[part])

    def set_external_forces_on_dofs(self, indices, forces):
        for actor, part in zip(self.actors, self.slices, strict=True):
            selected = (indices >= part.start) & (indices < part.stop)
            actor.set_external_forces_on_dofs(indices[selected] - part.start, forces[selected])


# --------------------------------------------------------------------- #
# Cold-path authoring plans shared by SuperDex materialization and runtime
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class SensorPlan:
    name: str
    kind: str
    body_id: int = 0
    local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    joint_index: int = -1
    dim: int = 3
    native_link_index: int = -1
    other_actor_name: str | None = None
    contact_distance: float = 0.0


@dataclass
class RigidPlan:
    """One standalone rigid body; static bodies have no state coordinates."""

    name: str
    body_id: int
    qpos_index: int | None
    qvel_index: int | None


@dataclass
class ModelPlan:
    """Canonical arrays and a native scene constructor; no hot-path XML access.

    Body zero is the world. ``body_link_indices`` maps every other canonical
    robot body to the native articulation's nested link actors; rigid prefab
    bodies use -1 and have separate ownership in ``rigids``. Dynamic rigid
    coordinates follow the robot, with seven qpos and six qvel values each.
    Static prefab bodies have no generalized coordinates. Free-root
    qpos is world xyz + wxyz followed by
    single-DoF joints; qvel is world origin velocity + body angular velocity
    followed by single-DoF joints. Native free qvel uses the parent joint's axes;
    ``root_reference`` maps authored offsets (None for identity references).
    """

    source_file: str
    nq: int
    nv: int
    root_body_id: int
    floating: bool
    body_names: tuple[str, ...]
    body_parent_ids: np.ndarray
    body_link_indices: np.ndarray
    body_mass: np.ndarray
    body_ipos: np.ndarray
    joint_names: tuple[str, ...]
    joint_qpos_indices: np.ndarray
    joint_qvel_indices: np.ndarray
    joint_ranges: np.ndarray
    actuator_names: tuple[str, ...]
    actuator_joint_names: tuple[str, ...]
    actuator_qpos_indices: np.ndarray
    actuator_qvel_indices: np.ndarray
    actuator_ctrl_ranges: np.ndarray
    actuator_gear: np.ndarray
    actuator_kp: np.ndarray
    actuator_kd: np.ndarray
    default_qpos: np.ndarray
    keyframes: dict[str, np.ndarray]
    gravity: np.ndarray
    sensors: tuple[SensorPlan, ...]
    # Receives a native scene and returns its articulated actor and an
    # idempotent owner cleanup callback (e.g. RoboticsContext/Bot teardown).
    spawn_actor: Callable[[Any], tuple[Any, Callable[[], None]]]
    cleanup: Callable[[], None]
    actuator_force_ranges: np.ndarray | None = None
    dof_armature: np.ndarray | None = None
    root_reference: RootReference | None = None
    rigids: tuple[RigidPlan, ...] = ()
    spawn_rigids: Callable[[Any], list[Any]] | None = None
    default_qvel: np.ndarray | None = None
    native_bots: dict[Any, tuple[Any, Any, dict[str, Any]]] = field(default_factory=dict)
    camera_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    effective_scene_settings: dict[str, Any] | None = None
    restore_scene_controller: Callable[[Any], None] | None = None
    # Native scene articulation slices (one or more, independent of rigid bodies).
    articulations: tuple[Any, ...] = ()
    serial_only: bool = False
    coordinate_kinds: tuple[str, ...] = ()
    joint_coordinate_groups: dict[str, tuple[int, ...]] = field(default_factory=dict)

    @property
    def robot_nq(self) -> int:
        return self.nq - 7 * sum(item.qpos_index is not None for item in self.rigids)

    @property
    def robot_nv(self) -> int:
        return self.nv - 6 * sum(item.qvel_index is not None for item in self.rigids)


# --------------------------------------------------------------------- #
# Project a private authoring plan onto detached public metadata
# --------------------------------------------------------------------- #

def model_info(plan) -> BackendModelInfo:
    articulations = []
    owners = [None] * len(plan.body_names)
    if plan.articulations:
        for index, layout in enumerate(plan.articulations):
            bodies = tuple(
                i
                for i, link in enumerate(plan.body_link_indices)
                if layout.link_start <= link < layout.link_start + layout.link_count
            )
            articulations.append(
                BackendArticulationInfo(
                    layout.name,
                    bodies,
                    layout.root_body_id,
                    layout.floating,
                    tuple(
                        range(
                            layout.qpos_start,
                            layout.qpos_start + layout.native_size + int(layout.floating),
                        )
                    ),
                    tuple(range(layout.qvel_start, layout.qvel_start + layout.native_size)),
                )
            )
            for i in bodies:
                owners[i] = index
    else:
        bodies = tuple(i for i, link in enumerate(plan.body_link_indices) if link >= 0)
        if bodies:
            articulations.append(
                BackendArticulationInfo(
                    Path(plan.source_file).stem,
                    bodies,
                    plan.root_body_id,
                    plan.floating,
                    tuple(range(plan.robot_nq)),
                    tuple(range(plan.robot_nv)),
                )
            )
            for i in bodies:
                owners[i] = 0
    kinds = plan.coordinate_kinds
    if len(kinds) != len(plan.joint_names):
        raise RuntimeError("superdex coordinate representations missing from model plan")
    groups = plan.joint_coordinate_groups or {name: (i,) for i, name in enumerate(plan.joint_names)}
    return BackendModelInfo(
        plan.nq,
        plan.nv,
        tuple(plan.body_names),
        tuple(int(i) for i in plan.body_parent_ids),
        tuple(owners),
        tuple(articulations),
        tuple(plan.joint_names),
        tuple(int(i) for i in plan.joint_qpos_indices),
        tuple(int(i) for i in plan.joint_qvel_indices),
        tuple(kinds),
        tuple("m" if k == "translation" else "rad" for k in kinds),
        tuple("m/s" if k == "translation" else "rad/s" for k in kinds),
        {k: tuple(v) for k, v in groups.items()},
    )
