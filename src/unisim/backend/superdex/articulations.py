"""Per-articulation ownership and native/canonical coordinate translation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .root_state import RootReference


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
