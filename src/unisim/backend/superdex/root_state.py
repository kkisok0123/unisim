"""Native free-joint reference frames and canonical root-state translation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unisim.utils.rotation import np_quat_apply_batched as rotate
from unisim.utils.rotation import np_quat_apply_inverse_batched as unrotate
from unisim.utils.rotation import np_quat_conjugate_batched as conjugate
from unisim.utils.rotation import np_quat_mul_batched as multiply


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
