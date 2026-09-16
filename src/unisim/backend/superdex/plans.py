"""Cold-path authoring plans shared by SuperDex materialization and runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from unisim.backend.superdex.root_state import RootReference


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

    @property
    def robot_nq(self) -> int:
        return self.nq - 7 * sum(item.qpos_index is not None for item in self.rigids)

    @property
    def robot_nv(self) -> int:
        return self.nv - 6 * sum(item.qvel_index is not None for item in self.rigids)
