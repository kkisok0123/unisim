"""SDK-independent model descriptions and controller inputs.

All positions use metres, angles use radians, and pose arrays use xyz + wxyz.
These types contain data only; adapters validate model-dependent sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class JointTarget:
    """Joint positions in ``get_dof_pos()`` order, excluding floating roots."""

    positions: np.ndarray | tuple[float, ...] | list[float]


@dataclass(frozen=True)
class CartesianTarget:
    """End-effector xyz + normalized wxyz relative to the configured root link."""

    pose: np.ndarray | tuple[float, ...] | list[float]


@dataclass(frozen=True)
class ArticulationPoseTarget:
    """World root pose and exactly one joint or parent-relative link pose array.

    Joint positions follow ``get_dof_pos()`` order. Link poses have shape
    (link_count, 7), including the root, in the controller description's order.
    """

    root_pose: np.ndarray | tuple[float, ...] | list[float]
    joint_positions: np.ndarray | tuple[float, ...] | list[float] | None = None
    link_poses: np.ndarray | list[list[float]] | None = None


ControllerTarget = JointTarget | CartesianTarget | ArticulationPoseTarget


@dataclass(frozen=True)
class BackendArticulationInfo:
    """One articulation's ownership in the complete public state and body tables."""

    name: str
    body_ids: tuple[int, ...]
    root_body_id: int
    floating: bool
    qpos_indices: tuple[int, ...]
    qvel_indices: tuple[int, ...]


@dataclass(frozen=True)
class BackendModelInfo:
    """Detached model metadata, with no native handles or mutable engine state.

    Coordinate tables follow get_dof_pos/get_dof_vel order; groups map authored
    joint names and explicit coordinate names to columns in those tables.
    Spherical positions are rotation-vector components; their velocities are
    joint-frame angular velocities, not rotation-vector time derivatives.
    A body ownership entry is an articulation index or None for world/rigids.
    Floating roots use world xyz + wxyz positions, world-origin linear velocity
    and body-frame angular velocity; query get_root_state_layout for their slots.
    """

    nq: int
    nv: int
    body_names: tuple[str, ...]
    body_parent_ids: tuple[int, ...]
    body_articulation_ids: tuple[int | None, ...]
    articulations: tuple[BackendArticulationInfo, ...]
    coordinate_names: tuple[str, ...]
    coordinate_qpos_indices: tuple[int, ...]
    coordinate_qvel_indices: tuple[int, ...]
    coordinate_representations: tuple[str, ...]
    coordinate_position_units: tuple[str, ...]
    coordinate_velocity_units: tuple[str, ...]
    joint_coordinate_groups: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class BackendControllerInfo:
    """Runtime-defined controller identity and the shared target it accepts.

    Availability applies to this loaded profile, not every model in an engine.
    Authored configuration files/parameters remain runtime-specific.
    """

    identifier: str
    target_kind: Literal["joint", "cartesian", "articulation_pose"]
    available: bool
    unavailable_reason: str | None
    coordinate_names: tuple[str, ...]
    link_names: tuple[str, ...]
