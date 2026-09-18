"""Native joint layouts, including three-component spherical rotation vectors."""

from __future__ import annotations

import numpy as np


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
