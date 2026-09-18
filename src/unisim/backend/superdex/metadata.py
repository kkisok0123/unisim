"""Project a private authoring plan onto detached public metadata."""

from __future__ import annotations

from pathlib import Path

from unisim.backend.api_types import BackendArticulationInfo, BackendModelInfo


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
