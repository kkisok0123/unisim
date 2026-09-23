#!/usr/bin/env python3
"""Author and qualify the Isaac apple scene before porting the pickup controller.

Run with Isaac Sim's Python 3.11 interpreter. ``--author`` writes the source
initial pose and explicit qualification profile into the generated USD. A
subsequent run without ``--author`` checks the saved asset. The script never
saves a post-run state: velocities and poses observed while stepping are
runtime values only. The hold drives are a scene-qualification profile, not
the pickup controller.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

SCENE = Path(__file__).resolve().with_name("scene_sdf.usd")
DT = 0.002
STEPS = 1000
EXPECTED_POSITIVE_MASS_LINKS = 79
EXPECTED_SOURCE_CONTACT_EXCLUSIONS = 140
SETTLE_WINDOW = 250
TABLE_TOP = 0.2995
TABLE_ROOT = "/scene_sdf/worldBody/table"
ROBOT_ROOT = "/scene_sdf/openarm_body_link0/openarm_body_link0"
APPLE_ROOT = "/scene_sdf/apple_with_stem/apple_with_stem"
APPLE_COLLIDER = f"{APPLE_ROOT}/collisions/apple_with_stem/apple_with_stem"
ROOT_JOINT = "/scene_sdf/joints/rootJoint_openarm_body_link0"
APPLE_TRANSLATION = np.array([0.3, 0.4, 0.34783494])
VIRTUAL_FRAME_LINKS = frozenset(
    {"openarm_left_base_link", "openarm_right_base_link"}
    | {
        f"{side}_{finger}_tip"
        for side in ("l", "r")
        for finger in ("index_finger", "middle_finger", "pinky", "ring_finger", "thumb")
    }
)
ROBOT_LINK_PREFIX = "/scene_sdf/openarm_body_link0/"
# These two frame-only links are intermediates between the physical root and arm
# base. Their frame joints are identity rotations, so rewiring only translates.
VIRTUAL_FRAME_CHILD_JOINTS = {
    "openarm_left_base_link": "arm_openarm_left_base_link",
    "openarm_right_base_link": "arm_openarm_right_base_link",
}
BAD_PHYSX_MESSAGES = (
    "Invalid PhysX transform",
    "disjointed body transforms",
    "negative mass",
)


def _require_close(actual, expected, label: str, *, atol: float = 1e-6) -> None:
    if not np.allclose(actual, expected, rtol=0, atol=atol):
        raise ValueError(f"{label}: {actual} != {expected}")


def _check_sdf_scene(stage, *, author: bool) -> np.ndarray:
    from pxr import UsdGeom

    apple = stage.GetPrimAtPath(APPLE_ROOT)
    collider = stage.GetPrimAtPath(APPLE_COLLIDER)
    if not apple or "PhysicsRigidBodyAPI" not in apple.GetAppliedSchemas():
        raise ValueError("Isaac scene has no dynamic apple body")
    if stage.GetPrimAtPath("/scene_sdf/joints/rootJoint_apple_with_stem"):
        raise ValueError("Isaac importer fixed the apple to the world")
    if not collider or collider.GetAttribute("physics:approximation").Get() != "sdf":
        raise ValueError("Apple collider is not a PhysX SDF mesh")
    if collider.GetAttribute("physics:collisionEnabled").Get() is not True:
        raise ValueError("Apple SDF collision is disabled")
    if "PhysxSDFMeshCollisionAPI" not in str(collider.GetMetadata("apiSchemas")):
        raise ValueError("Apple SDF cooking API is absent")
    if collider.GetAttribute("physxSDFMeshCollision:sdfResolution").Get() != 524:
        raise ValueError("Apple SDF resolution is not 524")
    scene = stage.GetPrimAtPath("/physicsScene")
    if not scene:
        raise ValueError("Isaac scene has no PhysicsScene")
    if author:
        scene.GetAttribute("physxScene:enableGPUDynamics").Set(True)
        scene.GetAttribute("physxScene:broadphaseType").Set("GPU")
        # Enabling the source selective self-collisions raises aggregate-pair load
        # beyond Isaac's 1024 default; allocate deterministically above 2648.
        scene.GetAttribute("physxScene:gpuFoundLostAggregatePairsCapacity").Set(8192)
        scene.GetAttribute("physxScene:gpuTotalAggregatePairsCapacity").Set(8192)
    if (
        scene.GetAttribute("physxScene:enableGPUDynamics").Get() is not True
        or scene.GetAttribute("physxScene:broadphaseType").Get() != "GPU"
    ):
        raise ValueError("PhysX GPU dynamics and broadphase are required")
    if (
        scene.GetAttribute("physxScene:gpuFoundLostAggregatePairsCapacity").Get() < 2648
        or scene.GetAttribute("physxScene:gpuTotalAggregatePairsCapacity").Get() < 2648
    ):
        raise ValueError("GPU aggregate-pair capacity is too small for selective self-collision")
    points = UsdGeom.Mesh(collider).GetPointsAttr().Get()
    vertices = np.asarray(points, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Apple collider has no triangle mesh vertices")
    spacing = float(np.ptp(vertices, axis=0).max()) / 524
    if spacing > 0.0002:
        raise ValueError(f"Apple SDF spacing is too coarse: {spacing} m")
    return vertices


def _author_or_check_initial_state(stage, *, author: bool) -> None:
    from pxr import Gf

    apple = stage.GetPrimAtPath(APPLE_ROOT)
    if not apple:
        raise ValueError("Apple root is missing")
    translate = apple.GetAttribute("xformOp:translate")
    orient = apple.GetAttribute("xformOp:orient")
    if author:
        translate.Set(Gf.Vec3d(*APPLE_TRANSLATION))
        orient.Set(Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0)))
    else:
        _require_close(translate.Get(), APPLE_TRANSLATION, "authored apple translation", atol=1e-12)
        actual_rotation = orient.Get()
        _require_close(
            [actual_rotation.GetReal(), *actual_rotation.GetImaginary()],
            [1.0, 0.0, 0.0, 0.0],
            "authored apple rotation",
        )

    dynamic_prims = []
    for prim in stage.Traverse():
        if "PhysicsRigidBodyAPI" not in prim.GetAppliedSchemas():
            continue
        path = str(prim.GetPath())
        if path.startswith("/scene_sdf/openarm_body_link0/") or path == APPLE_ROOT:
            dynamic_prims.append(prim)
    if author:
        for prim in dynamic_prims:
            prim.GetAttribute("physics:velocity").Set(Gf.Vec3f(0.0, 0.0, 0.0))
            prim.GetAttribute("physics:angularVelocity").Set(Gf.Vec3f(0.0, 0.0, 0.0))
        stage.GetPrimAtPath(ROBOT_ROOT).GetAttribute("physxArticulation:enabledSelfCollisions").Set(
            False
        )
        table = stage.GetPrimAtPath(TABLE_ROOT)
        table.GetAttribute("physics:kinematicEnabled").Set(False)
        table.GetAttribute("physics:rigidBodyEnabled").Set(False)
    else:
        for prim in dynamic_prims:
            _require_close(
                prim.GetAttribute("physics:velocity").Get(),
                [0.0] * 3,
                f"{prim.GetName()} velocity",
            )
            _require_close(
                prim.GetAttribute("physics:angularVelocity").Get(),
                [0.0] * 3,
                f"{prim.GetName()} angular velocity",
            )
        if (
            stage.GetPrimAtPath(ROBOT_ROOT)
            .GetAttribute("physxArticulation:enabledSelfCollisions")
            .Get()
            is not False
        ):
            raise ValueError(
                "Source selective robot self-collision is disabled; "
                "enable it only after resolving the imported-articulation instability"
            )
        table = stage.GetPrimAtPath(TABLE_ROOT)
        if table.GetAttribute("physics:kinematicEnabled").Get() is not False:
            raise ValueError("Table is kinematic rather than static")
        if table.GetAttribute("physics:rigidBodyEnabled").Get() is not False:
            raise ValueError("Table rigid-body simulation is enabled")

    pairs = set()
    for prim in stage.Traverse():
        if "PhysicsFilteredPairsAPI" not in prim.GetAppliedSchemas():
            continue
        relationship = prim.GetRelationship("physics:filteredPairs")
        if relationship:
            pairs.update(
                tuple(sorted((str(prim.GetPath()), str(target))))
                for target in relationship.GetTargets()
            )
    if len(pairs) != 140:
        raise ValueError(f"Expected 140 source contact exclusions, found {len(pairs)}")


def _author_or_check_robot(stage, *, author: bool) -> None:
    from pxr import Gf, UsdGeom

    base = stage.GetPrimAtPath(ROBOT_ROOT)
    joint = stage.GetPrimAtPath(ROOT_JOINT)
    if not base or not joint:
        raise ValueError("Imported robot root or world fixed joint is missing")
    if [str(path) for path in joint.GetRelationship("physics:body1").GetTargets()] != [ROBOT_ROOT]:
        raise ValueError("World fixed joint does not target the robot root")

    world_from_base = UsdGeom.XformCache().GetLocalToWorldTransform(base)
    position = world_from_base.ExtractTranslation()
    rotation = world_from_base.ExtractRotationQuat()
    position = Gf.Vec3f(*position)
    rotation = Gf.Quatf(float(rotation.GetReal()), Gf.Vec3f(*rotation.GetImaginary()))
    position_attr = joint.GetAttribute("physics:localPos0")
    rotation_attr = joint.GetAttribute("physics:localRot0")
    if author:
        position_attr.Set(position)
        rotation_attr.Set(rotation)
    else:
        _require_close(position_attr.Get(), position, "root fixed-joint position")
        actual_rotation = rotation_attr.Get()
        _require_close(
            [actual_rotation.GetReal(), *actual_rotation.GetImaginary()],
            [rotation.GetReal(), *rotation.GetImaginary()],
            "root fixed-joint rotation",
        )

    if author:
        frame_transforms = {}
        for frame, child in VIRTUAL_FRAME_CHILD_JOINTS.items():
            frame_joint = stage.GetPrimAtPath(f"/scene_sdf/joints/{frame}")
            frame_transforms[frame] = (
                frame_joint.GetAttribute("physics:localPos0").Get(),
                frame_joint.GetAttribute("physics:localRot0").Get(),
            )
        for path in [
            *(f"{ROBOT_LINK_PREFIX}{name}" for name in sorted(VIRTUAL_FRAME_LINKS)),
            *(f"/scene_sdf/joints/{name}" for name in sorted(VIRTUAL_FRAME_LINKS)),
        ]:
            stage.GetPrimAtPath(path).SetActive(False)
        for frame, child in VIRTUAL_FRAME_CHILD_JOINTS.items():
            position, rotation = frame_transforms[frame]
            child_joint = stage.GetPrimAtPath(f"/scene_sdf/joints/{child}")
            relationship = child_joint.GetRelationship("physics:body0")
            relationship.ClearTargets(True)
            relationship.SetTargets([ROBOT_ROOT])
            child_joint.GetAttribute("physics:localPos0").Set(position)
            child_joint.GetAttribute("physics:localRot0").Set(rotation)

    for frame, child in VIRTUAL_FRAME_CHILD_JOINTS.items():
        if stage.GetPrimAtPath(f"/scene_sdf/joints/{frame}").IsActive():
            raise ValueError(f"Frame joint remains active: {frame}")
        child_joint = stage.GetPrimAtPath(f"/scene_sdf/joints/{child}")
        if not child_joint:
            raise ValueError(f"Arm-base joint is missing: {child}")
        body0 = [str(path) for path in child_joint.GetRelationship("physics:body0").GetTargets()]
        if body0 != [ROBOT_ROOT]:
            raise ValueError(f"{child} does not attach to the physical root: {body0}")
        expected_body1 = f"{ROBOT_LINK_PREFIX}{child}"
        body1 = [str(path) for path in child_joint.GetRelationship("physics:body1").GetTargets()]
        if body1 != [expected_body1]:
            raise ValueError(f"{child} has an unexpected child body: {body1}")

    remaining_frames = sorted(
        str(prim.GetPath()).removeprefix(ROBOT_LINK_PREFIX)
        for name in VIRTUAL_FRAME_LINKS
        if (prim := stage.GetPrimAtPath(f"{ROBOT_LINK_PREFIX}{name}")).IsValid() and prim.IsActive()
    )
    if remaining_frames:
        raise ValueError(f"Virtual frame links remain active: {remaining_frames}")
    positive_mass = [
        prim.GetPath().pathString.removeprefix(ROBOT_LINK_PREFIX)
        for prim in stage.Traverse()
        if "PhysicsRigidBodyAPI" in prim.GetAppliedSchemas()
        and str(prim.GetPath()).startswith(ROBOT_LINK_PREFIX)
        and (prim.GetAttribute("physics:mass").Get() or 0) > 0
    ]
    if len(positive_mass) != 79:
        raise ValueError(
            f"Expected {EXPECTED_POSITIVE_MASS_LINKS} positive-mass links, "
            f"found {len(positive_mass)}"
        )

    drives = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsRevoluteJoint":
            continue
        name = prim.GetName()
        if name.startswith("arm_"):
            kp, kd, effort = 1000.0, 40.0, 100.0
        elif name.startswith(("l_", "r_")):
            kp, kd, effort = 30.0, 2.0, 5.0
        else:
            raise ValueError(f"Unclassified Isaac robot joint: {name}")
        # USD angular drive gains are per degree; these targets are per radian.
        values = {
            "drive:angular:physics:stiffness": kp * math.pi / 180.0,
            "drive:angular:physics:damping": kd * math.pi / 180.0,
            "drive:angular:physics:maxForce": effort,
            "drive:angular:physics:targetPosition": 0.0,
        }
        for key, expected in values.items():
            attr = prim.GetAttribute(key)
            if not attr:
                raise ValueError(f"{name} lacks {key}")
            if author:
                attr.Set(expected)
            else:
                _require_close(attr.Get(), expected, f"{name} {key}")
        drives += 1
    if drives != 54:
        raise ValueError(f"Expected 54 hinge drives, found {drives}")


def _to_numpy(value, *, dtype=float) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def _angle_between(quaternions: np.ndarray, reference: np.ndarray) -> np.ndarray:
    unit = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
    reference = reference / np.linalg.norm(reference)
    dot = np.clip(np.abs(unit @ reference), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def _apple_clearance(vertices: np.ndarray, position: np.ndarray, quat: np.ndarray) -> float:
    vector = quat[1:]
    rotated = vertices + 2.0 * np.cross(vector, np.cross(vector, vertices) + quat[0] * vertices)
    return float(np.min(rotated[:, 2] + position[2]) - TABLE_TOP)


def _qualify(stage, vertices: np.ndarray, log_stream, physics_messages: list[str]) -> dict:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.simulation_manager import SimulationManager

    SimulationManager.set_physics_sim_device("cuda:0")
    world = World(
        physics_dt=DT,
        stage_units_in_meters=1.0,
        set_defaults=False,
        backend="torch",
        device="cuda:0",
    )
    robot = world.scene.add(
        SingleArticulation(ROBOT_ROOT, name="qualification_robot", reset_xform_properties=False)
    )
    apple = world.scene.add(
        SingleRigidPrim(APPLE_ROOT, name="qualification_apple", reset_xform_properties=False)
    )
    world.reset()
    initial_joints = _to_numpy(robot.get_joint_positions(), dtype=float).copy()
    base_position, base_quat = robot.get_world_pose()
    base_position = _to_numpy(base_position, dtype=float)
    base_quat = _to_numpy(base_quat, dtype=float)
    if len(initial_joints) != 54:
        raise ValueError(f"Expected 54 joints, found {len(initial_joints)}")
    if np.max(np.abs(initial_joints)) > 1e-4:
        raise ValueError("The zero-position hold profile does not match the initial joint state")

    max_tracking = 0.0
    max_base_move = 0.0
    max_base_angle = 0.0
    apple_window_pos = []
    apple_window_quat = []
    for step in range(STEPS):
        world.step(render=False)
        joints = _to_numpy(robot.get_joint_positions(), dtype=float)
        velocities = _to_numpy(robot.get_joint_velocities(), dtype=float)
        robot_pos, robot_quat = robot.get_world_pose()
        apple_pos, apple_quat = apple.get_world_pose()
        robot_pos = _to_numpy(robot_pos, dtype=float)
        robot_quat = _to_numpy(robot_quat, dtype=float)
        apple_pos = _to_numpy(apple_pos, dtype=float)
        apple_quat = _to_numpy(apple_quat, dtype=float)
        apple_linear = _to_numpy(apple.get_linear_velocity(), dtype=float)
        apple_angular = _to_numpy(apple.get_angular_velocity(), dtype=float)
        finite_values = {
            "joints": joints,
            "velocities": velocities,
            "robot_pos": robot_pos,
            "robot_quat": robot_quat,
            "apple_pos": apple_pos,
            "apple_quat": apple_quat,
            "apple_linear": apple_linear,
            "apple_angular": apple_angular,
        }
        bad = [name for name, value in finite_values.items() if not np.isfinite(value).all()]
        if bad:
            diagnostics = {name: finite_values[name].tolist() for name in bad}
            raise RuntimeError(f"Non-finite physics state at step {step + 1}: {bad}={diagnostics}")
        max_tracking = max(max_tracking, float(np.max(np.abs(joints - initial_joints))))
        max_base_move = max(max_base_move, float(np.linalg.norm(robot_pos - base_position)))
        max_base_angle = max(max_base_angle, float(_angle_between(robot_quat[None], base_quat)[0]))
        if step >= STEPS - SETTLE_WINDOW:
            apple_window_pos.append(apple_pos)
            apple_window_quat.append(apple_quat)

    log_stream.pump()
    positions = _to_numpy(apple_window_pos)
    quaternions = _to_numpy(apple_window_quat)
    position_excursion = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
    angle_excursion = float(np.max(_angle_between(quaternions, quaternions[0])))
    clearance = _apple_clearance(vertices, apple_pos, apple_quat)
    link_transforms = _to_numpy(robot._articulation_view._physics_view.get_link_transforms())
    checks = {
        "all_robot_links_finite": bool(np.isfinite(link_transforms).all()),
        "robot_tracking": max_tracking < 0.01,
        "base_stationary": max_base_move < 1e-5 and max_base_angle < 1e-3,
        "apple_settled": position_excursion < 0.00025
        and angle_excursion < math.radians(0.25)
        and float(np.linalg.norm(apple_linear)) < 0.01
        and float(np.linalg.norm(apple_angular)) < 0.05,
        "apple_table_supported": -0.003 <= clearance <= 0.003,
        "no_invalid_physx_warnings": not physics_messages,
        "requested_steps_completed": world.current_time >= STEPS * DT,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "dt_s": DT,
        "steps": STEPS,
        "time_s": world.current_time,
        "max_joint_tracking_error_rad": max_tracking,
        "max_base_translation_m": max_base_move,
        "max_base_rotation_rad": max_base_angle,
        "apple_last_half_second_position_excursion_m": position_excursion,
        "apple_last_half_second_rotation_rad": angle_excursion,
        "apple_final_clearance_m": clearance,
        "apple_final_linear_speed_m_s": float(np.linalg.norm(apple_linear)),
        "apple_final_angular_speed_rad_s": float(np.linalg.norm(apple_angular)),
        "invalid_physx_messages": physics_messages,
        "authored_initial_state": True,
        "source_selective_self_collision_enabled": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--author",
        action="store_true",
        help="write the source initial state and qualification profile after a passing run",
    )
    args = parser.parse_args()
    if not SCENE.is_file():
        parser.error(f"Isaac SDF scene is missing: {SCENE}")

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    error = None
    try:
        import omni.kit.app
        import omni.usd

        if omni.usd.get_context().open_stage(str(SCENE)) is False:
            raise RuntimeError(f"Could not open {SCENE}")
        for _ in range(5):
            app.update()
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetRootLayer() if args.author else stage.GetSessionLayer())
        vertices = _check_sdf_scene(stage, author=args.author)
        _author_or_check_initial_state(stage, author=args.author)
        _author_or_check_robot(stage, author=args.author)
        # Let composition and Fabric observe deactivated frame topology before
        # PhysicsWorld creation; otherwise PhysX can miss the rewritten body0.
        for _ in range(2):
            app.update()

        physics_messages: list[str] = []
        log_stream = omni.kit.app.get_app().get_log_event_stream()

        def on_log_event(event) -> None:
            message = str(event.payload)
            if any(fragment in message for fragment in BAD_PHYSX_MESSAGES):
                physics_messages.append(message)

        subscription = log_stream.create_subscription_to_pop(
            on_log_event, name="apple Isaac qualification"
        )
        if args.author:
            # Save only the authored initial definition. Stepping may write live
            # pose/velocity values into the stage; never persist that snapshot.
            stage.GetRootLayer().Save()
        report = _qualify(stage, vertices, log_stream, physics_messages)
        if not report["passed"]:
            raise RuntimeError(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        del subscription
    except BaseException as exc:
        error = exc
        traceback.print_exc()
    try:
        app.close()
    except SystemExit:
        pass
    if error is not None:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
