#!/usr/bin/env python3
"""Run the apple task in Isaac with the shared SuperDex/MuJoCo controller.

Joint targets and PD gains come from the same planner and gain schedule; recorded
DexLab body poses are not control inputs. Run with Isaac Sim's Python 3.11.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCENE = Path(__file__).resolve().with_name("scene_sdf.usd")
FULL_EXPERIENCE = (
    Path.home()
    / ".unilab/isaacsim/venv/lib/python3.11/site-packages/isaacsim/apps/isaacsim.exp.full.kit"
)
DT = 0.002
CONTACT_SAMPLE_DT = 0.05
CONTACT_SAMPLE_STEPS = round(CONTACT_SAMPLE_DT / DT)
MAX_STEM_CONTACTS = 1024
GAIN_SWITCH_TIME = 23.0
SETTLE_MIN_SECONDS = 3.0
SETTLE_WINDOW_SECONDS = 0.5
SETTLE_TIMEOUT_SECONDS = 10.0
SETTLE_POSITION_TOLERANCE = 0.00025
SETTLE_ANGLE_TOLERANCE = math.radians(0.25)
DEFAULT_VIEWER_RATE_HZ = 20.0
CAMERA_EYE = np.asarray([0.85, 0.85, 0.85], dtype=float)
CAMERA_TARGET = np.asarray([0.25, 0.30, 0.42], dtype=float)
ROBOT_ROOT = "/scene_sdf/openarm_body_link0/openarm_body_link0"
APPLE_ROOT = "/scene_sdf/apple_with_stem/apple_with_stem"
APPLE_COLLIDER = f"{APPLE_ROOT}/collisions/apple_with_stem/apple_with_stem"
STEM_PAD_NAMES = ("r_index_finger_pad", "r_thumb_pad")
STEM_PAD_PATHS = tuple(f"/scene_sdf/openarm_body_link0/{name}" for name in STEM_PAD_NAMES)
GRASP_FRICTION = 2.0
GRASP_MATERIAL_PATH = "/IsaacAppleGraspMaterial"
GRASP_FRICTION_PRIMS = (APPLE_ROOT, *STEM_PAD_PATHS)
TABLE_TOP = 0.2995  # Authored table height, also checked by qualify.py.
BAD_PHYSX_MESSAGES = (
    "Invalid PhysX transform",
    "disjointed body transforms",
    "negative mass",
    "non-finite",
    "NaN",
)


def to_numpy(value, *, dtype=float) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value, dtype=dtype)


def to_torch(value, *, integer: bool = False):
    """Convert an Isaac API argument to the backend's required CPU tensor."""
    import torch

    dtype = torch.int32 if integer else torch.float32
    return torch.as_tensor(
        np.asarray(value, dtype=np.int32 if integer else np.float32), dtype=dtype
    )


def app_is_running(app) -> bool:
    return app is None or app.is_running()


def configure_viewer_camera(*, enabled: bool) -> None:
    if not enabled:
        return
    from isaacsim.core.utils.viewports import set_camera_view

    set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)


def wait_while_timeline_paused(*, app, world, viewer_clock, enabled: bool) -> int:
    """Let the user pause the full GUI without advancing control or physics."""
    if app is None or not enabled:
        return 0

    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    waited_updates = 0
    while not timeline.is_playing():
        if not app_is_running(app):
            break
        app.update()
        waited_updates += 1
        time.sleep(0.01)
    if waited_updates and viewer_clock.world is world:
        viewer_clock.reset()
    return waited_updates


class ViewerClock:
    """Service Isaac rendering independently from the 2 ms physics cadence."""

    def __init__(
        self, *, enabled: bool, rate_hz: float, app=None, world=None, pace_realtime: bool = True
    ) -> None:
        self.enabled = enabled
        self.rate_hz = min(max(float(rate_hz), 1.0), 60.0)
        self.app = app
        self.world = world
        self.pace_realtime = pace_realtime
        self.reset()

    def reset(self) -> None:
        self.start = time.monotonic()
        self.next_update = 0.0
        self.maximum_lag_seconds = 0.0

    @property
    def real_time_pacing(self) -> bool | None:
        if not self.enabled:
            return None
        if not self.pace_realtime:
            return False
        return self.maximum_lag_seconds <= max(2.0 / self.rate_hz, DT)

    def service(self, completed_steps: int) -> None:
        if not self.enabled or self.app is None:
            return
        now = time.monotonic()
        if now >= self.next_update:
            if not app_is_running(self.app):
                return
            # World.render synchronizes PhysX transforms with the viewport.
            if self.world is None:
                self.app.update()
            else:
                self.world.render()
            self.next_update = time.monotonic() + 1.0 / self.rate_hz
        if self.pace_realtime:
            deadline = self.start + completed_steps * DT
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                self.maximum_lag_seconds = max(self.maximum_lag_seconds, -remaining)


def build_position_gains(
    joint_names: tuple[str, ...] | list[str], *, body_grasp: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return the unchanged shared gains in per-radian torque units."""
    kp = np.zeros(len(joint_names), dtype=float)
    kv = np.zeros(len(joint_names), dtype=float)
    for index, name in enumerate(joint_names):
        if name.startswith(("l_", "r_")):
            if body_grasp and name.startswith("r_"):
                kp[index], kv[index] = 0.8, 0.01
            else:
                kp[index] = 30.0 if name.startswith(("r_index_finger", "r_thumb")) else 0.8
                kv[index] = 0.02
        elif name.startswith("arm_openarm_"):
            kp[index], kv[index] = 1000.0, 40.0
        else:
            raise ValueError(f"Unsupported robot joint for controller gains: {name}")
    return kp, kv


def isaac_drive_gains(kp: np.ndarray, kv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert runtime gains for authored USD angular-drive attributes only."""
    return kp * math.pi / 180.0, kv * math.pi / 180.0


def shared_targets(
    *,
    apple_pose: np.ndarray,
    stop: float,
    start: float = 0.0,
    joint_positions: np.ndarray | None = None,
    replan: bool = False,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    request = {
        "apple_pose": apple_pose.tolist(),
        "start": start,
        "stop": stop,
        "replan": replan,
    }
    if replan:
        request["joint_positions"] = joint_positions.tolist()
    with tempfile.TemporaryDirectory(prefix="isaac-apple-plan-") as directory:
        request_path = Path(directory) / "request.json"
        output_path = Path(directory) / "targets.npz"
        request_path.write_text(json.dumps(request))
        result = subprocess.run(
            [
                str(ROOT.parents[1] / ".venv/bin/python"),
                str(Path(__file__).resolve().with_name("targets.py")),
                "--request",
                str(request_path),
                "--output",
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Shared target generator failed:\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        with np.load(output_path) as data:
            targets = np.asarray(data["targets"], dtype=float).copy()
            names = tuple(str(name) for name in data["names"])
            pre = np.asarray(data["pre"], dtype=float).copy()
    return targets, names, pre


def angle_between(quaternion: np.ndarray, reference: np.ndarray) -> float:
    quaternion = quaternion / np.linalg.norm(quaternion)
    reference = reference / np.linalg.norm(reference)
    dot = float(np.clip(abs(quaternion @ reference), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def settle_apple(
    world,
    apple,
    *,
    app=None,
    viewer_clock: ViewerClock | None = None,
    ui_pause: bool = False,
) -> tuple[np.ndarray, dict]:
    window_size = int(np.ceil(SETTLE_WINDOW_SECONDS / DT)) + 1
    poses = deque(maxlen=window_size)
    diagnostics: dict = {}
    for step in range(1, int(np.floor(SETTLE_TIMEOUT_SECONDS / DT)) + 1):
        if not app_is_running(app):
            raise RuntimeError("Isaac viewer was closed while settling the apple")
        wait_while_timeline_paused(
            app=app, world=world, viewer_clock=viewer_clock, enabled=ui_pause
        )
        if not app_is_running(app):
            raise RuntimeError("Isaac viewer was closed while settling the apple")
        world.step(render=False)
        if viewer_clock is not None:
            viewer_clock.service(step)
        position, quaternion = apple.get_world_pose()
        position = to_numpy(position)
        quaternion = to_numpy(quaternion)
        linear = to_numpy(apple.get_linear_velocity())
        angular = to_numpy(apple.get_angular_velocity())
        values = (position, quaternion, linear, angular)
        if not all(np.isfinite(value).all() for value in values):
            raise RuntimeError(f"Apple settling became non-finite at {step * DT:.3f}s")
        poses.append(np.r_[position, quaternion])
        if step * DT < SETTLE_MIN_SECONDS or len(poses) < window_size:
            continue
        array = np.asarray(poses)
        position_excursion = float(np.max(np.linalg.norm(array[:, :3] - array[0, :3], axis=1)))
        angle_excursion = max(angle_between(row, array[0, 3:]) for row in array[:, 3:])
        if (
            position_excursion <= SETTLE_POSITION_TOLERANCE
            and angle_excursion <= SETTLE_ANGLE_TOLERANCE
        ):
            diagnostics.update(
                seconds=step * DT,
                minimum_seconds=SETTLE_MIN_SECONDS,
                window_seconds=(window_size - 1) * DT,
                position_tolerance_m=SETTLE_POSITION_TOLERANCE,
                angle_tolerance_rad=SETTLE_ANGLE_TOLERANCE,
                position_excursion_m=position_excursion,
                angle_excursion_rad=angle_excursion,
                final_linear_speed_m_s=float(np.linalg.norm(linear)),
                final_angular_speed_m_s=float(np.linalg.norm(angular)),
            )
            return np.r_[position, quaternion], diagnostics
    raise RuntimeError("Apple did not sustain the shared 0.5-second stability window")


def apply_and_verify_gains(robot, names, *, body_grasp: bool) -> dict:
    kp, kv = build_position_gains(names, body_grasp=body_grasp)
    controller = robot.get_articulation_controller()
    controller.switch_control_mode("position")
    # The live tensor API takes per-radian gains. USD attribute conversion is
    # only needed when authoring gains into the stopped stage.
    controller.set_gains(kps=to_torch(kp), kds=to_torch(kv))
    actual_kp, actual_kv = controller.get_gains()
    actual_kp = to_numpy(actual_kp)
    actual_kv = to_numpy(actual_kv)
    if not (
        np.allclose(actual_kp, kp, rtol=0, atol=1e-5)
        and np.allclose(actual_kv, kv, rtol=0, atol=1e-6)
    ):
        raise RuntimeError(
            "Isaac gain readback differs from the shared schedule: "
            f"kp max error {np.max(np.abs(actual_kp - kp))}, "
            f"kv max error {np.max(np.abs(actual_kv - kv))}"
        )
    return {
        "body_grasp": body_grasp,
        "per_radian_kp": kp.tolist(),
        "per_radian_kv": kv.tolist(),
        "readback_per_radian_kp": actual_kp.tolist(),
        "readback_per_radian_kv": actual_kv.tolist(),
    }


def remove_effort_limits(robot, joint_count: int) -> None:
    """Match the unsaturated SuperDex and MuJoCo position controllers."""
    controller = robot.get_articulation_controller()
    controller.set_max_efforts(np.full(joint_count, np.inf, dtype=np.float32))
    actual = to_numpy(controller.get_max_efforts())
    if actual.shape != (joint_count,) or not np.isposinf(actual).all():
        raise RuntimeError(f"Isaac drive effort limits remain active: {actual}")


def configure_grasp_friction(stage) -> dict:
    """Bind the validated pinch material in the unsaved runtime session layer."""
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    previous_target = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())
    try:
        material = UsdShade.Material.Define(stage, GRASP_MATERIAL_PATH)
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr().Set(GRASP_FRICTION)
        physics.CreateDynamicFrictionAttr().Set(GRASP_FRICTION)
        physics.CreateRestitutionAttr().Set(0.0)
        PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        for path in GRASP_FRICTION_PRIMS:
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise RuntimeError(f"Isaac grasp material target is missing: {path}")
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")
            bound, _ = binding.ComputeBoundMaterial("physics")
            if not bound or bound.GetPath() != material.GetPath():
                raise RuntimeError(f"Isaac grasp material was not bound to {path}")
    finally:
        stage.SetEditTarget(previous_target)
    return {
        "static_friction": GRASP_FRICTION,
        "dynamic_friction": GRASP_FRICTION,
        "material_path": GRASP_MATERIAL_PATH,
        "bound_prims": list(GRASP_FRICTION_PRIMS),
    }


def apple_clearance(vertices: np.ndarray, position: np.ndarray, quaternion: np.ndarray) -> float:
    """Minimum apple-mesh height above the authored table top; quaternion is wxyz."""
    vector = quaternion[1:]
    rotated = vertices + 2.0 * np.cross(
        vector, np.cross(vector, vertices) + quaternion[0] * vertices
    )
    return float(np.min(rotated[:, 2] + position[2]) - TABLE_TOP)


def stem_pad_normal_forces(contact_view, apple_pose, fruit_mesh, stem_mesh) -> dict[str, list[float]]:
    """Sum each pad's normal force on the stem in world coordinates."""
    import trimesh

    data = contact_view.get_contact_force_data(dt=DT)
    if data is None:
        raise RuntimeError("Isaac stem contact view is unavailable")
    magnitudes, points, normals, _, counts, starts = (to_numpy(value) for value in data)
    counts = np.asarray(counts, dtype=int)
    starts = np.asarray(starts, dtype=int)
    if counts.shape != (1, len(STEM_PAD_NAMES)) or starts.shape != counts.shape:
        raise RuntimeError(f"Unexpected Isaac stem contact layout: {counts.shape}, {starts.shape}")
    if int(counts.sum()) >= len(magnitudes):
        raise RuntimeError("Isaac stem contact buffer is full; increase MAX_STEM_CONTACTS")

    position = np.asarray(apple_pose[:3], dtype=float)
    quaternion = np.asarray(apple_pose[3:], dtype=float)
    inverse_vector = -quaternion[1:]
    forces = {}
    for index, name in enumerate(STEM_PAD_NAMES):
        start, count = int(starts[0, index]), int(counts[0, index])
        if start < 0 or count < 0 or start + count > len(magnitudes):
            raise RuntimeError(f"Invalid Isaac contact range for {name}: {start}, {count}")
        force = np.zeros(3, dtype=float)
        if count:
            contact_points = np.asarray(points[start : start + count], dtype=float)
            relative = contact_points - position
            local = relative + 2.0 * np.cross(
                inverse_vector,
                np.cross(inverse_vector, relative) + quaternion[0] * relative,
            )
            _, fruit_distance, _ = trimesh.proximity.closest_point(fruit_mesh, local)
            _, stem_distance, _ = trimesh.proximity.closest_point(stem_mesh, local)
            on_stem = stem_distance + 0.0002 < fruit_distance
            normal_forces = np.asarray(magnitudes[start : start + count], dtype=float).reshape(-1)
            contact_normals = np.asarray(normals[start : start + count], dtype=float)
            if not np.isfinite(normal_forces).all() or not np.isfinite(contact_normals).all():
                raise RuntimeError(f"Non-finite Isaac contact force for {name}")
            force = np.sum(normal_forces[on_stem, None] * contact_normals[on_stem], axis=0)
        forces[name] = force.tolist()
    return forces


def write_stem_contact_csv(path: Path, records: list[dict]) -> None:
    """Write the same per-pad normal-force columns as the MuJoCo runner."""
    fields = ["time_s"]
    for label in ("index", "thumb"):
        fields.extend(f"{label}_normal_force_{axis}_n" for axis in ("x", "y", "z"))
        fields.append(f"{label}_normal_force_magnitude_n")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {"time_s": record["time"] + DT}
            for name, label in zip(STEM_PAD_NAMES, ("index", "thumb"), strict=True):
                force = record["stem_pad_normal_forces_world_n"][name]
                for axis, value in zip(("x", "y", "z"), force, strict=True):
                    row[f"{label}_normal_force_{axis}_n"] = value
                row[f"{label}_normal_force_magnitude_n"] = float(np.linalg.norm(force))
            writer.writerow(row)


def phase_clearance_summary(records: list[dict]) -> tuple[dict, dict]:
    """Apply the same three clearance gates used by the MuJoCo replay."""
    phases = {}
    checks = {}
    for name, (start, end) in {
        "stem_hold": (11.0, 14.0),
        "released": (22.0, 23.0),
        "body_hold": (37.0, 40.0),
    }.items():
        rows = [row for row in records if start <= row["time"] < end]
        if not rows:
            checks[f"{name}_recorded"] = False
            continue
        minimum = min(row["clearance_m"] for row in rows)
        maximum = max(row["clearance_m"] for row in rows)
        phases[name] = {
            "minimum_clearance_m": minimum,
            "maximum_clearance_m": maximum,
        }
        if name == "released":
            checks["released_on_table"] = maximum < 0.001
        else:
            checks[f"{name}_lifted"] = minimum > 0.07
    return phases, checks


def run_task(
    seconds: float,
    *,
    output: Path | None = None,
    app=None,
    viewer: bool = False,
    viewer_rate_hz: float = DEFAULT_VIEWER_RATE_HZ,
    viewer_pace_realtime: bool = True,
    viewer_ui_pause: bool = False,
) -> dict:
    from isaacsim.core.api import World
    from isaacsim.core.api.sensors.rigid_contact_view import RigidContactView
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import UsdGeom

    SimulationManager.set_physics_sim_device("cuda:0")
    world = World(
        physics_dt=DT,
        stage_units_in_meters=1.0,
        set_defaults=False,
        backend="torch",
        device="cuda:0",
    )
    grasp_friction = configure_grasp_friction(world.stage)
    monitor_contacts = output is not None
    if monitor_contacts:
        import trimesh

        mesh_dir = ROOT / "superdex/assets/objects"
        fruit_mesh = trimesh.load_mesh(mesh_dir / "apple-collision.obj", process=False)
        stem_mesh = trimesh.load_mesh(mesh_dir / "stem-collision.obj", process=False)
        contact_view = RigidContactView(
            prim_paths_expr=[APPLE_ROOT],
            filter_paths_expr=list(STEM_PAD_PATHS),
            prepare_contact_sensors=True,
            disable_stablization=False,
            max_contact_count=MAX_STEM_CONTACTS,
        )
    robot = world.scene.add(
        SingleArticulation(ROBOT_ROOT, name="apple_robot", reset_xform_properties=False)
    )
    apple = world.scene.add(SingleRigidPrim(APPLE_ROOT, name="apple", reset_xform_properties=False))
    world.reset()
    if monitor_contacts:
        contact_view.initialize(SimulationManager.get_physics_sim_view())
        if contact_view.num_shapes != 1 or contact_view.num_filters != len(STEM_PAD_NAMES):
            raise RuntimeError("Isaac stem contact view did not resolve the apple and both pads")
    if not math.isclose(float(world.get_physics_dt()), DT, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError(f"Isaac physics timestep is not {DT}: {world.get_physics_dt()}")
    isaac_names = tuple(robot.dof_names)
    if len(isaac_names) != 54:
        raise ValueError(f"Expected 54 Isaac robot DOFs, found {len(isaac_names)}")
    if not np.isfinite(to_numpy(robot.get_joint_positions())).all():
        raise RuntimeError("Initial Isaac robot state is non-finite")
    collider = world.stage.GetPrimAtPath(APPLE_COLLIDER)
    vertices = np.asarray(UsdGeom.Mesh(collider).GetPointsAttr().Get(), dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError("Isaac apple collider has no mesh vertices")

    viewer_clock = ViewerClock(
        enabled=viewer, rate_hz=viewer_rate_hz, app=app, world=world,
        pace_realtime=viewer_pace_realtime,
    )
    apple_pose, settling = settle_apple(
        world,
        apple,
        app=app,
        viewer_clock=viewer_clock,
        ui_pause=viewer and viewer_ui_pause,
    )
    targets, plan_names, plan_pre = shared_targets(apple_pose=apple_pose, stop=seconds)
    permutation = np.asarray([plan_names.index(name) for name in isaac_names], dtype=int)
    targets = targets[:, permutation]
    pre = plan_pre[permutation]
    if len(plan_names) != len(isaac_names) or set(plan_names) != set(isaac_names):
        raise ValueError(
            "Shared planner and Isaac joint sets differ: "
            f"{len(plan_names)} planner joints vs {len(isaac_names)} Isaac joints"
        )
    joint_indices = to_torch(np.arange(len(pre), dtype=np.int32), integer=True)
    robot.set_joint_positions(to_torch(pre), joint_indices=joint_indices)
    robot.set_joint_velocities(to_torch(np.zeros(len(pre))), joint_indices=joint_indices)
    remove_effort_limits(robot, len(isaac_names))
    normal_gain_readback = apply_and_verify_gains(robot, isaac_names, body_grasp=False)

    total_steps = round(seconds / DT)
    if targets.shape != (total_steps, len(isaac_names)):
        raise ValueError(
            f"Target sequence shape {targets.shape} does not match task "
            f"({total_steps}, {len(isaac_names)})"
        )

    records = []
    stem_contact_records = []
    switched_gain_readback = None
    viewer_closed_early = False
    ui_pause_updates = 0
    steps_completed = 0
    max_tracking_error = 0.0
    gain_switch_applied = False
    state_finite = True
    monotonic_time = True
    previous_time = float(world.current_time)
    base_position, base_quaternion = robot.get_world_pose()
    base_position = to_numpy(base_position)
    base_quaternion = to_numpy(base_quaternion)
    max_base_translation = 0.0
    max_base_rotation = 0.0

    # set_joint_positions/velocities operate while the timeline is stopped; get a
    # post-setup baseline because Isaac may reconcile cached physics time there.
    previous_time = float(world.current_time)
    viewer_clock.reset()
    for count in range(total_steps):
        if not app_is_running(app):
            viewer_closed_early = True
            break
        ui_pause_updates += wait_while_timeline_paused(
            app=app, world=world, viewer_clock=viewer_clock, enabled=viewer and viewer_ui_pause
        )
        if not app_is_running(app):
            viewer_closed_early = True
            break
        t = count * DT
        if count == round(GAIN_SWITCH_TIME / DT):
            current_joints = to_numpy(robot.get_joint_positions()).copy()
            position, quaternion = apple.get_world_pose()
            current_apple = np.r_[to_numpy(position), to_numpy(quaternion)]
            remaining, remaining_names, _ = shared_targets(
                apple_pose=current_apple,
                start=GAIN_SWITCH_TIME,
                stop=seconds,
                joint_positions=current_joints[np.argsort(permutation)],
                replan=True,
            )
            if len(remaining_names) != len(isaac_names) or set(remaining_names) != set(isaac_names):
                raise RuntimeError("Shared replan returned a different joint set")
            replan_permutation = np.asarray(
                [remaining_names.index(name) for name in isaac_names], dtype=int
            )
            remaining = remaining[:, replan_permutation]
            expected_remaining = total_steps - round(GAIN_SWITCH_TIME / DT)
            if remaining.shape != (expected_remaining, len(isaac_names)):
                raise RuntimeError(
                    f"Shared replan shape {remaining.shape} does not match "
                    f"({expected_remaining}, {len(isaac_names)})"
                )
            targets[round(GAIN_SWITCH_TIME / DT) :] = remaining
            switched_gain_readback = apply_and_verify_gains(robot, isaac_names, body_grasp=True)
            gain_switch_applied = True

        target = targets[count]
        if target.shape != (len(isaac_names),) or not np.isfinite(target).all():
            raise RuntimeError(f"Invalid shared target at step {count}")
        robot.apply_action(ArticulationAction(joint_positions=to_torch(target)))
        world.step(render=False)
        steps_completed = count + 1
        viewer_clock.service(steps_completed)
        joints = to_numpy(robot.get_joint_positions())
        velocities = to_numpy(robot.get_joint_velocities())
        position, quaternion = apple.get_world_pose()
        apple_pose_now = np.r_[to_numpy(position), to_numpy(quaternion)]
        values = (joints, velocities, apple_pose_now)
        if not all(np.isfinite(value).all() for value in values):
            state_finite = False
            raise RuntimeError(f"Non-finite controller state at {t:.3f}s")
        tracking_error = float(np.max(np.abs(target - joints)))
        max_tracking_error = max(max_tracking_error, tracking_error)
        current_time = float(world.current_time)
        # Isaac may reconcile cached GUI physics time for several control calls
        # at startup, so do not treat those whole-tick catch-ups as controller
        # failures. The meaningful gates are no regression and the configured
        # physics timestep checked at setup.
        time_valid = current_time > previous_time
        monotonic_time &= time_valid
        if not time_valid:
            raise RuntimeError(
                f"Isaac time regressed at task time {t:.3f}s: "
                f"{previous_time:.6f} -> {current_time:.6f}"
            )
        previous_time = current_time
        robot_position, robot_quaternion = robot.get_world_pose()
        robot_position = to_numpy(robot_position)
        robot_quaternion = to_numpy(robot_quaternion)
        if not np.isfinite(robot_position).all() or not np.isfinite(robot_quaternion).all():
            raise RuntimeError(f"Non-finite robot base state at {t:.3f}s")
        max_base_translation = max(
            max_base_translation, float(np.linalg.norm(robot_position - base_position))
        )
        max_base_rotation = max(max_base_rotation, angle_between(robot_quaternion, base_quaternion))
        if count % CONTACT_SAMPLE_STEPS == 0:
            if monitor_contacts:
                stem_contact_records.append(
                    {
                        "time": t,
                        "stem_pad_normal_forces_world_n": stem_pad_normal_forces(
                            contact_view, apple_pose_now, fruit_mesh, stem_mesh
                        ),
                    }
                )
            records.append(
                {
                    "time": t,
                    "max_joint_tracking_error": tracking_error,
                    "apple_position": apple_pose_now[:3].tolist(),
                    "apple_quaternion_wxyz": apple_pose_now[3:].tolist(),
                    "clearance_m": apple_clearance(
                        vertices, apple_pose_now[:3], apple_pose_now[3:]
                    ),
                }
            )
        if count % round(1 / DT) == 0:
            print(
                f"{t:.1f}s tracking={tracking_error:.5f} rad apple_z={apple_pose_now[2]:.4f}",
                flush=True,
            )

    checks = {
        "joint_count_matches_shared_plan": len(isaac_names) == 54,
        "joint_set_matches_shared_plan": set(plan_names) == set(isaac_names),
        "finite_joint_and_apple_state": state_finite,
        "all_targets_finite": bool(np.isfinite(targets).all()),
        "gain_switch_applied_at_23s": seconds <= GAIN_SWITCH_TIME or gain_switch_applied,
        "monotonic_2ms_simulation_time": monotonic_time,
        "requested_steps_completed": steps_completed == total_steps,
        "base_stationary": max_base_translation < 1e-5 and max_base_rotation < 1e-3,
        "apple_above_table": bool(records)
        and min(row["clearance_m"] for row in records) > -0.01,
    }
    if seconds >= 9:
        lift_rows = [row for row in records if 8 <= row["time"] < 9]
        checks["nine_second_apple_lifted"] = bool(lift_rows) and max(
            row["clearance_m"] for row in lift_rows
        ) > 0.01
    complete = steps_completed >= round(40 / DT)
    phases = {}
    if complete:
        phases, phase_checks = phase_clearance_summary(records)
        checks.update(phase_checks)
    summary = {
        "grasp_friction": grasp_friction,
        "normal_gain_readback": normal_gain_readback,
        "max_efforts_unlimited": True,
        "switched_gain_readback": switched_gain_readback,
        "gain_switch_applied": gain_switch_applied,
        "physics": "NVIDIA Isaac Sim 5.1 PhysX",
        "controller": "shared SuperDex/MuJoCo joint-position PD schedule",
        "physics_dt": DT,
        "control_dt": DT,
        "requested_seconds": seconds,
        "complete_sequence": complete,
        "seconds": steps_completed * DT,
        "steps": steps_completed,
        "viewer_enabled": viewer,
        "viewer_rate_hz": viewer_clock.rate_hz if viewer else None,
        "viewer_unpaced": viewer and not viewer_pace_realtime,
        "viewer_ui_pause_enabled": viewer and viewer_ui_pause,
        "viewer_ui_pause_updates": ui_pause_updates if viewer and viewer_ui_pause else 0,
        "viewer_closed_early": viewer_closed_early,
        "real_time_pacing": viewer_clock.real_time_pacing,
        "joint_names": list(isaac_names),
        "stem_contact_monitor": (
            {
                "sample_dt_s": CONTACT_SAMPLE_DT,
                "sample_time_offset_s": DT,
                "csv_file": "stem_contact_forces.csv",
                "method": "Isaac contact-view normal impulse divided by physics timestep",
                "classification": "stem surface at least 0.2 mm closer than fruit surface",
                "force_direction": (
                    "normal force on apple/stem from each right finger pad, world frame"
                ),
            }
            if monitor_contacts else None
        ),
        "settling": settling,
        "maximum_joint_tracking_error_rad": max_tracking_error,
        "maximum_base_translation_m": max_base_translation,
        "maximum_base_rotation_rad": max_base_rotation,
        "checks": checks,
        "phases": phases,
        "records": records,
    }
    summary["passed"] = bool(all(checks.values()))
    if output:
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (output / "metrics.json").write_text(json.dumps(records) + "\n")
        write_stem_contact_csv(output / "stem_contact_forces.csv", stem_contact_records)
    return summary



def launch_checked_worker(args) -> int:
    """Return the task result after Kit's fast shutdown exits the worker."""
    with tempfile.TemporaryDirectory(prefix="isaac-apple-result-") as directory:
        report_path = Path(directory) / "summary.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--_worker",
            "--out",
            str(report_path),
        ]
        result = subprocess.run(command, check=False)
        if not report_path.is_file():
            return result.returncode or 1
        summary = json.loads(report_path.read_text())
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(summary, indent=2) + "\n")
        return 0 if result.returncode == 0 and summary.get("passed") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument(
        "--viewer", action="store_true", help="open a live observational Isaac viewport"
    )
    parser.add_argument(
        "--viewer-rate", type=float, default=DEFAULT_VIEWER_RATE_HZ, help="viewport rate in Hz"
    )
    parser.add_argument(
        "--unpaced", action="store_true",
        help="run physics without real-time sleeping while keeping the viewer",
    )
    parser.add_argument(
        "--viewer-ui-pause", action="store_true",
        help="allow the full-mode viewer pause button to hold the control loop",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not SCENE.is_file():
        parser.error(f"Isaac scene is missing: {SCENE}")
    if not math.isfinite(args.seconds) or not DT <= args.seconds <= 40:
        parser.error("--seconds must be between 0.002 and 40")
    if not math.isfinite(args.viewer_rate) or not 1 <= args.viewer_rate <= 60:
        parser.error("--viewer-rate must be between 1 and 60")
    if not args._worker:
        return launch_checked_worker(args)

    from isaacsim import SimulationApp

    # Full mode exposes the complete Physics menu in the viewer; headless runs stay lightweight.
    app = SimulationApp(
        {"headless": not args.viewer},
        experience=str(FULL_EXPERIENCE) if args.viewer else "",
    )
    error = None
    summary = None
    try:
        import omni.kit.app
        import omni.usd

        if omni.usd.get_context().open_stage(str(SCENE)) is False:
            raise RuntimeError(f"Could not open {SCENE}")
        for _ in range(5):
            app.update()
        configure_viewer_camera(enabled=args.viewer)
        if args.viewer:
            app.update()
        log_stream = omni.kit.app.get_app().get_log_event_stream()
        messages: list[str] = []

        def on_log_event(event) -> None:
            message = str(event.payload)
            if any(fragment.lower() in message.lower() for fragment in BAD_PHYSX_MESSAGES):
                messages.append(message)

        subscription = log_stream.create_subscription_to_pop(
            on_log_event, name="apple Isaac controller parity"
        )
        summary = run_task(
            args.seconds,
            output=args.output,
            app=app,
            viewer=args.viewer,
            viewer_rate_hz=args.viewer_rate,
            viewer_pace_realtime=not args.unpaced,
            viewer_ui_pause=args.viewer_ui_pause,
        )
        log_stream.pump()
        summary["invalid_physx_messages"] = messages
        summary["checks"]["no_invalid_physx_messages"] = not messages
        summary["passed"] = bool(all(summary["checks"].values()))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(summary, indent=2) + "\n")
        if args.output:
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
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
    return 0 if summary and summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
