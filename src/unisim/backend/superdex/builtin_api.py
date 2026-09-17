"""Explicit SuperDex-only camera and controller APIs; no eager SDK imports."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

CONTROLLERS = {
    "BASIC_JSC_PD": "ControllerBasicJscPdTarget",
    "BASIC_OSC_PD": "ControllerBasicOscPdTarget",
    "MOCHI_ARTICULATED_POSE": "ControllerMochiArticulatedPoseTarget",
}
POSE_CONTROLLER = "MOCHI_ARTICULATED_POSE"


def _finite(value: Any, shape: tuple[int, ...], label: str) -> None:
    array = np.asarray(value)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite with shape {shape}")


def _transform(value: Any) -> None:
    _finite(value.translation, (3,), "target translation")
    _finite(value.rotation, (4,), "target quaternion")
    if not np.isclose(np.linalg.norm(value.rotation), 1, atol=1e-5):
        raise ValueError("target quaternion must be normalized (SDK xyzw)")


class BuiltinAPI:
    """Implemented by SuperDexBackend, separate from the shared SimBackend contract."""

    def get_camera_names(self) -> tuple[str, ...]:
        self._check_open()
        return tuple(self.model.camera_params)

    def get_camera_parameters(self, name: str) -> dict[str, Any]:
        self._check_open()
        return copy.deepcopy(self.model.camera_params[name])

    def get_camera_poses(self, name: str, env_ids=None) -> np.ndarray:
        """World position + wxyz orientation; SDK mount/optical settings stay separate."""
        self._check_open()
        if name not in self.model.camera_params:
            raise KeyError(name)
        ids = self._env_ids if env_ids is None else self._ids(env_ids)
        result = np.empty((len(ids), 7))
        for row, i in enumerate(ids):
            _, _, cameras = self.model.native_bots[self._actors[i].get_handle()]
            pose = cameras[name].get_world_transform()
            result[row, :3] = pose.translation
            result[row, 3:] = np.asarray(pose.rotation)[[3, 0, 1, 2]]
        return result

    def set_pre_step_control(self, fn) -> None:
        if fn is not None and self._controllers:
            raise RuntimeError("clear_controller() before installing a pre-step callback")
        super().set_pre_step_control(fn)

    def configure_controller(self, type_name: str, *, param_args="", init_args="") -> None:
        """Configure one native controller per environment using SDK JSON/file arguments."""
        self._check_open()
        if self.model.restore_scene_controller is not None:
            raise NotImplementedError("superdex cannot replace an authored scene controller")
        if type_name not in CONTROLLERS:
            raise NotImplementedError(f"superdex unsupported controller {type_name!r}")
        if self._controllers:
            raise RuntimeError("clear_controller() before configuring another controller")
        if self._pre_step_control_fn is not None:
            raise RuntimeError("remove the pre-step callback before configuring a controller")
        if not self.model.native_bots:
            raise NotImplementedError(
                "superdex built-in controllers require a native .superdex_bot"
            )
        if not isinstance(param_args, str) or not isinstance(init_args, str):
            raise TypeError("controller param_args/init_args must be SDK JSON or file strings")
        for value in (param_args, init_args):
            if value.strip():
                data = json.loads(
                    value if value.lstrip().startswith("{") else Path(value).read_text()
                )

                def validate(item):
                    if isinstance(item, dict):
                        for child in item.values():
                            validate(child)
                    elif isinstance(item, list):
                        for child in item:
                            validate(child)
                    elif isinstance(item, (int, float)):
                        if not np.isfinite(item) or abs(item) > np.finfo(self._dtype).max:
                            raise ValueError(
                                "controller parameters must be finite in SDK precision"
                            )

                validate(data)
        self._controller_type = type_name
        try:
            for actor in self._actors:
                context, bot, _ = self.model.native_bots[actor.get_handle()]
                before = set(bot.get_controller_handles())
                controller = bot.create_controller(type_name)
                handles = set(bot.get_controller_handles()) - before
                if len(handles) != 1:
                    raise RuntimeError("superdex controller inventory differs after creation")
                self._controllers.append((context, handles.pop(), controller))
                controller.configure_from_scene_entry(param_args, init_args)
        except BaseException as exc:
            self._clear_controllers()
            if type_name == "BASIC_OSC_PD" and "bot-space vs actor-space" in str(exc):
                raise NotImplementedError(
                    "SuperDex SDK OSC cannot initialize this floating-base bot: "
                    "bot-space vs actor-space effort-limit indexing mismatch"
                ) from exc
            raise

    def clear_controller(self) -> None:
        self._check_open()
        self._clear_controllers()

    def _clear_controllers(self) -> None:
        # Remove solver-side state before releasing its robotics wrapper.
        while self._controllers:
            i = len(self._controllers) - 1
            context, handle, _ = self._controllers[i]
            actor = self._actors[i]
            if self._controller_type == POSE_CONTROLLER and actor.has_articulated_pose_controller():
                actor.remove_articulated_pose_controller()
            context.destroy_controller(handle)
            self._controllers.pop()
        self._controller_type = None

    def _suspend_controller(self, i: int) -> None:
        if self._controllers:
            self._controllers[i][2].reset()
            actor = self._actors[i]
            if self._controller_type == POSE_CONTROLLER and actor.has_articulated_pose_controller():
                actor.remove_articulated_pose_controller()

    def _restore_controller(self, i: int) -> None:
        if self.model.restore_scene_controller is not None:
            self.model.restore_scene_controller(self._actors[i])
        if self._controllers and self._controller_type == POSE_CONTROLLER:
            self._controllers[i][2].initialize(False)

    def _validate_targets(self, targets) -> list[Any]:
        if not self._controllers:
            raise RuntimeError("configure_controller() before step_controller()")
        targets = list(targets)
        if len(targets) != self.num_envs:
            raise ValueError("provide exactly one controller target per environment")
        kind = self._controller_type
        expected = getattr(self._r, CONTROLLERS[kind])
        for target in targets:
            if not isinstance(target, expected):
                raise TypeError(f"{kind} requires {expected.__name__} targets")
            if kind == "BASIC_JSC_PD":
                _finite(target.target_pose, (self.model.robot_nv,), "joint target")
            elif kind == "BASIC_OSC_PD":
                _transform(target.root_from_target_ee)
            else:
                _transform(target.world_from_root)
                poses = list(target.local_to_parent_transforms)
                dofs = np.asarray(target.pose_dofs)
                if bool(poses) == bool(dofs.size):
                    raise ValueError(
                        "provide exactly one of pose_dofs or local_to_parent_transforms"
                    )
                if poses:
                    if len(poses) != len(self._links[0]):
                        raise ValueError(
                            "pose transforms must contain one transform per robot link"
                        )
                    for pose in poses:
                        _transform(pose)
                else:
                    _finite(dofs, (self.num_actuators,), "non-root pose DOFs")
        return targets

    def step_controller(self, targets, nsteps: int = 1) -> None:
        """Hold SDK targets, recomputing native control before each physics substep."""
        self._check_open()
        if isinstance(nsteps, bool) or not isinstance(nsteps, (int, np.integer)) or nsteps < 1:
            raise ValueError("nsteps must be a positive integer")
        targets = self._validate_targets(targets)
        if self._execution_mode == "batch":
            self._reject_if_debugger_attached()
        pending = self._pending_wrench.copy() if self._pending_wrench.any() else None
        for _ in range(nsteps):
            commands = np.zeros_like(self._ctrl)
            for i, ((_, _, controller), target) in enumerate(zip(self._controllers, targets)):
                if self._controller_type == POSE_CONTROLLER:
                    obsv = self._r.ControllerMochiArticulatedPoseObsv()
                    controller.compute_output(obsv, target)
                else:
                    obsv = controller.get_current_observations_from_mochi()
                    if self._controller_type == "BASIC_JSC_PD":
                        obsv.dt = self._dt
                    effort = np.asarray(controller.compute_output(obsv, target))
                    _finite(effort, (self.model.robot_nv,), "controller output")
                    if self.model.floating and np.any(effort[:6] != 0):
                        raise ValueError(
                            "controller requested effort on an unactuated floating base"
                        )
                    commands[i] = effort[self.model.actuator_qvel_indices]
            # The regular kernels clip joint efforts and preserve body/rigid wrenches.
            # One substep prevents the fused torque-only fast path from bypassing control.
            if pending is not None:
                self._pending_wrench[:] = pending
            self._step_physics(commands, 1)
