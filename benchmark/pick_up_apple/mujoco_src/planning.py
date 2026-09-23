"""MuJoCo FK port of the apple-stem / fruit grasp planner."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import hinge_names, load_planning_model

OPEN = np.array([0.05, 0, 0.05, 0.05] * 4 + [0.05, -0.39, 0.05, 0.05])
CLOSED = np.array([0.9, 0, 1.1, 0.8] * 4 + [0.8, -0.6, 1, 1])
NATIVE_PALM_ROTATION = Rotation.from_matrix([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
NATIVE_MIDPOINT = np.array(
    [0.007220762637925224, 0.04766233634419183, -0.10329213992842383]
)
NATIVE_PINCH_AXIS = np.array(
    [0.3907896639367602, -0.00021523124219297515, 0.9204799792693519]
)
NATIVE_PAD_CONTACT_POINTS = {
    "index_finger": np.array([0.0004635374880864237, 0.00342909332255811, -0.022]),
    "thumb": np.array([-0.0005446435071906924, 0.004403587628418048, -0.026]),
}
NATIVE_PAD_CONTACT_NORMALS = {
    "index_finger": np.array([0.05059817682549827, 0.7464876577935838, -0.6634726831330623]),
    "thumb": np.array([-0.06805676636206305, 0.7540358824666177, -0.6532979140523324]),
}
PINCH_SEED = np.array(
    [
        0.2773231275960221,
        0.4929599999416662,
        1.2039319909469213,
        0.9286209939225749,
        1.0745031684903399,
        -0.15277445624527874,
        0.9198764317912855,
        -1.0469962940130848,
    ]
)


def smooth(t: float, start: float, duration: float) -> float:
    x = np.clip((t - start) / duration, 0, 1)
    return float(x * x * (3 - 2 * x))


class Kinematics:
    """Cold-path MuJoCo FK around one controller-free model."""

    def __init__(self) -> None:
        self.model, self.data, self.qadr, self.vadr = load_planning_model()
        self.names = hinge_names(self.model)
        self.indices = {name: index for index, name in enumerate(self.names)}
        joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.names
        ]
        self.bounds = np.asarray(self.model.jnt_range[joint_ids]).T
        self.arm = np.asarray(
            [self.indices[n] for n in self.names if n.startswith("arm_openarm_right_")]
        )
        self.hand = np.asarray([self.indices[n] for n in self.names if n.startswith("r_")])
        self.right = np.r_[self.arm, self.hand]
        self.default = np.zeros(len(self.names), dtype=float)
        for side, shoulder in (("left", -0.52359879016876221), ("right", 0.52359879016876221)):
            arm = np.asarray(
                [self.indices[n] for n in self.names if n.startswith(f"arm_openarm_{side}_")]
            )
            self.default[arm] = [0.0, shoulder, 0.0, 1.3962634801864624, 0, 0, 0]
        hand_default = np.array(
            [0.26150000095367432, 0, 0.52350002527236938, 0.26150000095367432] * 4
            + [
                0.051999986171722412,
                -0.39299997687349783,
                0.26150000095367432,
                0.26150000095367432,
            ]
        )
        for prefix in ("l_", "r_"):
            hand = np.asarray([self.indices[n] for n in self.names if n.startswith(prefix)])
            self.default[hand] = hand_default
        self.PALM_ROTATION = NATIVE_PALM_ROTATION
        self.MIDPOINT = NATIVE_MIDPOINT.copy()
        self.PINCH_AXIS = NATIVE_PINCH_AXIS.copy()
        self.PINCH_AXIS /= np.linalg.norm(self.PINCH_AXIS)
        self.PAD_CONTACT_POINTS = {
            name: value.copy()
            for name, value in NATIVE_PAD_CONTACT_POINTS.items()
        }
        self.PAD_CONTACT_NORMALS = {
            name: value.copy()
            for name, value in NATIVE_PAD_CONTACT_NORMALS.items()
        }

    def set_pose(self, pose: np.ndarray) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[self.qadr] = pose
        mujoco.mj_forward(self.model, self.data)

    def body_pose(self, name: str) -> tuple[np.ndarray, Rotation]:
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 0:
            raise ValueError(f"Unknown MuJoCo planning body: {name}")
        position = np.asarray(self.data.xpos[body], dtype=float).copy()
        rotation = Rotation.from_quat(
            np.asarray(self.data.xquat[body], dtype=float), scalar_first=True
        )
        return position, rotation

    def link_world_transform(self, name: str) -> tuple[np.ndarray, Rotation]:
        return self.body_pose(name)

def solve_arm(kin: Kinematics, reference: np.ndarray, position: np.ndarray, palm_rotation=None):
    ids = kin.arm
    if palm_rotation is None:
        palm_rotation = kin.PALM_ROTATION

    def residual(angles):
        pose = reference.copy()
        pose[ids] = angles
        kin.set_pose(pose)
        link_position, link_rotation = kin.link_world_transform("r_wrist")
        r = link_rotation * palm_rotation.inv()
        return np.r_[link_position - position, 0.2 * r.as_rotvec()]

    def jacobian(angles):
        return np.column_stack(
            [
                (residual(angles + 0.001 * e) - residual(angles - 0.001 * e)) / 0.002
                for e in np.eye(len(ids))
            ]
        )

    rng = np.random.default_rng(8)
    best = None
    for attempt in range(16):
        seed = reference[ids].astype(float) if attempt == 0 else rng.uniform(*kin.bounds[:, ids])
        result = least_squares(
            residual,
            np.clip(seed, kin.bounds[0, ids] + 1e-5, kin.bounds[1, ids] - 1e-5),
            jac=jacobian,
            bounds=(kin.bounds[0, ids] + 1e-5, kin.bounds[1, ids] - 1e-5),
            max_nfev=200,
        )
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
        if np.linalg.norm(best.fun) < 2e-5:
            break
    if np.linalg.norm(best.fun[:3]) > 0.02 or np.linalg.norm(best.fun[3:]) > 0.02:
        raise RuntimeError(f"Unreachable wrist pose {position}: residual {best.fun}")
    pose = reference.copy()
    pose[ids] = best.x
    return pose


def pinch_pose(kin: Kinematics, reference: np.ndarray, gap: float):
    pose = reference.copy()
    for finger, side in [("index_finger", -1), ("thumb", 1)]:
        ids = np.asarray([kin.indices[n] for n in kin.names if n.startswith("r_" + finger)])
        target = kin.MIDPOINT + side * gap / 2 * kin.PINCH_AXIS
        seed = reference[ids].astype(float)

        def residual(x, ids=ids, finger=finger, target=target, seed=seed, side=side):
            pose[ids] = x
            kin.set_pose(pose)
            wrist_position, wrist_rotation = kin.link_world_transform("r_wrist")
            pad_position, pad_rotation = kin.link_world_transform(f"r_{finger}_pad")
            contact = pad_rotation.apply(kin.PAD_CONTACT_POINTS[finger]) + pad_position
            local = wrist_rotation.inv().apply(contact - wrist_position)
            normal = (wrist_rotation.inv() * pad_rotation).apply(kin.PAD_CONTACT_NORMALS[finger])
            return np.r_[
                10 * (local - target),
                0.003 * (normal + side * kin.PINCH_AXIS),
                0.00005 * (x - seed),
            ]

        def jac(x):
            return np.column_stack(
                [
                    (residual(x + 0.001 * e) - residual(x - 0.001 * e)) / 0.002
                    for e in np.eye(len(ids))
                ]
            )

        result = least_squares(
            residual,
            np.clip(seed, kin.bounds[0, ids] + 1e-5, kin.bounds[1, ids] - 1e-5),
            jac=jac,
            bounds=(kin.bounds[0, ids] + 1e-5, kin.bounds[1, ids] - 1e-5),
            max_nfev=200,
        )
        if np.linalg.norm(result.fun[:3]) / 10 > 0.0001:
            raise RuntimeError(f"{finger} pinch IK residual: {result.fun}")
        pose[ids] = result.x
    return pose


def gravity_offset(kin: Kinematics, pose: np.ndarray, ids: np.ndarray, gains: np.ndarray):
    def potential(q):
        kin.set_pose(q)
        value = 0.0
        for body in range(kin.model.nbody):
            name = mujoco.mj_id2name(kin.model, mujoco.mjtObj.mjOBJ_BODY, body)
            if name is None or name == "apple_with_stem":
                continue
            value += kin.model.body_mass[body] * 9.81 * kin.data.xipos[body, 2]
        return value

    offset = np.zeros_like(pose)
    for i, gain in zip(ids, gains, strict=True):
        plus, minus = pose.copy(), pose.copy()
        plus[i] += 0.001
        minus[i] -= 0.001
        offset[i] = (potential(plus) - potential(minus)) / 0.002 / gain
    return offset


def body_grasp_wrist_position(apple_position: np.ndarray, height: float = 0.0) -> np.ndarray:
    """Return the native fruit-grasp wrist target in the shared world frame."""
    position = np.asarray(apple_position, dtype=float).copy()
    position += [0.03, -0.10, 0.045 + height]
    return position


def plan_body_grasp(kin: Kinematics, reference: np.ndarray, apple_position: np.ndarray):
    opened = reference.copy()
    opened[kin.hand] = OPEN
    position = body_grasp_wrist_position(apple_position)
    pre = solve_arm(kin, opened, body_grasp_wrist_position(apple_position, 0.14), kin.PALM_ROTATION)
    grasp = solve_arm(kin, pre, position, kin.PALM_ROTATION)
    closed = grasp.copy()
    closed[kin.hand] = CLOSED
    raised = solve_arm(
        kin, closed, body_grasp_wrist_position(apple_position, 0.16), kin.PALM_ROTATION
    )
    return [
        (23.0, reference.copy()),
        (26.0, pre),
        (29.0, grasp),
        (29.5, grasp),
        (32.5, closed),
        (33.0, closed),
        (36.0, raised),
        (40.0, raised),
    ]


def body_target(keyframes, time):
    for (start, first), (end, second) in itertools.pairwise(keyframes):
        if time <= end:
            return first + smooth(time, start, end - start) * (second - first)
    return keyframes[-1][1].copy()


class GraspPlan:
    def __init__(self, kin: Kinematics, apple_pose: np.ndarray, stem_info: dict):
        self.kin = kin
        home = kin.default.copy()
        pinch_ids = np.asarray(
            [
                kin.indices[n]
                for prefix in ("r_index_finger", "r_thumb")
                for n in kin.names
                if n.startswith(prefix)
            ]
        )
        home[kin.hand] = 0
        home[pinch_ids] = PINCH_SEED
        self.hand_path = [pinch_pose(kin, home, 0.026)]
        for gap in np.linspace(0.026, 0.001, 13)[1:]:
            self.hand_path.append(pinch_pose(kin, self.hand_path[-1], gap))
        self.hand_path = np.asarray(self.hand_path)
        opened, closed = self.hand_path[0], self.hand_path[-1]
        section = min(stem_info["sections"], key=lambda row: abs(row["z"] - 0.056))
        local = np.array(section["center"])
        local[2] += 0.0015
        contact_world = Rotation.from_quat(apple_pose[[4, 5, 6, 3]]).apply(local) + apple_pose[:3]
        tilt = Rotation.from_rotvec(
            kin.PALM_ROTATION.apply(kin.PINCH_AXIS) * np.deg2rad(-20)
        )
        rotation = tilt * kin.PALM_ROTATION
        wrist_position = contact_world - rotation.apply(kin.MIDPOINT)
        self.grasp = solve_arm(kin, opened, wrist_position, rotation)
        self.pre = solve_arm(kin, self.grasp, wrist_position + [0, 0, 0.08], rotation)
        self.lift_path = [self.grasp]
        for height in np.linspace(0.01, 0.12, 12):
            self.lift_path.append(
                solve_arm(kin, self.lift_path[-1], wrist_position + [0, 0, height], rotation)
            )
        self.lift_path = np.asarray(self.lift_path)
        gains = np.full(len(kin.names), 1000.0)
        gains[kin.hand] = 0.8
        gains[pinch_ids] = 30.0
        grasp_closed = self.grasp.copy()
        grasp_closed[kin.hand] = closed[kin.hand]
        raised_closed = self.lift_path[-1].copy()
        raised_closed[kin.hand] = closed[kin.hand]
        controlled = np.concatenate((kin.arm, kin.hand))
        self.feedforward = [
            gravity_offset(kin, q, controlled, gains[controlled])
            for q in [self.pre, self.grasp, grasp_closed, raised_closed]
        ]
        self.regrasp = None

    @staticmethod
    def interpolate(path, fraction):
        coordinate = fraction * (len(path) - 1)
        lower = min(int(coordinate), len(path) - 2)
        blend = coordinate - lower
        return (1 - blend) * path[lower] + blend * path[lower + 1]

    def target(self, time):
        if self.regrasp is not None:
            return body_target(self.regrasp, time)
        t = time
        target = self.pre.copy()
        target[self.kin.arm] += smooth(t, 0.5, 2) * (
            self.grasp[self.kin.arm] - self.pre[self.kin.arm]
        )
        closing = smooth(t, 3, 3) * (1 - smooth(t, 17.5, 2))
        target[self.kin.hand] = self.interpolate(self.hand_path, closing)[self.kin.hand]
        lift = smooth(t, 7, 3) * (1 - smooth(t, 14, 3))
        target[self.kin.arm] += (
            self.interpolate(self.lift_path, lift)[self.kin.arm] - self.grasp[self.kin.arm]
        )
        ff_pre, ff_open, ff_closed, ff_raised = self.feedforward
        target += (
            ff_pre
            + smooth(t, 0.5, 2) * (ff_open - ff_pre)
            + smooth(t, 3, 3) * (ff_closed - ff_open)
            + smooth(t, 7, 3) * (ff_raised - ff_closed)
            + smooth(t, 14, 3) * (ff_closed - ff_raised)
            + smooth(t, 17.5, 2) * (ff_open - ff_closed)
        )
        retreat = smooth(t, 20, 2)
        target[self.kin.arm] += retreat * (self.pre[self.kin.arm] - self.grasp[self.kin.arm])
        target += retreat * (ff_pre - ff_open)
        return target
