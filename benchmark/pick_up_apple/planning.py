"""Dexlab known-pose IK helpers, adapted from its Apache-2.0 apple-stem demo."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from superdex import physics, robotics

OPEN = np.array([0.05, 0, 0.05, 0.05] * 4 + [0.05, -0.39, 0.05, 0.05])
CLOSED = np.array([0.9, 0, 1.1, 0.8] * 4 + [0.8, -0.6, 1, 1])
PALM_ROTATION = Rotation.from_matrix([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])


def smooth(t: float, start: float, duration: float) -> float:
    x = np.clip((t - start) / duration, 0, 1)
    return float(x * x * (3 - 2 * x))


def solve_arm(actor, wrist, reference, arm_ids, bounds, position, palm_rotation=PALM_ROTATION):
    """Solve native unloaded FK before dynamic actors are created."""

    def residual(angles):
        pose = reference.copy()
        pose[arm_ids] = angles
        actor.set_articulated_pose_from_joints(pose=pose)
        transform = wrist.get_root_transform()
        r = Rotation.from_quat(np.array(transform.rotation))
        return np.r_[
            np.array(transform.translation) - position,
            0.2 * (r * palm_rotation.inv()).as_rotvec(),
        ]

    def jacobian(angles):
        return np.column_stack(
            [
                (residual(angles + 0.001 * e) - residual(angles - 0.001 * e)) / 0.002
                for e in np.eye(len(arm_ids))
            ]
        )

    rng = np.random.default_rng(8)
    best = None
    for attempt in range(16):
        seed = reference[arm_ids].astype(float) if attempt == 0 else rng.uniform(*bounds)
        result = least_squares(
            residual,
            np.clip(seed, bounds[0] + 1e-5, bounds[1] - 1e-5),
            jac=jacobian,
            bounds=bounds,
            max_nfev=200,
        )
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
        if np.linalg.norm(best.fun) < 2e-5:
            break
    if np.linalg.norm(best.fun) > 0.0002:
        raise RuntimeError(f"Unreachable wrist pose {position}: residual {best.fun}")
    pose = reference.copy()
    pose[arm_ids] = best.x
    return pose


MIDPOINT = np.array([0.007220762637925224, 0.04766233634419183, -0.10329213992842383])
PINCH_AXIS = np.array([0.3907896639367602, -0.00021523124219297515, 0.9204799792693519])
# Native IK opposition of measured distal surfaces on the current Beta 1 pads.
PAD_CONTACT_POINTS = {
    "index_finger": np.array([0.0004635374880864237, 0.00342909332255811, -0.022]),
    "thumb": np.array([-0.0005446435071906924, 0.004403587628418048, -0.026]),
}
PAD_CONTACT_NORMALS = {
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


def pinch_pose(actor, links, joints, reference, gap):
    pose = reference.copy()
    wrist = links["r_wrist"]

    for finger, side in [("index_finger", -1), ("thumb", 1)]:
        ids = np.array([i for i, j in enumerate(joints) if j.name.startswith("r_" + finger)])
        bounds = np.array(
            [
                [np.dot(np.array(getattr(joints[i], k)), np.array(joints[i].axis)) for i in ids]
                for k in ["min_limit", "max_limit"]
            ]
        )
        target = MIDPOINT + side * gap / 2 * PINCH_AXIS
        seed = reference[ids].astype(float)

        def residual(x, ids=ids, finger=finger, target=target, seed=seed, side=side):
            pose[ids] = x
            actor.set_articulated_pose_from_joints(pose=pose)
            transform = wrist.get_root_transform()
            r = Rotation.from_quat(np.array(transform.rotation))
            pad = links[f"r_{finger}_pad"].get_root_transform()
            pad_rotation = Rotation.from_quat(np.array(pad.rotation))
            contact = pad_rotation.apply(PAD_CONTACT_POINTS[finger]) + np.array(pad.translation)
            local = r.inv().apply(contact - np.array(transform.translation))
            normal = (r.inv() * pad_rotation).apply(PAD_CONTACT_NORMALS[finger])
            return np.r_[
                10 * (local - target),
                0.003 * (normal + side * PINCH_AXIS),
                0.00005 * (x - seed),
            ]

        def jac(x):
            return np.column_stack(
                [(residual(x + 0.001 * e) - residual(x - 0.001 * e)) / 0.002 for e in np.eye(4)]
            )

        result = least_squares(
            residual,
            np.clip(seed, bounds[0] + 1e-5, bounds[1] - 1e-5),
            jac=jac,
            bounds=bounds,
            max_nfev=200,
        )
        if np.linalg.norm(result.fun[:3]) / 10 > 0.0001:
            raise RuntimeError(f"{finger} pinch IK residual: {result.fun}")
        pose[ids] = result.x
    return pose


def gravity_offset(actor, prefab, links, pose, ids, gains):
    """Feed forward dU/dq as a PD target offset; only evaluate unloaded FK."""

    def potential(q):
        actor.set_articulated_pose_from_joints(pose=q)
        value = 0.0
        for info, link in zip(prefab.links, links):
            if info.mass is None or info.center_of_mass is None:
                continue
            t = link.get_root_transform()
            com = Rotation.from_quat(np.array(t.rotation)).apply(
                np.array(info.center_of_mass)
            ) + np.array(t.translation)
            value += info.mass * 9.81 * com[2]
        return value

    offset = np.zeros_like(pose)
    for i in ids:
        plus, minus = pose.copy(), pose.copy()
        plus[i] += 0.001
        minus[i] -= 0.001
        offset[i] = (potential(plus) - potential(minus)) / 0.002 / gains[i]
    return offset


def plan_body_grasp(prefab, context, reference, arm, hand, bounds, apple_position):
    """Never change the live robot or apple while evaluating inverse kinematics."""
    scene = physics.create_scene("body grasp inverse kinematics only")
    bot = None
    try:
        bot = robotics.create_bot(scene, prefab, context)
        actor = bot.get_articulated_actor()
        links = [scene.get_actor(h) for h in actor.get_nested_link_actors()]
        wrist = links[[link.name for link in prefab.links].index("r_wrist")]
        opened = reference.copy()
        opened[hand] = OPEN
        position = np.asarray(apple_position) + [0.03, -0.10, 0.045]
        pre = solve_arm(actor, wrist, opened, arm, bounds, position + [0, 0, 0.14], PALM_ROTATION)
        grasp = solve_arm(actor, wrist, pre, arm, bounds, position, PALM_ROTATION)
        closed = grasp.copy()
        closed[hand] = CLOSED
        raised = solve_arm(
            actor, wrist, closed, arm, bounds, position + [0, 0, 0.16], PALM_ROTATION
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
    finally:
        if bot is not None:
            robotics.destroy_bot(scene, bot)
        physics.destroy_scene(scene)


def body_target(keyframes, time):
    for (start, first), (end, second) in itertools.pairwise(keyframes):
        if time <= end:
            return first + smooth(time, start, end - start) * (second - first)
    return keyframes[-1][1].copy()


class GraspPlan:
    """Plan joint targets from the live settled apple, without recorded trajectories."""

    def __init__(self, prefab, actor, links, apple_pose, stem_info):
        self.prefab = prefab
        joints = [j for j in prefab.joints if j.type == physics.ArticulatedJointType.REVOLUTE]
        self.joint_names = [j.name for j in joints]
        self.arm = np.array(
            [i for i, j in enumerate(joints) if j.name.startswith("arm_openarm_right_joint")]
        )
        self.hand = np.array([i for i, j in enumerate(joints) if j.name.startswith("r_")])
        self.bounds = np.array(
            [
                [
                    np.dot(np.asarray(getattr(joints[i], k)), np.asarray(joints[i].axis))
                    for i in self.arm
                ]
                for k in ["min_limit", "max_limit"]
            ]
        )
        home = np.empty(actor.get_num_dofs(), dtype=np.float64)
        actor.get_articulated_pose(home)
        home[self.hand] = 0
        pinch_ids = np.array(
            [
                i
                for prefix in ("r_index_finger", "r_thumb")
                for i, j in enumerate(joints)
                if j.name.startswith(prefix)
            ]
        )
        home[pinch_ids] = PINCH_SEED
        # Native actor names are namespaced; IK uses authored prefab link names.
        lookup = {info.name: link for info, link in zip(prefab.links, links)}
        self.hand_path = [pinch_pose(actor, lookup, joints, home, 0.026)]
        for gap in np.linspace(0.026, 0.001, 13)[1:]:
            self.hand_path.append(pinch_pose(actor, lookup, joints, self.hand_path[-1], gap))
        self.hand_path = np.asarray(self.hand_path)
        opened, closed = self.hand_path[0], self.hand_path[-1]
        section = min(stem_info["sections"], key=lambda row: abs(row["z"] - 0.056))
        local = np.array(section["center"])
        local[2] += 0.0015
        contact_world = Rotation.from_quat(apple_pose[[4, 5, 6, 3]]).apply(local) + apple_pose[:3]
        rotation = (
            Rotation.from_rotvec(PALM_ROTATION.apply(PINCH_AXIS) * np.deg2rad(-20)) * PALM_ROTATION
        )
        wrist_position = contact_world - rotation.apply(MIDPOINT)
        self.grasp = solve_arm(
            actor, lookup["r_wrist"], opened, self.arm, self.bounds, wrist_position, rotation
        )
        self.pre = solve_arm(
            actor,
            lookup["r_wrist"],
            self.grasp,
            self.arm,
            self.bounds,
            wrist_position + [0, 0, 0.08],
            rotation,
        )
        self.lift_path = [self.grasp]
        for height in np.linspace(0.01, 0.12, 12):
            self.lift_path.append(
                solve_arm(
                    actor,
                    lookup["r_wrist"],
                    self.lift_path[-1],
                    self.arm,
                    self.bounds,
                    wrist_position + [0, 0, height],
                    rotation,
                )
            )
        self.lift_path = np.asarray(self.lift_path)
        gains = np.full(len(home), 1000.0)
        gains[self.hand] = 0.8
        gains[pinch_ids] = 30
        grasp_closed = self.grasp.copy()
        grasp_closed[self.hand] = closed[self.hand]
        raised_closed = self.lift_path[-1].copy()
        raised_closed[self.hand] = closed[self.hand]
        self.feedforward = [
            gravity_offset(actor, prefab, links, q, np.r_[self.arm, self.hand], gains)
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
        target[self.arm] += smooth(t, 0.5, 2) * (self.grasp[self.arm] - self.pre[self.arm])
        closing = smooth(t, 3, 3) * (1 - smooth(t, 17.5, 2))
        target[self.hand] = self.interpolate(self.hand_path, closing)[self.hand]
        lift = smooth(t, 7, 3) * (1 - smooth(t, 14, 3))
        target[self.arm] += self.interpolate(self.lift_path, lift)[self.arm] - self.grasp[self.arm]
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
        target[self.arm] += retreat * (self.pre[self.arm] - self.grasp[self.arm])
        target += retreat * (ff_pre - ff_open)
        return target
