"""SuperDex-only setup and contact instrumentation for the apple benchmark.

The shared backend owns the runtime and all task stepping. Keep SDK access here:
solver tuning, isolated preparation scenes, and contact geometry/force attribution
are not part of SimBackend's portable API.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
from scipy.spatial.transform import Rotation
from superdex import physics, robotics


def configure_solver(scene):
    solver = scene.get_solver_params()
    solver.non_linear_solver.max_iter = 128
    solver.non_linear_solver.rel_step_tol = 0
    solver.linear_solver.solver_type = physics.LinearSolverType.GMRES
    solver.linear_solver.max_iter = 200
    solver.experimental_eval.fitted_saturation_hessian.contact_friction = False
    scene.set_solver_params(solver)


@contextmanager
def planning_robot(path):
    """IK and gravity compensation must never move the live robot."""
    scene = physics.create_scene("apple grasp planning only")
    bot = None
    context = robotics.create_context()
    try:
        prefab = robotics.load_bot_prefab_from_file(str(path))
        bot = robotics.create_bot(scene, prefab, context)
        actor = bot.get_articulated_actor()
        links = [scene.get_actor(h) for h in actor.get_nested_link_actors()]
        yield prefab, context, actor, links
    finally:
        if bot is not None:
            robotics.destroy_bot(scene, bot)
        physics.destroy_scene(scene)


def settled_apple(backend, dt):
    """Settle the fruit without a robot, exactly as in Dexlab's preparation."""
    scene = physics.create_scene("apple settling only")
    try:
        scene.set_gravity([0, 0, -9.81])
        configure_solver(scene)
        # Share immutable shapes with the live scene so the fine SDF is cooked once.
        # Only this isolated scene is stepped; the live robot remains untouched.
        sources = {a.get_name(): a for a in backend._rigids[0]}
        table = sources["table"]
        scene.create_rigid_actor(
            name="table",
            shape=table.get_reference_shape(),
            is_static=True,
            density=250,
            collider_type=physics.ColliderType.BOX,
            world_from_local=table.get_root_transform(),
        )
        source = sources["apple_with_stem"]
        apple = scene.create_rigid_actor(
            name="apple_with_stem",
            shape=source.get_reference_shape(),
            mass=0.2,
            collider_type=physics.ColliderType.SDF,
            sdf=physics.GridSdfParams(
                resolution_mode=physics.GridSdfResolutionMode.EXPLICIT,
                resolution_delta=[0.0002] * 3,
            ),
            world_from_local=source.get_root_transform(),
        )
        for _ in range(round(3 / dt)):
            scene.step(dt)
        pose = apple.get_root_transform()
        # The reference creates its robot after settling; preserve the apple velocity.
        qpos = np.r_[pose.translation, np.asarray(pose.rotation)[[3, 0, 1, 2]]]
        rotation = Rotation.from_quat(np.asarray(pose.rotation))
        # Public root linear velocity is measured at the body origin, not the COM.
        omega = np.asarray(apple.get_angular_velocity())
        offset = np.asarray(apple.get_center_of_mass_transform().translation) - np.asarray(
            pose.translation
        )
        linear = np.asarray(apple.get_linear_velocity()) - np.cross(omega, offset)
        return qpos, np.r_[linear, rotation.inv().apply(omega)]
    finally:
        # Return detached values only; the actor is not valid beyond this scene.
        physics.destroy_scene(scene)


class Instrumentation:
    """Narrow SDK boundary; never advances or teleports the live task scene."""

    def __init__(self, backend, body_mesh, stem_mesh, prefab):
        self.scene = backend._worlds[0]
        self.links = backend._links[0]
        self.names = list(
            next(
                item.link_names
                for item in backend.get_controller_descriptions()
                if item.identifier == "MOCHI_ARTICULATED_POSE"
            )
        )
        self.handles = {link.get_handle().value: name for link, name in zip(self.links, self.names)}
        rigids = {a.get_name(): a for a in backend._rigids[0]}
        self.apple, self.table = rigids["apple_with_stem"], rigids["table"]
        self.body_mesh, self.stem_mesh = body_mesh, stem_mesh
        configure_solver(self.scene)
        self.apple.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
        self.apple.register_query(physics.QueryType.CONTACT_POINTS)
        for link in [self.apple] + [
            link
            for info, link in zip(prefab.links, self.links)
            if info.name.startswith("r_") and info.shape_file
        ]:
            contact = link.get_contact_params()
            contact.penalty_threshold_default = 0.0001
            contact.penalty_smoothing_half_distance = 0.00015
            contact.penalty_coefficient = 1e10
            contact.friction_falloff_vel = 0.00002
            link.set_contact_params(contact)

    def velocity(self):
        return np.array(self.apple.get_linear_velocity())

    def dynamics(self, time, velocity_before):
        force = np.array(self.apple.get_contact_force_world())
        table = np.array(self.apple.get_contact_force_from_actor_world(self.table))
        return dict(
            time=time,
            total_force=force.tolist(),
            table_force=table.tolist(),
            hand_force=(force - table).tolist(),
            velocity_before=velocity_before.tolist(),
            velocity=self.velocity().tolist(),
            apple_status=self.apple.get_convergence_status().name,
            scene_status=self.scene.get_solver_stats().convergence_status.name,
        )

    def metrics(self, time, pose):
        import trimesh

        rotation = Rotation.from_quat(pose[[4, 5, 6, 3]])
        vertices = rotation.apply(self.body_mesh.vertices) + pose[:3]
        forces = {
            n: np.asarray(self.apple.get_contact_force_from_actor_world(link)).tolist()
            for n, link in zip(self.names, self.links)
        }
        contacts = []
        depths = []
        for point in self.apple.get_contact_points_world():
            on_a = point.actor_a == self.apple.get_handle()
            other = point.actor_b if on_a else point.actor_a
            name = self.handles.get(other.value)
            if name is None:
                continue
            force = np.asarray(point.force) * (1 if on_a else -1)
            if np.linalg.norm(force) < 1e-6:
                continue
            position = np.asarray(point.pos_a if on_a else point.pos_b)
            local = rotation.inv().apply(position - pose[:3])
            depths.append(max(0.0, -point.distance))
            contacts.append((name, force, local))
        off_stem = 0.0
        if contacts:
            points = np.array([row[2] for row in contacts])
            _, fruit_distance, _ = trimesh.proximity.closest_point(self.body_mesh, points)
            _, stem_distance, _ = trimesh.proximity.closest_point(self.stem_mesh, points)
            off_stem = sum(
                float(np.linalg.norm(row[1]))
                for row, df, ds in zip(contacts, fruit_distance, stem_distance)
                if df <= ds + 0.0002
            )
        stats = self.scene.get_solver_stats()
        return dict(
            time=time,
            apple_position=pose[:3].tolist(),
            clearance=float(vertices[:, 2].min() - 0.2995),
            robot_contact_forces=forces,
            off_stem_contact_force_n=off_stem,
            maximum_robot_apple_penetration_m=max(depths, default=0.0),
            solver_status=stats.convergence_status.name,
            table_force=np.asarray(
                self.apple.get_contact_force_from_actor_world(self.table)
            ).tolist(),
        )
