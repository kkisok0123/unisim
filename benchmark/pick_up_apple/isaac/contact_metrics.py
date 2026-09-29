"""Convert Isaac apple contacts to the shared comparison CSV schema."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PAIRS = (
    "index_stem", "thumb_stem", "index_fruit", "thumb_fruit",
    "apple_table", "other_apple_contact",
)


def _array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)


def _empty_pair() -> dict:
    return {
        "count": 0,
        "normal": np.zeros(3),
        "friction": np.zeros(3),
        "normal_load": 0.0,
        "friction_load": 0.0,
        "max_depth": 0.0,
        "depth_sum": 0.0,
        "position_sum": np.zeros(3),
        "negative_normal_count": 0,
    }


def _surface(apple_pose: np.ndarray, points: np.ndarray, fruit_mesh, stem_mesh) -> list[str]:
    import trimesh

    if not len(points):
        return []
    position, quaternion = apple_pose[:3], apple_pose[3:]
    inverse_vector = -quaternion[1:]
    relative = points - position
    local = relative + 2.0 * np.cross(
        inverse_vector, np.cross(inverse_vector, relative) + quaternion[0] * relative,
    )
    _, fruit_distance, _ = trimesh.proximity.closest_point(fruit_mesh, local)
    _, stem_distance, _ = trimesh.proximity.closest_point(stem_mesh, local)
    return [
        "fruit" if fruit <= stem + 0.0002 else "stem"
        for fruit, stem in zip(fruit_distance, stem_distance, strict=True)
    ]


def _check_ranges(counts: np.ndarray, starts: np.ndarray, capacity: int, filters: int) -> None:
    if counts.shape != (1, filters) or starts.shape != counts.shape:
        raise RuntimeError(f"Unexpected Isaac contact layout: {counts.shape}, {starts.shape}")
    if int(counts.sum()) >= capacity:
        raise RuntimeError("Isaac contact buffer is full; increase MAX_CONTACTS")
    if np.any(starts < 0) or np.any(counts < 0) or np.any(starts + counts > capacity):
        raise RuntimeError("Isaac contact buffer contains an invalid range")


def capture(
    contact_view, apple_pose, fruit_mesh, stem_mesh, filters: tuple[str, ...], dt: float
) -> dict:
    """Return normal, friction, point and separation summaries for each apple pair."""
    normal_data = contact_view.get_contact_force_data(dt=dt)
    friction_data = contact_view.get_friction_data(dt=dt)
    if normal_data is None or friction_data is None:
        raise RuntimeError("Isaac apple contact data is unavailable")
    magnitudes, points, normals, distances, counts, starts = map(_array, normal_data)
    friction_forces, friction_points, friction_counts, friction_starts = map(
        _array, friction_data
    )
    counts, starts = counts.astype(int), starts.astype(int)
    friction_counts, friction_starts = friction_counts.astype(int), friction_starts.astype(int)
    _check_ranges(counts, starts, len(magnitudes), len(filters))
    _check_ranges(friction_counts, friction_starts, len(friction_forces), len(filters))
    pairs = {name: _empty_pair() for name in PAIRS}
    pose = np.asarray(apple_pose, dtype=float)

    def label_for(filter_name: str, surface: str | None) -> str:
        if filter_name in ("index", "thumb"):
            return f"{filter_name}_{surface}"
        return "apple_table" if filter_name == "table" else "other_apple_contact"

    for index, filter_name in enumerate(filters):
        start, count = int(starts[0, index]), int(counts[0, index])
        normal_points = np.asarray(points[start : start + count], dtype=float).reshape(-1, 3)
        surfaces = _surface(pose, normal_points, fruit_mesh, stem_mesh) if filter_name in (
            "index", "thumb"
        ) else [None] * count
        for offset, surface in enumerate(surfaces):
            force = float(magnitudes[start + offset, 0])
            normal = np.asarray(normals[start + offset], dtype=float)
            distance = float(distances[start + offset, 0])
            if not np.isfinite([force, distance, *normal, *normal_points[offset]]).all():
                raise RuntimeError("Non-finite Isaac normal contact data")
            if abs(force) <= 1e-12:
                continue
            pair = pairs[label_for(filter_name, surface)]
            pair["count"] += 1
            pair["normal"] += force * normal
            # PhysX signs this impulse by pair orientation; load is its magnitude.
            pair["normal_load"] += abs(force)
            pair["max_depth"] = max(pair["max_depth"], -distance)
            pair["depth_sum"] += max(0.0, -distance)
            pair["position_sum"] += normal_points[offset]

        start, count = int(friction_starts[0, index]), int(friction_counts[0, index])
        tangent_points = np.asarray(
            friction_points[start : start + count], dtype=float
        ).reshape(-1, 3)
        surfaces = _surface(pose, tangent_points, fruit_mesh, stem_mesh) if filter_name in (
            "index", "thumb"
        ) else [None] * count
        for offset, surface in enumerate(surfaces):
            force = np.asarray(friction_forces[start + offset], dtype=float)
            if not np.isfinite(force).all():
                raise RuntimeError("Non-finite Isaac friction contact data")
            pair = pairs[label_for(filter_name, surface)]
            pair["friction"] += force
            pair["friction_load"] += float(np.linalg.norm(force))
    return pairs


def _row(time_s: float, pair_name: str, pair: dict, apple_pose: np.ndarray) -> dict:
    count = pair["count"]
    row = {
        "time_s": time_s,
        "pair": pair_name,
        "contact_count": count,
        "normal_load_sum_n": pair["normal_load"],
        "friction_load_sum_n": pair["friction_load"],
        "penetration_max_mm": pair["max_depth"] * 1000,
        "penetration_mean_mm": pair["depth_sum"] * 1000 / count if count else 0.0,
        "negative_normal_count": pair["negative_normal_count"],
    }
    for name, vector in (
        ("total_force", pair["normal"] + pair["friction"]),
        ("normal_force", pair["normal"]),
        ("friction_force", pair["friction"]),
    ):
        for axis, value in zip("xyz", vector, strict=True):
            row[f"{name}_{axis}_n"] = float(value)
        row[f"{name}_magnitude_n"] = float(np.linalg.norm(vector))
    center = pair["position_sum"] / count if count else np.zeros(3)
    for axis, value in zip("xyz", center, strict=True):
        row[f"contact_center_{axis}_m"] = float(value)
    for axis, value in zip("xyz", apple_pose[:3], strict=True):
        row[f"apple_{axis}_m"] = float(value)
    for axis, value in zip("wxyz", apple_pose[3:], strict=True):
        row[f"apple_quat_{axis}"] = float(value)
    return row


def write_csv(path: Path, samples: list[tuple[float, dict, np.ndarray]]) -> None:
    if not samples:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    first = _row(samples[0][0], PAIRS[0], samples[0][1][PAIRS[0]], samples[0][2])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        for time_s, pairs, pose in samples:
            for name in PAIRS:
                writer.writerow(_row(time_s, name, pairs[name], pose))
