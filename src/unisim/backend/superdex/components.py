"""Cold-path audit and verification of built-in native camera components."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CAMERA_TYPE = "SENSOR_CAMERA"
CAMERA_FIELDS = {
    "name": "name",
    "imageWidth": "image_width",
    "imageHeight": "image_height",
    "fovVerticalDeg": "fov_vertical_deg",
    "nearClip": "near_clip",
    "farClip": "far_clip",
    "forwardAxis": "forward_axis",
    "upAxisLocal": "up_axis_local",
    "offsetLocal": "offset_local",
    "lookAt": "look_at",
    "lookDistance": "look_distance",
}


def camera_parameters(params: Any) -> dict[str, Any]:
    """Return detached, JSON-compatible SDK settings (no native handles)."""
    result = {}
    for key in CAMERA_FIELDS.values():
        value = getattr(params, key)
        result[key] = value if isinstance(value, (str, int, float)) else np.asarray(value).tolist()
    return result


@dataclass(frozen=True)
class CameraSpec:
    name: str
    link_index: int
    translation: tuple[float, ...]
    rotation: tuple[float, ...]  # SDK xyzw mount, not the public pose convention.
    params: str


def audit_components(links: list[Any]) -> tuple[CameraSpec, ...]:
    cameras = []
    names = set()
    for index, link in enumerate(links):
        for item in link.actuators:
            raise NotImplementedError(
                f"superdex unsupported actuator component {item.name!r}: {item.type!r}"
            )
        for item in link.sensors:
            if item.type != CAMERA_TYPE:
                raise NotImplementedError(
                    f"superdex unsupported sensor component {item.name!r}: {item.type!r}"
                )
            if not item.name or item.name in names:
                raise ValueError("superdex camera names must be nonempty and unique")
            names.add(item.name)
            t = item.parent_from_sensor
            pos, quat = np.asarray(t.translation), np.asarray(t.rotation)
            if not np.isfinite(pos).all() or not np.isfinite(quat).all():
                raise ValueError("superdex camera mount must be finite")
            if not np.isclose(np.linalg.norm(quat), 1, atol=1e-5):
                raise ValueError("superdex camera mount quaternion must be normalized")
            cameras.append(CameraSpec(item.name, index, tuple(pos), tuple(quat), item.params))
    return tuple(cameras)


def expected_parameters(r: Any, spec: CameraSpec, path: Path) -> dict[str, Any]:
    """Resolve authored settings before spawning; the SDK still owns materialization."""
    value = spec.params.strip()
    if value and not value.startswith("{"):
        root = next((d for d in path.parents if (d / ".superdex_root").is_file()), path.parent)
        target = root / value[2:] if value.startswith("//") else path.parent / value
        data = json.loads(target.read_text())
    else:
        data = json.loads(value or "{}")
    if not isinstance(data, dict) or set(data) - set(CAMERA_FIELDS) - {"comment"}:
        raise ValueError("superdex camera parameters contain unsupported fields")
    params = r.CameraSensorParams()
    for key, value in data.items():
        if key != "comment":
            setattr(params, CAMERA_FIELDS[key], value)
    result = camera_parameters(params)
    numeric = [v for k, v in result.items() if k != "name"]
    if not all(np.isfinite(v).all() for v in numeric):
        raise ValueError("superdex camera parameters must be finite")
    return result


def verify_cameras(bot: Any, specs: tuple[CameraSpec, ...], expected: dict) -> dict[str, Any]:
    handles = list(bot.get_sensor_handles())
    if list(bot.get_actuator_handles()) or len(handles) != len(specs):
        raise RuntimeError("superdex native component inventory differs from authored bot")
    by_name = {s.name: s for s in specs}
    links = list(bot.get_articulated_actor().get_nested_link_actors())
    cameras = {}
    for handle in handles:
        sensor = bot.get_sensor(handle)
        name = sensor.get_name()
        if name not in by_name or name in cameras or sensor.get_type_name() != CAMERA_TYPE:
            raise RuntimeError("superdex camera inventory differs from authored bot")
        spec = by_name[name]
        mount = sensor.get_parent_from_sensor()
        if (
            sensor.get_actor().get_handle() != links[spec.link_index]
            or not np.allclose(mount.translation, spec.translation, atol=1e-7)
            or not np.allclose(mount.rotation, spec.rotation, atol=1e-7)
            or camera_parameters(sensor.get_params()) != expected[name]
        ):
            raise RuntimeError(f"superdex camera {name!r} differs from authored definition")
        cameras[name] = sensor
    return cameras
