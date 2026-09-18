"""Small serialized Stage 8 fixtures derived from public SDK schemas.

Generated locally; no SDK checkout, downloads or existing asset modifications.
The cube is closed and has nonzero volume so mass/inertia remain well defined.
"""

from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path

CUBE = (
    "v -.1 -.1 -.1\nv .1 -.1 -.1\nv .1 .1 -.1\nv -.1 .1 -.1\n"
    "v -.1 -.1 .1\nv .1 -.1 .1\nv .1 .1 .1\nv -.1 .1 .1\n"
    "f 1 3 2\nf 1 4 3\nf 5 6 7\nf 5 7 8\nf 1 2 6\nf 1 6 5\n"
    "f 4 8 7\nf 4 7 3\nf 1 5 8\nf 1 8 4\nf 2 3 7\nf 2 7 6\n"
)


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def generate(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "cube.obj").write_text(CUBE)
    (root / ".superdex_root").write_text("")
    actor = {
        "name": "robot",
        "joints": [
            {"name": "base", "type": "Hard"},
            {"name": "a", "type": "Revolute", "axis": [0, 0, 1], "inertia": 0.1},
            {"name": "b", "type": "Revolute", "axis": [0, 0, 1], "inertia": 0.1},
        ],
        "links": [
            {"name": "root", "mass": 1, "shape": "./cube.obj"},
            {
                "name": "left",
                "mass": 1,
                "parentLink": 0,
                "shape": "./cube.obj",
                "parentJointFromLink": {"translation": [0.3, 0, 0]},
            },
            {
                "name": "right",
                "mass": 1,
                "parentLink": 0,
                "shape": "./cube.obj",
                "parentJointFromLink": {"translation": [-0.3, 0, 0]},
            },
        ],
    }
    for link in actor["links"]:
        link["colliderType"] = "None"
    bot = copy.deepcopy(actor)
    bot["defaultPose"] = [0, 0]
    for joint in bot["joints"]:
        joint["effortLimit"] = 1
    plain = write(root / "plain.superdex_bot", bot)
    transmission = copy.deepcopy(bot)
    transmission["linearTransmissions"] = [
        {
            "name": "coupler",
            "jointIndices": [1, 2],
            "jointCoefficients": [0.1, -0.1],
            "jointAxisDisps": [0, 0],
            "targetDisplacement": 0.01,
            "stiffness": 20,
            "damping": 1,
            "allowCompressiveForce": True,
        }
    ]
    tendon = copy.deepcopy(bot)
    tendon["spatialTendons"] = [
        {
            "name": "tendon",
            "routingElements": [
                {"type": "Waypoint", "index": 1, "localPosition": [0, 0.1, 0]},
                {"type": "Waypoint", "index": 2, "localPosition": [0, -0.1, 0]},
                {"type": "LinearJoint", "index": 2, "coefficient": 0.02},
            ],
            "targetDisplacement": -0.05,
            "stiffness": 20,
            "damping": 1,
            "allowCompressiveForce": False,
        }
    ]
    paths = {
        "plain": plain,
        "transmission": write(root / "transmission.superdex_bot", transmission),
        "tendon": write(root / "tendon.superdex_bot", tendon),
    }
    ball_bot = copy.deepcopy(bot)
    ball_bot["joints"][1].update(type="Spherical", minLimit=[-0.4] * 3, maxLimit=[0.4] * 3)
    ball_bot["defaultPose"] = [0] * 4
    paths["ball_bot"] = write(root / "ball.superdex_bot", ball_bot)
    archive = root / "robot.superdex_bot_archive"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
        for name in ("plain.superdex_bot", "cube.obj", ".superdex_root"):
            out.writestr(zipfile.ZipInfo(name), (root / name).read_bytes())
        out.writestr(
            zipfile.ZipInfo(".mochi_bot_archive_metadata"),
            json.dumps(
                {
                    "target": "plain.superdex_bot",
                    "date": "2026-09-17",
                    "botHash": "synthetic",
                    "commitHash": "synthetic",
                    "warnings": [],
                }
            ),
        )
    paths["archive"] = archive
    camera_bot = copy.deepcopy(ball_bot)
    camera_bot["links"][1]["sensors"] = [
        {
            "name": "camera",
            "type": "SENSOR_CAMERA",
            "params": "./camera.superdex_sensor",
            "parentFromSensor": {"translation": [0.1, 0.2, 0.3]},
        }
    ]
    camera_archive = root / "camera.superdex_bot_archive"
    with zipfile.ZipFile(camera_archive, "w", zipfile.ZIP_DEFLATED) as out:
        for name, contents in {
            ".superdex_root": "",
            "cube.obj": CUBE,
            "camera.superdex_bot": json.dumps(camera_bot),
            "camera.superdex_sensor": json.dumps({"imageWidth": 320, "imageHeight": 240}),
            ".mochi_bot_archive_metadata": json.dumps(
                {
                    "target": "camera.superdex_bot",
                    "date": "2026-09-17",
                    "botHash": "synthetic",
                    "commitHash": "synthetic",
                    "warnings": [],
                }
            ),
        }.items():
            out.writestr(zipfile.ZipInfo(name), contents)
    paths["camera_archive"] = camera_archive
    spherical = copy.deepcopy(actor)
    spherical["joints"][0]["type"] = "Free"
    spherical["joints"][1].update(type="Spherical", minLimit=[-0.4] * 3, maxLimit=[0.4] * 3)
    spherical["joints"][0]["parentLinkFromJoint"] = {
        "translation": [0.4, 0.3, 0.2],
        "rotation": [0, 0, 0.382683432365, 0.923879532511],
    }
    spherical["links"][0]["parentJointFromLink"] = {"translation": [0.1, 0.2, 0]}
    spherical["jointVelocities"] = [0.01] * 10
    paths["spherical"] = write(
        root / "spherical.mochi_prefab",
        {
            "actors": {"articulated": [spherical]},
            "scene": {"gravity": [0, 0, 0]},
        },
    )
    paths["multiple"] = write(
        root / "multiple.mochi_scene",
        {
            "prefabs": [
                {"name": "first", "path": "./spherical.mochi_prefab"},
                {
                    "name": "second",
                    "path": "./spherical.mochi_prefab",
                    "translation": [2, 0, 0],
                    "rotation": [0, 0, 0.707106781187, 0.707106781187],
                },
            ],
            "scene": {"gravity": [0, 0, 0]},
        },
    )
    paths["nested"] = write(
        root / "nested.mochi_scene",
        {
            "prefabs": [{"name": "outer", "path": "./multiple.mochi_scene"}],
            "scene": {"gravity": [0, 0, 0]},
        },
    )
    paths["contact"] = write(
        root / "contact.mochi_scene",
        {
            "actors": {
                "rigid": [
                    {"name": "cube", "shape": "./cube.obj", "mass": 1, "translation": [0, 0.5, 0]},
                    {"name": "floor", "shape": "./cube.obj", "scale": [5, 1, 5], "isStatic": True},
                ]
            },
            "scene": {"gravity": [0, -9.81, 0]},
        },
    )
    # Cross-root bot dependency resolution belongs to the SDK, including @tags.
    dep = root / "dependencies"
    dep.mkdir(exist_ok=True)
    (dep / ".superdex_root").write_text("")
    (dep / "cube.obj").write_text(CUBE)
    external = root / "external"
    external.mkdir(exist_ok=True)
    write(external / ".superdex_root", {"@shapes": "../dependencies"})
    external_bot = copy.deepcopy(bot)
    for link in external_bot["links"]:
        link["shape"] = "@shapes/cube.obj"
    paths["external"] = write(external / "external.superdex_bot", external_bot)
    return paths
