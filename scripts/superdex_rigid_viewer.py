#!/usr/bin/env python3
"""Inspect an externally authored rigid asset; optionally record a renderer smoke.

This uses native playback without saving images. It does not author a scene.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--controlled-joints", default="")
    parser.add_argument("--effort-limit", type=float, default=1)
    parser.add_argument(
        "--no-gravity", action="store_true",
        help="Disable gravity for this viewer session; leave the asset unchanged.",
    )
    parser.add_argument("--frames", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.frames is not None and args.frames < 3:
        parser.error("--frames must be at least 3")
    from superdex import physics as p
    from superdex import robotics as r

    from unisim import create_backend
    from unisim.backend.superdex.joints import joint_coordinates
    from unisim.scene import SceneCfg

    path = args.model.resolve()
    options = {}
    if path.suffix in {".superdex_bot", ".superdex_bot_archive"}:
        cfg = r.load_bot_prefab_from_file(str(path))
        names, _, _, _ = joint_coordinates(p, list(cfg.joints))
        options["superdex_effort_limits"] = [args.effort_limit] * len(names)
    else:
        selected = args.controlled_joints.split(",") if args.controlled_joints else []
        options["superdex_controlled_joints"] = selected
        # Explicit component names make the number of effort limits unambiguous.
        options["superdex_effort_limits"] = [args.effort_limit] * len(selected)
    backend = create_backend("superdex", SceneCfg(str(path)), 1, .002,
                             superdex_execution_mode="serial", **options)
    frame, error = 0, 0.0
    frames = args.frames
    reset_frame = None if frames is None else frames * 2 // 3

    def advance(_):
        nonlocal frame, error
        if frames is None or frames // 3 <= frame < reset_frame:
            tick = frame if frames is None else frame - frames // 3
            command = (.005 * np.sin(tick * .013 + np.arange(backend.num_actuators)))[None]
            backend.step(command)
        elif frame == reset_frame:
            backend.reset()
            error = float(np.max(np.abs(backend.get_state()["qpos"][0]
                                       - backend.get_default_qpos()), initial=0))
        frame += 1
        return None

    try:
        if args.no_gravity:
            backend.model.gravity[:] = 0
            for world in backend._worlds:
                world.set_gravity(backend.model.gravity)
        backend.run_playback(
            env=SimpleNamespace(cfg=SimpleNamespace(ctrl_dt=.002)), initialize=lambda: None,
            step=advance, num_steps=frames, headless=frames is not None, record_video=False,
        )
        if error > 3e-5:
            raise RuntimeError(f"viewer reset error: {error}")
    finally:
        backend.close()
    result = {"asset": str(path), "status": "passed", "frames": frame,
              "reset_error": error, "evidence": "renderer smoke" if frames else "interactive",
              "manual_inspection": "not claimed", "saved_images": False,
              "gravity_disabled": args.no_gravity}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
