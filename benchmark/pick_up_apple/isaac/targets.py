"""Generate shared MuJoCo/SuperDex apple-plan targets for the Isaac runtime.

This helper runs in the repository Python environment because the planner uses
MuJoCo FK.  The Isaac runner invokes it as a subprocess; Isaac never invents its
own joint trajectory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TASK_DT = 0.002
sys.path.insert(0, str(ROOT / "mujoco_src"))

from planning import GraspPlan, Kinematics, plan_body_grasp  # noqa: E402
from runtime import GAIN_SWITCH_TIME  # noqa: E402


def target_sequence(plan: GraspPlan, start: float, stop: float) -> np.ndarray:
    steps = np.arange(0, round((stop - start) / TASK_DT), dtype=int)
    return np.asarray([plan.target(start + step * TASK_DT) for step in steps], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="JSON request path")
    parser.add_argument("--output", required=True, help="NPZ output path")
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text())
    apple_pose = np.asarray(request["apple_pose"], dtype=float)
    if apple_pose.shape != (7,) or not np.isfinite(apple_pose).all():
        raise ValueError("apple_pose must be a finite xyz + wxyz seven-vector")
    stem = json.loads((ROOT / "superdex/assets/stem.json").read_text())
    kin = Kinematics()
    plan = GraspPlan(kin, apple_pose, stem)

    if request.get("replan"):
        reference = np.asarray(request["joint_positions"], dtype=float)
        if reference.shape != (len(kin.names),) or not np.isfinite(reference).all():
            raise ValueError("joint_positions must be finite and match the shared planner order")
        plan.regrasp = plan_body_grasp(kin, reference, apple_pose[:3])

    start = float(request.get("start", 0.0))
    stop = float(request["stop"])
    targets = target_sequence(plan, start, stop)
    if targets.shape != (round((stop - start) / TASK_DT), len(kin.names)):
        raise AssertionError("target generator returned an unexpected shape")
    if not np.isfinite(targets).all():
        raise RuntimeError("shared planner generated a non-finite target")
    if request.get("replan") and start != GAIN_SWITCH_TIME:
        raise ValueError("Isaac replan must occur at the shared 23 second gain switch")

    metadata = {
        "start_s": start,
        "stop_s": stop,
        "dt_s": TASK_DT,
        "replan": bool(request.get("replan", False)),
        "gain_switch_time_s": GAIN_SWITCH_TIME,
        "pre": plan.pre.tolist(),
    }
    np.savez_compressed(args.output, targets=targets, names=np.asarray(kin.names), **metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
