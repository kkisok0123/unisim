#!/usr/bin/env python3
"""Sweep first-pickup MuJoCo pinch Kp and Kv on a powers-of-two grid."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np
from planning import GraspPlan, Kinematics
from runtime import (
    DT,
    ROOT,
    _apply_model_gains,
    compile_physics_model,
    hinge_names,
    model_joint_arrays,
    write_runtime_scene,
)

SAMPLE_DT = 0.05
FIRST_PICKUP_SECONDS = 14.0
PAD_NAMES = ("r_index_finger_pad", "r_thumb_pad")
CONTACT_NAMES = (*PAD_NAMES, "table")
CONTACT_LABELS = tuple(zip(CONTACT_NAMES, ("index", "thumb", "table"), strict=True))
DEFAULT_OUTPUT = ROOT.parents[1] / "results/pick_up_apple_kp_kv_sweep"
SUMMARY_FILE = "sweep_summary.csv"
DEFAULT_KPS = tuple(2**exponent for exponent in range(5, 11))
DEFAULT_KVS = tuple(2**exponent for exponent in range(6))
PINCH_PREFIXES = ("r_index_finger", "r_thumb")


def _gain_arrays(
    names: tuple[str, ...], pinch_kp: float, pinch_kv: float
) -> tuple[np.ndarray, np.ndarray]:
    """Change only right index/thumb gains; preserve all other joint gains."""
    kp = np.empty(len(names), dtype=float)
    kv = np.empty(len(names), dtype=float)
    for i, name in enumerate(names):
        if name.startswith(PINCH_PREFIXES):
            kp[i], kv[i] = pinch_kp, pinch_kv
        elif name.startswith(("l_", "r_")):
            kp[i], kv[i] = 0.8, 0.1
        elif name.startswith("arm_openarm_"):
            kp[i], kv[i] = 1000.0, 40.0
        else:
            raise ValueError(f"Unsupported joint: {name}")
    return kp, kv


def _trial_file(kp: int, kv: int) -> str:
    return f"contact_force_kp_{kp:04d}_kv_{kv:02d}.csv"


def _save_samples(output: Path, kp: int, kv: int, records: list[dict]) -> None:
    fields = ["pinch_kp", "pinch_kv", "time_s", "apple_z_m"]
    for _, short in CONTACT_LABELS:
        fields.extend(f"{short}_force_{axis}_n" for axis in ("x", "y", "z", "magnitude"))
        fields.append(f"{short}_active_contacts")
    with (output / _trial_file(kp, kv)).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {
                "pinch_kp": kp, "pinch_kv": kv,
                "time_s": record["time_s"], "apple_z_m": record["apple_z_m"],
            }
            for name, short in CONTACT_LABELS:
                force = record["forces_world_n"][name]
                for axis, value in zip(("x", "y", "z"), force, strict=True):
                    row[f"{short}_force_{axis}_n"] = value
                row[f"{short}_force_magnitude_n"] = float(np.linalg.norm(force))
                row[f"{short}_active_contacts"] = record["active_contact_counts"][name]
            writer.writerow(row)


def _load_samples(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def _contact_forces(
    model: mujoco.MjModel, data: mujoco.MjData, apple_geom: int, geom_names: dict[int, str]
) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Return world forces on the apple, accounting for MuJoCo geom order."""
    forces = {name: np.zeros(3) for name in CONTACT_NAMES}
    counts = {name: 0 for name in CONTACT_NAMES}
    local = np.zeros(6)
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.geom1 == apple_geom:
            other, sign = contact.geom2, -1
        elif contact.geom2 == apple_geom:
            other, sign = contact.geom1, 1
        else:
            continue
        name = geom_names.get(other)
        if name is None:
            continue
        local.fill(0)
        mujoco.mj_contactForce(model, data, i, local)
        forces[name] += sign * (np.asarray(contact.frame).reshape(3, 3).T @ local[:3])
        if local[0] > 0:
            counts[name] += 1
    return {name: value.tolist() for name, value in forces.items()}, counts


def _mean(records: list[dict], name: str, start: float, stop: float) -> float | None:
    values = [
        np.linalg.norm(row["forces_world_n"][name])
        for row in records
        if start <= row["time_s"] < stop
    ]
    return float(np.mean(values)) if values else None


def _summary(
    kp: int,
    kv: int,
    records: list[dict],
    completed: int,
    requested: int,
    warnings: list[str],
) -> dict:
    result = {
        "pinch_kp": kp,
        "pinch_kv": kv,
        "trial_file": _trial_file(kp, kv),
        "completed_steps": completed,
        "requested_steps": requested,
        "complete": completed == requested and not warnings,
        "warnings": warnings,
        "max_apple_z_m": max((row["apple_z_m"] for row in records), default=None),
    }
    for name, short in ((PAD_NAMES[0], "index"), (PAD_NAMES[1], "thumb")):
        magnitudes = [float(np.linalg.norm(row["forces_world_n"][name])) for row in records]
        active = [row["time_s"] for row, value in zip(records, magnitudes) if value > 0.01]
        for start, stop in ((5, 8), (8, 9), (11, 14)):
            result[f"{short}_mean_{start}_{stop}_n"] = (
                _mean(records, name, start, stop) if completed * DT >= stop else None
            )
        result[f"{short}_peak_n"] = max(magnitudes, default=0.0)
        result[f"{short}_last_contact_s"] = active[-1] if active else None
    return result


def _run_chunk(
    trials: list[tuple[int, int]], seconds: float, runtime_scene: str, output: str
) -> list[dict]:
    """Compile once in this process, then reset the same model for each trial."""
    model, collision = compile_physics_model(runtime_scene)
    if not np.isclose(model.opt.timestep, DT, atol=1e-12, rtol=0):
        raise RuntimeError("Unexpected MuJoCo physics timestep")
    names = hinge_names(model)
    kin = Kinematics()
    if names != kin.names:
        raise RuntimeError("Runtime and planner joint orders differ")
    qadr, _ = model_joint_arrays(model)
    apple_joint = int(model.joint("apple_with_stem_root").qposadr[0])
    apple_pose = np.asarray(model.qpos0[apple_joint : apple_joint + 7], dtype=float).copy()
    stem = json.loads((ROOT / "superdex/assets/stem.json").read_text())
    plan = GraspPlan(kin, apple_pose, stem)
    data = mujoco.MjData(model)
    apple_geom = int(model.geom("apple_with_stem_collision").id)
    geom_names = {
        int(model.geom(f"{name}_collision").id): name for name in PAD_NAMES
    }
    geom_names[int(model.geom("table").id)] = "table"
    if not all(
        np.isclose(model.geom_friction[geom_id, 0], 1.0, atol=1e-12, rtol=0)
        for geom_id in (*geom_names, apple_geom)
    ):
        raise RuntimeError("The sweep requires default sliding friction 1.0")

    steps = round(seconds / DT)
    stride = round(SAMPLE_DT / DT)
    summaries = []
    previous_warning_handler = mujoco.get_mju_user_warning()
    try:
        for pinch_kp, pinch_kv in trials:
            kp, kv = _gain_arrays(names, pinch_kp, pinch_kv)
            _apply_model_gains(model, names, kp, kv)
            mujoco.mj_resetData(model, data)
            data.qpos[:] = model.qpos0
            data.qpos[qadr] = plan.pre
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            records: list[dict] = []
            warnings: list[str] = []
            mujoco.set_mju_user_warning(lambda message: warnings.append(str(message)))
            completed = 0
            for step in range(steps):
                t = step * DT
                data.ctrl[:] = plan.target(t)
                mujoco.mj_step(model, data)
                completed = step + 1
                if not all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc)):
                    warnings.append(f"Non-finite state at {t:.3f} s")
                    break
                if warnings:
                    break
                if completed % stride == 0:
                    forces, counts = _contact_forces(model, data, apple_geom, geom_names)
                    records.append(
                        {
                            "time_s": completed * DT,
                            "forces_world_n": forces,
                            "active_contact_counts": counts,
                            "apple_z_m": float(data.qpos[apple_joint + 2]),
                        }
                    )
            summary = _summary(
                pinch_kp, pinch_kv, records, completed, steps, warnings,
            )
            summary.update({
                "backend": "native MuJoCo",
                "physics_dt_s": DT,
                "sample_dt_s": SAMPLE_DT,
                "pinch_joints": "|".join(name for name in names if name.startswith(PINCH_PREFIXES)),
                "other_hand_kv": 0.1,
                "sliding_friction": 1.0,
                "planner_gravity_feedforward_kp": 30.0,
                "collision_model": json.dumps(collision, sort_keys=True),
            })
            _save_samples(Path(output), pinch_kp, pinch_kv, records)
            print(
                f"Kp={pinch_kp}, Kv={pinch_kv}: {completed}/{steps} steps, "
                f"complete={summary['complete']}",
                flush=True,
            )
            summaries.append(summary)
    finally:
        mujoco.set_mju_user_warning(previous_warning_handler)
    return summaries


def _save_summary(output: Path, summaries: list[dict]) -> None:
    for row in summaries:
        row["failure_time_s"] = None if row["complete"] else row["completed_steps"] * DT
        if not row["complete"]:
            for short in ("index", "thumb"):
                for suffix in ("mean_5_8_n", "mean_8_9_n", "mean_11_14_n", "peak_n"):
                    row[f"{short}_{suffix}"] = None
    with (output / SUMMARY_FILE).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows({**row, "warnings": json.dumps(row["warnings"])} for row in summaries)


def _load_summary(output: Path, trials: list[tuple[int, int]]) -> list[dict]:
    with (output / SUMMARY_FILE).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {}
    for row in rows:
        for key in ("pinch_kp", "pinch_kv", "completed_steps", "requested_steps"):
            row[key] = int(row[key])
        row["complete"] = row["complete"] == "True"
        row["warnings"] = json.loads(row["warnings"])
        for key in row:
            if key.endswith(("_n", "_m", "_s")):
                row[key] = float(row[key]) if row[key] else None
        lookup[(row["pinch_kp"], row["pinch_kv"])] = row
    missing = [trial for trial in trials if trial not in lookup]
    if missing:
        raise ValueError(f"Missing Kp/Kv trials in {SUMMARY_FILE}: {missing}")
    return [lookup[trial] for trial in trials]


def _plot(output: Path, summaries: list[dict]) -> None:
    """Plot 5–8 s force traces per Kp and mean force versus Kv for all Kp values."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kps = sorted({row["pinch_kp"] for row in summaries})
    kvs = sorted({row["pinch_kv"] for row in summaries})
    lookup = {(row["pinch_kp"], row["pinch_kv"]): row for row in summaries}
    pads = (("index", "Index pad"), ("thumb", "Thumb pad"))
    kv_colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(kvs)))
    for kp in kps:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
        for kv, color in zip(kvs, kv_colors, strict=True):
            row = lookup[(kp, kv)]
            samples = [
                sample for sample in _load_samples(output / row["trial_file"])
                if 5 <= sample["time_s"] <= 8
            ]
            partial = not row["complete"] or row["completed_steps"] * DT < 8
            suffix = " (partial)" if partial else ""
            if not samples:
                suffix += " (no samples)"
            for ax, (short, _) in zip(axes, pads, strict=True):
                ax.plot(
                    [sample["time_s"] for sample in samples],
                    [sample[f"{short}_force_magnitude_n"] for sample in samples],
                    color=color, linestyle="--" if partial else "-",
                    label=f"Kv = {kv}{suffix}",
                )
        for ax, (_, label) in zip(axes, pads, strict=True):
            ax.set_title(label)
            ax.set_ylabel("Contact-force magnitude (N)")
            ax.set_xlim(5, 8)
            ax.grid(alpha=0.3)
            ax.legend(ncol=3, fontsize=8)
        axes[-1].set_xlabel("Simulation time (s)")
        fig.suptitle(f"Contact force over time · Kp = {kp} · 5–8 s")
        fig.savefig(output / f"contact_force_vs_time_kp_{kp:04d}_5_8_s.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    kp_colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(kps)))
    for kp, color in zip(kps, kp_colors, strict=True):
        for ax, (short, _) in zip(axes, pads, strict=True):
            values = []
            for kv in kvs:
                row = lookup[(kp, kv)]
                value = row[f"{short}_mean_5_8_n"]
                values.append(value if row["complete"] and value is not None else np.nan)
            ax.plot(kvs, values, "o-", color=color, label=f"Kp = {kp}")
    for ax, (_, label) in zip(axes, pads, strict=True):
        ax.set_title(label)
        ax.set_ylabel("Mean contact-force magnitude (N)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(kvs, [str(kv) for kv in kvs])
        ax.grid(alpha=0.3)
        ax.legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Kv (N·m·s/rad)")
    fig.suptitle("Mean contact force versus Kv · 5–8 s · unavailable trials omitted")
    fig.savefig(output / "mean_contact_force_vs_kv_5_8_s.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=5, help="independent MuJoCo model processes")
    parser.add_argument("--seconds", type=float, default=FIRST_PICKUP_SECONDS)
    parser.add_argument("--kps", type=int, nargs="+", default=DEFAULT_KPS)
    parser.add_argument("--kvs", type=int, nargs="+", default=DEFAULT_KVS)
    parser.add_argument(
        "--plot-only", action="store_true", help="redraw PNG plots from saved CSV files"
    )
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or not DT <= args.seconds <= FIRST_PICKUP_SECONDS:
        parser.error("--seconds must be between one physics step and 14 s")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    for label, values in (("Kp", args.kps), ("Kv", args.kvs)):
        if any(value <= 0 for value in values) or len(set(values)) != len(values):
            parser.error(f"{label} values must be unique positive integers")
    kps = sorted(args.kps)
    kvs = sorted(args.kvs)
    trials = [(kp, kv) for kp in kps for kv in kvs]
    args.out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        summaries = _load_summary(args.out, trials)
        _plot(args.out, summaries)
        print(f"Redrew PNG plots from CSV files in {args.out}")
        return
    descriptor, name = tempfile.mkstemp(
        prefix="runtime-kp-sweep-", suffix=".xml", dir=ROOT / "mujoco"
    )
    os.close(descriptor)
    try:
        runtime_scene = write_runtime_scene(path=name)
        worker_count = min(args.jobs, len(trials))
        chunks = [
            trials[i * len(trials) // worker_count : (i + 1) * len(trials) // worker_count]
            for i in range(worker_count)
        ]
        print(f"Kp/Kv assignments: {chunks}", flush=True)
        summaries = []
        with ProcessPoolExecutor(
            max_workers=len(chunks), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(_run_chunk, chunk, args.seconds, str(runtime_scene), str(args.out))
                for chunk in chunks
            ]
            for future in as_completed(futures):
                summaries.extend(future.result())
    finally:
        Path(name).unlink(missing_ok=True)
    summaries.sort(key=lambda row: (row["pinch_kp"], row["pinch_kv"]))
    _save_summary(args.out, summaries)
    _plot(args.out, summaries)
    print(f"Saved force samples, summary, and charts to {args.out}")


if __name__ == "__main__":
    main()
