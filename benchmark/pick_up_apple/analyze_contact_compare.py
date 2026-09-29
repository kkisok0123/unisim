#!/usr/bin/env python3
"""Summarize matched MuJoCo and SuperDex apple-contact measurements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PAIRS = (
    "index_stem", "thumb_stem", "index_fruit", "thumb_fruit",
    "apple_table", "other_apple_contact",
)
FIELDS = (
    "contact_count", "normal_load_sum_n", "friction_load_sum_n",
    "penetration_max_mm", "penetration_mean_mm", "total_force_x_n",
    "total_force_y_n", "total_force_z_n", "normal_force_x_n",
    "normal_force_y_n", "normal_force_z_n", "friction_force_x_n",
    "friction_force_y_n", "friction_force_z_n",
)
PHASES = (
    ("approach", 0, 5),
    ("pinch", 5, 7),
    ("lift_onset", 7, 7.2),
    ("lift_transition", 7.2, 7.5),
    ("contact_loss", 7.5, 8),
    ("lift", 8, 9),
    ("stem_hold", 11, 14),
)


def _load(path: Path) -> dict[tuple[float, str], dict]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {(round(float(row["time_s"]), 6), row["pair"]): row for row in rows}


def _float(row: dict, key: str) -> float:
    return float(row[key])


def _fruit_vertices() -> np.ndarray:
    path = Path(__file__).resolve().parent / "superdex/assets/objects/apple-collision.obj"
    return np.asarray(
        [[float(value) for value in line.split()[1:4]]
         for line in path.read_text(encoding="ascii").splitlines() if line.startswith("v ")],
        dtype=float,
    )


def _geometric_clearance_mm(row: dict, vertices: np.ndarray) -> float:
    quaternion = [_float(row, f"apple_quat_{axis}") for axis in "xyzw"]
    rotated = Rotation.from_quat(quaternion).apply(vertices)
    return 1000 * (_float(row, "apple_z_m") + float(rotated[:, 2].min()) - 0.2995)


def _difference_csv(path: Path, superdex: dict, mujoco: dict, vertices: np.ndarray) -> dict:
    common = sorted(superdex.keys() & mujoco.keys())
    fields = ["time_s", "pair"]
    for name in ("superdex", "mujoco", "mujoco_minus_superdex"):
        fields.extend(f"{name}_{field}" for field in FIELDS)
        fields.extend(f"{name}_{axis}_m" for axis in ("apple_x", "apple_y", "apple_z"))
        fields.extend((f"{name}_apple_clearance_mm", f"{name}_table_geometric_overlap_mm"))
    clearances: dict[tuple[str, float], float] = {}
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for time_s, pair in common:
            a, b = superdex[(time_s, pair)], mujoco[(time_s, pair)]
            row = {"time_s": time_s, "pair": pair}
            for field in FIELDS:
                va, vb = _float(a, field), _float(b, field)
                row[f"superdex_{field}"] = va
                row[f"mujoco_{field}"] = vb
                row[f"mujoco_minus_superdex_{field}"] = vb - va
            for axis in ("apple_x", "apple_y", "apple_z"):
                field = f"{axis}_m"
                va, vb = _float(a, field), _float(b, field)
                row[f"superdex_{field}"] = va
                row[f"mujoco_{field}"] = vb
                row[f"mujoco_minus_superdex_{field}"] = vb - va
            for engine, source in (("superdex", a), ("mujoco", b)):
                key = engine, time_s
                if key not in clearances:
                    clearances[key] = _geometric_clearance_mm(source, vertices)
                row[f"{engine}_apple_clearance_mm"] = clearances[key]
                row[f"{engine}_table_geometric_overlap_mm"] = max(0, -clearances[key])
            for field in ("apple_clearance_mm", "table_geometric_overlap_mm"):
                row[f"mujoco_minus_superdex_{field}"] = (
                    row[f"mujoco_{field}"] - row[f"superdex_{field}"]
                )
            writer.writerow(row)
    return {
        "matched_times": len({time for time, _ in common}),
        "matched_rows": len(common),
        "superdex_unmatched_rows": len(superdex) - len(common),
        "mujoco_unmatched_rows": len(mujoco) - len(common),
    }


def _phase_metrics(rows: dict, pair: str, start: float, stop: float) -> dict | None:
    selected = [
        row for (time, name), row in rows.items()
        if name == pair and start <= time < stop
    ]
    if not selected:
        return None
    active = [row for row in selected if _float(row, "contact_count") > 0]

    def mean(key: str) -> float:
        return float(np.mean([_float(row, key) for row in selected]))

    return {
        "samples": len(selected),
        "active_fraction": len(active) / len(selected),
        "mean_normal_load_n": mean("normal_load_sum_n"),
        "mean_total_force_world_n": [mean(f"total_force_{axis}_n") for axis in "xyz"],
        "mean_normal_force_world_n": [mean(f"normal_force_{axis}_n") for axis in "xyz"],
        "mean_friction_force_world_n": [mean(f"friction_force_{axis}_n") for axis in "xyz"],
        "maximum_penetration_mm": max(_float(row, "penetration_max_mm") for row in selected),
        "mean_maximum_penetration_mm": mean("penetration_max_mm"),
    }


def _make_plot(path: Path, superdex: dict, mujoco: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    colors = {"superdex": "#2466a4", "mujoco": "#d95f02"}
    styles = {"index_stem": "-", "thumb_stem": "--"}

    def series(data: dict, pair: str, field: str, start: float, stop: float):
        rows = sorted(
            (time, row) for (time, name), row in data.items()
            if name == pair and start <= time <= stop
        )
        return [time for time, _ in rows], [_float(row, field) for _, row in rows]

    for engine, data in (("superdex", superdex), ("mujoco", mujoco)):
        color = colors[engine]
        for ax, field, label in (
            (axes[0, 0], "apple_z_m", "Apple Z (m)"),
            (axes[0, 1], "apple_x_m", "Apple X (m)"),
        ):
            x, y = series(data, "apple_table", field, 0, 14)
            ax.plot(x, y, color=color, label=engine)
            ax.set_ylabel(label)
        x, y = series(data, "apple_table", "total_force_z_n", 5, 9)
        axes[1, 1].plot(x, y, color=color, label=engine)
        for pair in ("index_stem", "thumb_stem"):
            label = f"{engine} {pair.split('_')[0]}"
            for ax, field in (
                (axes[1, 0], "normal_load_sum_n"),
                (axes[2, 0], "friction_force_z_n"),
                (axes[2, 1], "penetration_max_mm"),
            ):
                x, y = series(data, pair, field, 5, 9)
                ax.plot(x, y, color=color, linestyle=styles[pair], label=label)
    for ax, ylabel in (
        (axes[1, 0], "Pad normal load (N)"),
        (axes[1, 1], "Table upward force (N)"),
        (axes[2, 0], "Pad friction Z (N)"),
        (axes[2, 1], "Max pad penetration (mm)"),
    ):
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Planner time (s)")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Apple-stem grasp: fresh MuJoCo and SuperDex contact traces")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    data = {
        engine: {kind: _load(root / engine / f"{kind}.csv")
                 for kind in ("coarse", "event")}
        for engine in ("superdex", "mujoco")
    }
    vertices = _fruit_vertices()
    matches = {
        kind: _difference_csv(root / f"difference_{kind}.csv",
                              data["superdex"][kind], data["mujoco"][kind], vertices)
        for kind in ("coarse", "event")
    }
    phases = {}
    for label, start, stop in PHASES:
        phases[label] = {
            pair: {
                engine: _phase_metrics(data[engine]["coarse"], pair, start, stop)
                for engine in ("superdex", "mujoco")
            }
            for pair in PAIRS
        }
    report = {"matches": matches, "phases": phases}
    (root / "phase_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    _make_plot(root / "contact_comparison.png", data["superdex"]["coarse"],
               data["mujoco"]["coarse"])
    print(json.dumps(matches, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
