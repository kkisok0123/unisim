#!/usr/bin/env python3
"""Run native offscreen movement/reset smoke checks for Stage 4."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from superdex_bot_profiles import FLOATING_PROFILES  # noqa: E402
from superdex_bot_qualify import INVENTORY_REPORT, resolve_assets_root  # noqa: E402

from unisim.backend.superdex.assets import verify_asset_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets")
    parser.add_argument("--bots", default=",".join(sorted(FLOATING_PROFILES)))
    parser.add_argument("--out", type=Path,
                        default=ROOT / "docs/superdex-floating-qualification/viewers")
    args = parser.parse_args()
    root = resolve_assets_root(args.assets)
    verified = verify_asset_bundle(root, INVENTORY_REPORT)
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for key in args.bots.split(","):
        if key not in FLOATING_PROFILES:
            parser.error(f"unknown floating viewer profile: {key}")
        out = args.out / key
        out.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT / "scripts/superdex_bot_viewer.py"),
                   "--assets", str(root), "--bot", key, "--frames", "570",
                   "--skip-verification", "--out", str(out)]
        run = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (out / "run.log").write_text(run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError(f"{key}: viewer exited {run.returncode}; see {out / 'run.log'}")
        report = json.loads((out / "report.json").read_text())
        move = report["phases"]["bounded-movement"]
        reset = report["phases"]["reset"]
        if not (move["peak_abs_deviation_rad"] > 1e-4
                and move["peak_root_displacement_m"] > 1e-3
                and reset["restored_qpos_max_error_rad"] <= 1e-6):
            raise AssertionError(f"{key}: movement/reset demonstration did not pass")
        results[key] = {"movement": move, "reset": reset}
        print(f"{key}: movement and reset passed", flush=True)
    (args.out / "summary.json").write_text(json.dumps({
        "asset_verification": verified, "frames_per_model": 570, "models": results,
        "render_mode": "native offscreen Polyscope",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
