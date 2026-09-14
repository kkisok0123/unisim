"""SDK-free tests for the Superdex asset inventory and copy tooling.

All fixtures are synthetic ``tmp_path`` bundles; no engine SDK, network
access, or local asset checkout is required.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from unisim.backend.superdex.assets import (
    DISPOSITION_PROFILE_CANDIDATE,
    DISPOSITION_RECIPE_CANDIDATE,
    DISPOSITION_UNRESOLVED_DEPENDENCY,
    DISPOSITION_UNSUPPORTED_FEATURES,
    Provenance,
    SuperdexAssetError,
    build_inventory,
    hash_tree,
    resolve_reference,
    verify_asset_bundle,
    verify_bundle,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "copy_superdex_assets.py"


def _provenance() -> Provenance:
    return Provenance(
        source_repo="/tmp/source-repo",
        source_branch="test-branch",
        source_revision="a" * 40,
        source_clean=True,
        source_initial_commit="b" * 40,
        local_derivative_paths=(),
        historical_upstream_sha="c" * 40,
        generated_utc="2026-09-14T00:00:00Z",
    )


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _seed_bundle(root: Path) -> Path:
    """Create a small synthetic bundle with one marker root and two bots."""
    (root / "bots").mkdir(parents=True)
    (root / "bots" / ".superdex_root").write_bytes(b"")
    _write(
        root / "bots" / "arms" / "arm.superdex_bot",
        {
            "name": "arm",
            "links": [
                {"name": "base", "shape": "collision/base.mochi.h5",
                 "renderModel": "render/base.glb"},
            ],
            "joints": [{"name": "world_joint", "type": "Hard"}],
            "defaultPose": [0.0],
        },
    )
    (root / "bots" / "arms" / "collision").mkdir(parents=True)
    (root / "bots" / "arms" / "collision" / "base.mochi.h5").write_bytes(b"col")
    (root / "bots" / "arms" / "render").mkdir(parents=True)
    (root / "bots" / "arms" / "render" / "base.glb").write_bytes(b"ren")
    _write(
        root / "bots" / "hands" / "hand.superdex_bot",
        {
            "name": "hand",
            "links": [{"name": "palm", "shape": "collision/palm.mochi.h5"}],
            "joints": [{"name": "world_joint", "type": "Free"}],
        },
    )
    (root / "bots" / "hands" / "collision").mkdir(parents=True)
    (root / "bots" / "hands" / "collision" / "palm.mochi.h5").write_bytes(b"palm")
    return root


def test_module_imports_without_engine_sdks():
    code = (
        "import sys; from unisim.backend.superdex import assets; "
        "assert not [n for n in sys.modules if n.startswith(('superdex', 'mujoco', 'torch'))]"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_importing_unisim_does_not_load_asset_module():
    code = (
        "import sys; import unisim; "
        "assert 'unisim.backend.superdex.assets' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_build_inventory_classifies_fixed_base_bots(tmp_path):
    bundle = _seed_bundle(tmp_path / "bundle")
    inventory = build_inventory(bundle, _provenance())
    by_entry = {entry.entrypoint: entry for entry in inventory.entries}
    arm = by_entry["bots/arms/arm.superdex_bot"]
    assert arm.disposition == DISPOSITION_PROFILE_CANDIDATE
    assert arm.root_joint_type == "Hard"
    assert arm.link_count == 1
    assert {edge.target for edge in arm.dependencies} == {
        "bots/arms/collision/base.mochi.h5",
        "bots/arms/render/base.glb",
    }
    errors, _ = verify_bundle(inventory, bundle)
    assert errors == []


def test_floating_root_and_components_are_blocked_not_dropped(tmp_path):
    bundle = _seed_bundle(tmp_path / "bundle")
    hand = json.loads(
        (bundle / "bots" / "hands" / "hand.superdex_bot").read_text()
    )
    hand["links"][0]["actuators"] = [
        {"name": "a0", "type": "WUJI_POSITION_SERVO", "params": {}}
    ]
    hand["links"][0]["sensors"] = [
        {"name": "s0", "type": "WUJI_CONTACT_FORCE_SENSOR"}
    ]
    _write(bundle / "bots" / "hands" / "hand.superdex_bot", hand)
    inventory = build_inventory(bundle, _provenance())
    hand_entry = next(e for e in inventory.entries if "hand" in e.entrypoint)
    assert hand_entry.disposition == DISPOSITION_UNSUPPORTED_FEATURES
    assert hand_entry.root_joint_type == "Free"
    assert hand_entry.actuator_count == 1
    assert hand_entry.sensor_count == 1
    assert "native-floating-root" in hand_entry.required_capabilities
    assert "actuator-sensor-components" in hand_entry.required_capabilities


def test_recipe_dependencies_resolve_through_root_marker(tmp_path):
    bundle = _seed_bundle(tmp_path / "bundle")
    _write(
        bundle / "bots" / "combos" / "combo.superdex_bot",
        {
            "name": "combo",
            "base": "//arms/arm.superdex_bot",
            "modifications": [
                {
                    "AttachBot": {
                        "enabled": True,
                        "path": "//hands/hand.superdex_bot",
                        "name": "hand",
                        "parentLinkName": "base",
                        "joint": {"name": "j", "type": "Hard"},
                    }
                }
            ],
        },
    )
    (bundle / "bots" / "combos").mkdir(parents=True, exist_ok=True)
    inventory = build_inventory(bundle, _provenance())
    combo = next(e for e in inventory.entries if "combo" in e.entrypoint)
    assert combo.disposition == DISPOSITION_RECIPE_CANDIDATE
    roles = {edge.role for edge in combo.dependencies}
    assert {"base-bot", "attachment-bot"} <= roles
    errors, _ = verify_bundle(inventory, bundle)
    assert errors == []


def test_missing_dependency_is_reported_as_unresolved(tmp_path):
    bundle = _seed_bundle(tmp_path / "bundle")
    (bundle / "bots" / "arms" / "render" / "base.glb").unlink()
    inventory = build_inventory(bundle, _provenance())
    arm = next(e for e in inventory.entries if "arm" in e.entrypoint)
    assert arm.disposition == DISPOSITION_UNRESOLVED_DEPENDENCY
    errors, _ = verify_bundle(inventory, bundle)
    assert any("render" in error for error in errors)


def test_reference_escaping_bundle_is_rejected(tmp_path):
    bundle = _seed_bundle(tmp_path / "bundle")
    arm_path = bundle / "bots" / "arms" / "arm.superdex_bot"
    arm = json.loads(arm_path.read_text())
    # From bots/arms/, three levels up land outside the bundle root.
    arm["links"][0]["shape"] = "../../../outside.mochi.h5"
    _write(arm_path, arm)
    with pytest.raises(SuperdexAssetError) as excinfo:
        resolve_reference(arm_path, "../../../outside.mochi.h5", bundle)
    assert excinfo.value.code == "dependency_escape"
    inventory = build_inventory(bundle, _provenance())
    arm_entry = next(e for e in inventory.entries if "arm" in e.entrypoint)
    assert arm_entry.disposition == DISPOSITION_UNRESOLVED_DEPENDENCY


def test_rooted_reference_without_marker_is_rejected(tmp_path):
    bundle = _seed_bundle(tmp_path / "bundle")
    (bundle / "bots" / ".superdex_root").unlink()
    _write(
        bundle / "bots" / "combos" / "combo.superdex_bot",
        {"name": "combo", "base": "//arms/arm.superdex_bot", "modifications": []},
    )
    with pytest.raises(SuperdexAssetError) as excinfo:
        build_inventory(bundle, _provenance())
    assert excinfo.value.code == "not_a_bundle"


def test_reference_cycles_are_detected(tmp_path):
    bundle = _seed_bundle(tmp_path / "bundle")
    _write(
        bundle / "bots" / "combos" / "a.superdex_bot",
        {"name": "a", "base": "//combos/b.superdex_bot", "modifications": []},
    )
    _write(
        bundle / "bots" / "combos" / "b.superdex_bot",
        {"name": "b", "base": "//combos/a.superdex_bot", "modifications": []},
    )
    with pytest.raises(SuperdexAssetError) as excinfo:
        build_inventory(bundle, _provenance())
    assert excinfo.value.code == "recipe_cycle"


def test_shared_dependency_reuse_is_not_a_cycle(tmp_path):
    bundle = _seed_bundle(tmp_path / "bundle")
    for name in ("x", "y"):
        _write(
            bundle / "bots" / "combos" / f"{name}.superdex_bot",
            {"name": name, "base": "//arms/arm.superdex_bot", "modifications": []},
        )
    inventory = build_inventory(bundle, _provenance())  # must not raise
    assert len(inventory.entries) == 4


def test_lfs_pointer_detection(tmp_path):
    bundle = _seed_bundle(tmp_path / "bundle")
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64
    )
    (bundle / "bots" / "arms" / "collision" / "base.mochi.h5").write_bytes(pointer)
    inventory = build_inventory(bundle, _provenance())
    errors, _ = verify_bundle(inventory, bundle)
    assert any("lfs_pointer" in error for error in errors)


def test_scene_bundle_root_relative_resolution(tmp_path):
    bundle = tmp_path / "scene-bundle"
    bundle.mkdir()
    (bundle / "benchmarks").mkdir()
    _write(
        bundle / "benchmarks" / "cart.mochi_scene",
        {
            "name": "cart",
            "actors": {
                "articulated": [
                    {
                        "name": "cart",
                        "links": [
                            {"name": "pole", "shape": "benchmarks/pole.mochi.h5"}
                        ],
                        "joints": [{"name": "hinge", "type": "Revolute"}],
                    }
                ]
            },
            "scene": {},
        },
    )
    (bundle / "benchmarks" / "pole.mochi.h5").write_bytes(b"pole")
    (bundle / ".superdex_root").write_bytes(b"")
    inventory = build_inventory(bundle, _provenance())
    scene = next(e for e in inventory.entries if e.kind == "scene")
    assert scene.disposition == DISPOSITION_UNSUPPORTED_FEATURES
    assert "scene-assembly" in scene.required_capabilities
    errors, _ = verify_bundle(inventory, bundle)
    assert errors == []


def test_prefab_nested_references(tmp_path):
    bundle = tmp_path / "prefab-bundle"
    bundle.mkdir()
    (bundle / "prefabs").mkdir()
    (bundle / "prefabs" / ".superdex_root").write_bytes(b"")
    _write(
        bundle / "prefabs" / "blocks.mochi_prefab",
        {
            "actors": {
                "rigid": [
                    {
                        "name": "Box",
                        "isStatic": True,
                        "shape": "./collision/box.mochi.h5",
                        "renderModel": "./render/box.glb",
                    }
                ]
            },
            "prefabs": [
                {"name": "b", "path": "./block_b.mochi_prefab"}
            ],
        },
    )
    (bundle / "prefabs" / "collision").mkdir()
    (bundle / "prefabs" / "collision" / "box.mochi.h5").write_bytes(b"box")
    (bundle / "prefabs" / "render").mkdir()
    (bundle / "prefabs" / "render" / "box.glb").write_bytes(b"glb")
    _write(
        bundle / "prefabs" / "block_b.mochi_prefab",
        {"actors": {"rigid": [{"name": "B", "shape": "collision/b.mochi.h5"}]}},
    )
    (bundle / "prefabs" / "collision").mkdir(exist_ok=True)
    (bundle / "prefabs" / "collision" / "b.mochi.h5").write_bytes(b"b")
    inventory = build_inventory(bundle, _provenance())
    blocks = next(e for e in inventory.entries if "blocks" in e.entrypoint)
    assert blocks.kind == "prefab"
    targets = {edge.target for edge in blocks.dependencies}
    assert "prefabs/collision/box.mochi.h5" in targets
    assert "prefabs/render/box.glb" in targets
    assert "prefabs/block_b.mochi_prefab" in targets
    # Nested prefabs are entrypoints of their own, so their geometry is
    # recorded on their own inventory entry rather than the parent's.
    nested = next(e for e in inventory.entries if "block_b" in e.entrypoint)
    assert {edge.target for edge in nested.dependencies} == {
        "prefabs/collision/b.mochi.h5"
    }
    errors, _ = verify_bundle(inventory, bundle)
    assert errors == []


def test_controller_entry_is_self_contained(tmp_path):
    bundle = tmp_path / "controller-bundle"
    bundle.mkdir()
    (bundle / "bots").mkdir()
    (bundle / "bots" / ".superdex_root").write_bytes(b"")
    _write(
        bundle / "bots" / "control" / "pose.superdex_controller",
        {"poseControllerParams": {"jointTracking": [{"stiffness": 1.0}]}},
    )
    inventory = build_inventory(bundle, _provenance())
    controller = next(e for e in inventory.entries if e.kind == "controller")
    assert controller.dependencies == ()
    assert controller.joint_counts == {"PoseTrackingJoint": 1}
    errors, _ = verify_bundle(inventory, bundle)
    assert errors == []


def test_local_derivative_flag_is_recorded(tmp_path):
    bundle = _seed_bundle(tmp_path / "bundle")
    inventory = build_inventory(
        bundle, _provenance(), {"bots/arms/arm.superdex_bot"}
    )
    arm = next(e for e in inventory.entries if "arm" in e.entrypoint)
    assert arm.local_derivative
    hand = next(e for e in inventory.entries if "hand" in e.entrypoint)
    assert not hand.local_derivative


def test_inventory_json_is_deterministic(tmp_path):
    bundle = _seed_bundle(tmp_path / "bundle")
    first = build_inventory(bundle, _provenance()).to_json()
    second = build_inventory(bundle, _provenance()).to_json()
    assert first == second
    assert first["entry_counts_by_kind"] == {"bot": 2}


def test_hash_tree_deterministic_and_sensitive(tmp_path):
    bundle = _seed_bundle(tmp_path / "bundle")
    digest, count, _ = hash_tree(bundle)
    assert count == 6  # marker, 2 bots, 2 collision meshes, 1 render mesh
    assert digest == hash_tree(bundle)[0]
    (bundle / "bots" / "arms" / "collision" / "base.mochi.h5").write_bytes(b"X")
    assert hash_tree(bundle)[0] != digest


def test_missing_root_marker_rejects_bundle(tmp_path):
    bundle = _seed_bundle(tmp_path / "bundle")
    (bundle / "bots" / ".superdex_root").unlink()
    with pytest.raises(SuperdexAssetError) as excinfo:
        build_inventory(bundle, _provenance())
    assert excinfo.value.code == "not_a_bundle"


# ---------------------------------------------------------------------------
# Copy script tests (still SDK-free; they build synthetic source checkouts).
# ---------------------------------------------------------------------------


def _seed_source_repo(root: Path) -> Path:
    """Create a fake source checkout with a git repo and an assets tree."""
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    _seed_bundle(root / "assets")
    (root / "README.md").write_text("source repo\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "initial", "--no-verify", "--no-gpg-sign"],
        cwd=root,
        check=True,
    )
    return root / "assets"


def _run_copy(
    source: Path, destination: Path, report_dir: Path | None = None
) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        str(SCRIPT),
        "--source",
        str(source),
        "--destination",
        str(destination),
    ]
    if report_dir is not None:
        command += ["--report-dir", str(report_dir)]
    return subprocess.run(
        command, capture_output=True, text=True, cwd=SCRIPT.parents[1]
    )


def test_copy_script_copies_and_verifies(tmp_path):
    source = _seed_source_repo(tmp_path / "src-repo")
    destination = tmp_path / "dest" / "assets"
    reports = tmp_path / "reports"
    result = _run_copy(source, destination, reports)
    assert result.returncode == 0, result.stderr + result.stdout
    assert (destination / "bots" / ".superdex_root").is_file()
    assert (
        destination / "bots" / "arms" / "collision" / "base.mochi.h5"
    ).read_bytes() == b"col"
    report = json.loads(
        (reports / "superdex-assets-inventory.json").read_text()
    )
    assert report["entry_counts_by_kind"] == {"bot": 2}
    # Idempotent re-run also succeeds.
    result2 = _run_copy(source, destination, reports)
    assert result2.returncode == 0, result2.stderr


def test_copy_script_preserves_unrelated_destination_files(tmp_path):
    source = _seed_source_repo(tmp_path / "src-repo")
    destination = tmp_path / "dest" / "assets"
    destination.mkdir(parents=True)
    (destination / "notes.md").write_text("pre-existing\n")
    result = _run_copy(source, destination, tmp_path / "reports")
    assert result.returncode == 0, result.stderr + result.stdout
    assert (destination / "notes.md").read_text() == "pre-existing\n"


def test_copy_script_rejects_conflicting_destination(tmp_path):
    source = _seed_source_repo(tmp_path / "src-repo")
    destination = tmp_path / "dest" / "assets"
    conflict = destination / "bots" / "arms" / "collision" / "base.mochi.h5"
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"different content")
    result = _run_copy(source, destination, tmp_path / "reports")
    assert result.returncode == 1
    assert "collision/base.mochi.h5" in result.stderr


def test_copy_script_deterministic_reports(tmp_path):
    source = _seed_source_repo(tmp_path / "src-repo")
    destination = tmp_path / "dest" / "assets"
    reports = tmp_path / "reports"
    result = _run_copy(source, destination, reports)
    assert result.returncode == 0, result.stderr
    first = json.loads(
        (reports / "superdex-assets-inventory.json").read_text()
    )
    second_run = _run_copy(source, destination, reports)
    assert second_run.returncode == 0
    second = json.loads(
        (reports / "superdex-assets-inventory.json").read_text()
    )
    # The tree digest pins the structure; repeated runs must agree on it and
    # on the entry classification, independent of timestamps.
    assert first["tree_digest"] == second["tree_digest"]
    assert first["entries"] == second["entries"]


def _run_copy(source, destination, reports, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(source),
            "--destination",
            str(destination),
            "--report-dir",
            str(reports),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def test_copy_script_report_only_rebuilds_without_modifying(tmp_path):
    source = _seed_source_repo(tmp_path / "src-repo")
    destination = tmp_path / "dest" / "assets"
    reports = tmp_path / "reports"
    assert _run_copy(source, destination, reports).returncode == 0
    marker = destination / "bots" / "arms" / "collision" / "base.mochi.h5"
    before = marker.read_bytes()
    stamp = marker.stat().st_mtime_ns
    result = _run_copy(source, destination, reports, "--report-only")
    assert result.returncode == 0, result.stderr
    assert marker.read_bytes() == before
    assert marker.stat().st_mtime_ns == stamp


def test_copy_script_report_only_detects_modified_destination(tmp_path):
    source = _seed_source_repo(tmp_path / "src-repo")
    destination = tmp_path / "dest" / "assets"
    reports = tmp_path / "reports"
    assert _run_copy(source, destination, reports).returncode == 0
    (destination / "bots" / "arms" / "collision" / "base.mochi.h5").write_bytes(b"tampered")
    result = _run_copy(source, destination, reports, "--report-only")
    assert result.returncode == 1
    assert "comparison failed" in result.stderr


def _recorded_report(bundle: Path, tmp_path: Path) -> Path:
    inventory = build_inventory(bundle, _provenance())
    report = tmp_path / "recorded-inventory.json"
    report.write_text(json.dumps(inventory.to_json()), encoding="utf-8")
    return report


def test_verify_asset_bundle_matches_recorded_report(tmp_path):
    bundle = _seed_bundle(tmp_path / "bundle")
    report = _recorded_report(bundle, tmp_path)
    summary = verify_asset_bundle(bundle, report)
    assert summary["file_count"] > 0
    assert summary["dependency_errors"] == []


def test_verify_asset_bundle_rejects_modified_tree(tmp_path):
    bundle = _seed_bundle(tmp_path / "bundle")
    report = _recorded_report(bundle, tmp_path)
    (bundle / "bots" / "arms" / "render" / "base.glb").write_bytes(b"tampered")
    with pytest.raises(SuperdexAssetError) as excinfo:
        verify_asset_bundle(bundle, report)
    assert excinfo.value.code == "tree_mismatch"


def test_verify_asset_bundle_rejects_added_files(tmp_path):
    bundle = _seed_bundle(tmp_path / "bundle")
    report = _recorded_report(bundle, tmp_path)
    (bundle / "bots" / "extra.txt").write_text("unrecorded")
    with pytest.raises(SuperdexAssetError) as excinfo:
        verify_asset_bundle(bundle, report)
    assert excinfo.value.code == "tree_mismatch"


def test_verify_asset_bundle_reports_broken_dependency(tmp_path):
    bundle = _seed_bundle(tmp_path / "bundle")
    # Record the report against the reduced tree so its digest matches while
    # one dependency edge is broken; the edge itself must fail the run.
    (bundle / "bots" / "hands" / "collision" / "palm.mochi.h5").unlink()
    reduced = build_inventory(bundle, _provenance())
    report = tmp_path / "recorded-inventory.json"
    report.write_text(json.dumps(reduced.to_json()), encoding="utf-8")
    with pytest.raises(SuperdexAssetError) as excinfo:
        verify_asset_bundle(bundle, report)
    assert excinfo.value.code == "dependency_missing"


def test_verify_asset_bundle_rejects_broken_report(tmp_path):
    bundle = _seed_bundle(tmp_path / "bundle")
    report = tmp_path / "inventory.json"
    report.write_text(json.dumps({"entries": []}), encoding="utf-8")
    with pytest.raises(SuperdexAssetError, match="schema_unexpected"):
        verify_asset_bundle(bundle, report)
    with pytest.raises(SuperdexAssetError, match="schema_unexpected"):
        verify_asset_bundle(bundle, tmp_path / "absent.json")

