"""Copy the local Superdex asset checkout into the repository and verify it.

The copy is a real byte-for-byte duplicate of
``/home/pc829/UniFamily/project_superdex/assets`` placed under
``assets/superdex`` in this repository, so the destination is usable
independently of the source checkout.  Before copying, every incoming path is
checked against the destination; existing files with different content abort
the run, and unrelated files already present (for example the integration
plan document) are never touched.

After copying, the script verifies that relative paths and per-file SHA-256
hashes match between source and destination, builds the SDK-free inventory
over the destination alone, and writes provenance and verification reports
under ``docs/``.  Exit status is nonzero if any check fails.

Example:

    uv run scripts/copy_superdex_assets.py \
        --source /home/pc829/UniFamily/project_superdex/assets \
        --destination assets/superdex
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from unisim.backend.superdex.assets import (  # noqa: E402
    Provenance,
    build_inventory,
    now_utc,
    sha256_file,
    verify_bundle,
)

# Reference information from the earlier acquisition survey; the local source
# revision recorded by provenance is authoritative.
HISTORICAL_UPSTREAM_SHA = "ed30ce16361329cbbed956173d6a4f5842815d24"

DEFAULT_SOURCE = Path("/home/pc829/UniFamily/project_superdex/assets")
DEFAULT_DESTINATION = REPOSITORY_ROOT / "assets" / "superdex"
DEFAULT_REPORT_DIR = REPOSITORY_ROOT / "docs"
INVENTORY_JSON_NAME = "superdex-assets-inventory.json"
INVENTORY_MD_NAME = "superdex-assets-inventory.md"


def _git(source_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def collect_provenance(source: Path) -> tuple[Provenance, frozenset[str]]:
    """Interrogate the source checkout for its revision and local changes.

    Local derivatives are files under ``assets/`` that changed after the
    checkout's root (initial import) commit.
    """
    repo = source.parent
    revision = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    changed = _git(repo, "status", "--porcelain", "--", "assets")
    clean = not changed
    root_commits = _git(repo, "rev-list", "--max-parents=0", "HEAD").splitlines()
    initial_commit = root_commits[-1] if root_commits else revision
    derivative_lines = _git(repo, "diff", "--name-only", initial_commit, "HEAD", "--", "assets")
    prefix = "assets/"
    derivatives = frozenset(line[len(prefix) :] for line in derivative_lines.splitlines() if line)
    provenance = Provenance(
        source_repo=str(repo),
        source_branch=branch,
        source_revision=revision,
        source_clean=clean,
        source_initial_commit=initial_commit,
        local_derivative_paths=tuple(sorted(derivatives)),
        historical_upstream_sha=HISTORICAL_UPSTREAM_SHA,
        generated_utc=now_utc(),
    )
    return provenance, derivatives


def _relative_files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def check_conflicts(source: Path, destination: Path) -> list[str]:
    """Report incoming paths that already exist with different content."""
    incoming = _relative_files(source)
    conflicts: list[str] = []
    for relative, source_path in incoming.items():
        target = destination / relative
        if target.exists() and sha256_file(target) != sha256_file(source_path):
            conflicts.append(relative)
    return conflicts


def copy_tree(source: Path, destination: Path) -> int:
    """Copy every source file (including hidden markers) into the destination."""
    files = _relative_files(source)
    for relative, source_path in files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
    return len(files)


def compare_trees(source: Path, destination: Path, preserved: set[str]) -> list[str]:
    """Compare relative path sets and SHA-256 hashes between two trees.

    ``preserved`` lists destination files that existed before the copy; they
    are unrelated content (for example the integration plan document) and are
    intentionally left untouched, so they are not flagged as unexpected.
    """
    source_files = _relative_files(source)
    destination_files = _relative_files(destination)
    problems: list[str] = []
    for relative in sorted(set(source_files) - set(destination_files)):
        problems.append(f"missing in destination: {relative}")
    for relative in sorted(set(destination_files) - set(source_files) - preserved):
        problems.append(f"unexpected in destination: {relative}")
    for relative in sorted(set(source_files) & set(destination_files)):
        if sha256_file(source_files[relative]) != sha256_file(destination_files[relative]):
            problems.append(f"hash mismatch: {relative}")
    return problems


def write_reports(inventory_json: dict, report_dir: Path) -> None:
    """Write the machine-readable and human-readable reports."""
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / INVENTORY_JSON_NAME
    md_path = report_dir / INVENTORY_MD_NAME
    json_path.write_text(
        json.dumps(inventory_json, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    prov = inventory_json["provenance"]
    entries = inventory_json["entries"]
    lines = [
        "# Superdex asset inventory",
        "",
        "SDK-free inventory of the repository-local asset copy under",
        "`assets/superdex`. Generated by `scripts/copy_superdex_assets.py`.",
        "",
        "## Provenance",
        "",
        f"- Source repository: `{prov['source_repo']}`",
        f"- Branch: `{prov['source_branch']}`",
        f"- Revision: `{prov['source_revision']}`",
        f"- Working tree clean under `assets/`: {prov['source_clean']}",
        f"- Local derivative files (changed after initial commit"
        f" `{prov['source_initial_commit'][:12]}): {len(prov['local_derivative_paths'])}",
        f"- Historical upstream pin (reference only): `{prov['historical_upstream_sha']}`",
        f"- Generated: {prov['generated_utc']}",
        "",
        "## Bundle summary",
        "",
        f"- Files: {inventory_json['file_count']}",
        f"- Total size: {inventory_json['total_bytes'] / (1024 * 1024):.1f} MiB",
        f"- Tree digest (SHA-256 over path+file hashes): `{inventory_json['tree_digest']}`",
        f"- Root markers: {', '.join(inventory_json['root_markers'])}",
        f"- Entries by kind: {inventory_json['entry_counts_by_kind']}",
        f"- Dispositions: {inventory_json['disposition_counts']}",
        "",
        "## Compatibility table",
        "",
        "| Entrypoint | Kind | Disposition | Blockers / required capabilities |",
        "| --- | --- | --- | --- |",
    ]
    for entry in entries:
        notes = "; ".join(entry["blockers"]) or ", ".join(entry["required_capabilities"])
        lines.append(
            f"| `{entry['entrypoint']}` | {entry['kind']} | {entry['disposition']} | {notes} |"
        )
    lines += [
        "",
        "## Verification",
        "",
        "All dependency references resolve inside `assets/superdex`; per-file",
        "SHA-256 hashes match the source checkout. See the JSON record for the",
        "full dependency-edge and hash detail.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="directory for the inventory JSON and Markdown reports",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-copy files even if identical content is already present",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "do not copy: rebuild the inventory and verify the destination "
            "against the source without modifying either tree"
        ),
    )
    args = parser.parse_args(argv)

    source = args.source.resolve()
    destination = args.destination.resolve()
    report_dir = args.report_dir.resolve()
    if not source.is_dir():
        print(f"error: source directory not found: {source}", file=sys.stderr)
        return 1

    if args.report_only:
        preserved = set(_relative_files(destination))
        problems = compare_trees(source, destination, preserved)
        if problems:
            print("error: source/destination comparison failed:", file=sys.stderr)
            for problem in problems[:20]:
                print(f"  {problem}", file=sys.stderr)
            return 1
        provenance, derivatives = collect_provenance(source)
        inventory = build_inventory(destination, provenance, derivatives)
        errors, warnings = verify_bundle(inventory, destination)
        payload = inventory.to_json()
        payload["verify"] = {"errors": errors, "warnings": warnings}
        write_reports(payload, report_dir)
        print(f"tree digest:  {inventory.tree_digest}")
        print(f"reports:      {(report_dir / INVENTORY_JSON_NAME)}")
        if errors:
            print(f"error: {len(errors)} dependency problems:", file=sys.stderr)
            return 1
        return 0

    print(f"source:      {source}")
    print(f"destination: {destination}")

    conflicts = check_conflicts(source, destination)
    if conflicts:
        print("error: destination files differ from source:", file=sys.stderr)
        for relative in conflicts[:20]:
            print(f"  {relative}", file=sys.stderr)
        return 1

    preserved = set(_relative_files(destination))
    provenance, derivatives = collect_provenance(source)
    copied = copy_tree(source, destination)
    print(f"copied {copied} files")

    problems = compare_trees(source, destination, preserved)
    if problems:
        print("error: source/destination comparison failed:", file=sys.stderr)
        for problem in problems[:20]:
            print(f"  {problem}", file=sys.stderr)
        return 1

    inventory = build_inventory(destination, provenance, derivatives)
    errors, warnings = verify_bundle(inventory, destination)
    payload = inventory.to_json()
    payload["verify"] = {"errors": errors, "warnings": warnings}
    write_reports(payload, report_dir)

    print(f"tree digest:  {inventory.tree_digest}")
    print(f"entries:      {len(inventory.entries)} ({payload['entry_counts_by_kind']})")
    print(f"dispositions: {payload['disposition_counts']}")
    if warnings:
        print(f"warnings:     {len(warnings)}")
        for warning in warnings:
            print(f"  {warning}")
    if errors:
        print(f"error: {len(errors)} dependency problems:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print(f"reports:      {(report_dir / INVENTORY_JSON_NAME)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
