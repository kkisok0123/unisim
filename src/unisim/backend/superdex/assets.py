"""SDK-free inventory and verification of the repository-local Superdex assets.

This module reads the JSON manifest formats used by the SuperDex asset tree
(``.superdex_bot``, ``.mochi_prefab``, ``.mochi_scene`` and
``.superdex_controller`` files), resolves every cross-file reference between
them, and classifies each entry by the adapter capability it requires.  It is
deliberately engine-free: only the standard library is imported, so inventory
runs work on any machine without the optional SuperDex runtime installed.

The module is not imported by :mod:`unisim`'s public package boundary; use it
from tooling (``scripts/copy_superdex_assets.py``) or tests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_MARKER_NAME = ".superdex_root"
ROOTED_REFERENCE_PREFIX = "//"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
BOT_SUFFIX = ".superdex_bot"
PREFAB_SUFFIX = ".mochi_prefab"
SCENE_SUFFIX = ".mochi_scene"
CONTROLLER_SUFFIX = ".superdex_controller"
ENTRYPOINT_SUFFIXES = (BOT_SUFFIX, PREFAB_SUFFIX, SCENE_SUFFIX, CONTROLLER_SUFFIX)

# The native loader accepts a fixed (Hard) root plus these joint types.
PROFILE_JOINT_TYPES = frozenset({"Hard", "Revolute", "Prismatic"})
SUPPORTED_JOINT_TYPES = PROFILE_JOINT_TYPES | {"Free"}

# Reference-bearing JSON fields and the dependency role each one carries.
_ROLE_BY_FIELD = {
    "shape": "collision",
    "renderModel": "render",
    "base": "base-bot",
    "path": "attachment-bot",
}

# Dispositions used by the compatibility records.
DISPOSITION_PROFILE_CANDIDATE = "profile-candidate"
DISPOSITION_RECIPE_CANDIDATE = "recipe-candidate"
DISPOSITION_UNSUPPORTED_FEATURES = "unsupported-features"
DISPOSITION_UNRESOLVED_DEPENDENCY = "unresolved-dependency"
DISPOSITION_CONTROL_PROFILE = "control-profile"

# Capability names cited by blockers; each maps to a numbered integration
# stage in the asset integration plan.
CAPABILITY_NATIVE_FLOATING_ROOT = "native-floating-root"
CAPABILITY_ACTUATOR_SENSOR_COMPONENTS = "actuator-sensor-components"
CAPABILITY_MECHANICAL_CYCLES = "mechanical-cycles"
CAPABILITY_PREFAB_RIGID_ACTORS = "prefab-rigid-actors"
CAPABILITY_PREFAB_ARTICULATED_ACTORS = "prefab-articulated-actors"
CAPABILITY_SCENE_ASSEMBLY = "scene-assembly"
CAPABILITY_SCENE_CONTROLLERS = "scene-controllers"
CAPABILITY_POSE_CONTROLLER_PROFILE = "pose-controller-profile"
CAPABILITY_RECIPE_COMPOSITION = "recipe-composition"

_STAGE_BY_CAPABILITY = {
    CAPABILITY_NATIVE_FLOATING_ROOT: 4,
    CAPABILITY_ACTUATOR_SENSOR_COMPONENTS: 6,
    CAPABILITY_MECHANICAL_CYCLES: 8,
    CAPABILITY_PREFAB_RIGID_ACTORS: 5,
    CAPABILITY_PREFAB_ARTICULATED_ACTORS: 5,
    CAPABILITY_SCENE_ASSEMBLY: 7,
    CAPABILITY_SCENE_CONTROLLERS: 7,
    CAPABILITY_POSE_CONTROLLER_PROFILE: 6,
    CAPABILITY_RECIPE_COMPOSITION: 3,
}

_ERROR_CODE_DETAIL = {
    "dependency_missing": "referenced file is absent",
    "dependency_escape": "reference resolves outside the asset bundle",
    "lfs_pointer": "referenced file is an unresolved git-LFS pointer",
    "schema_unexpected": "manifest is not the expected JSON shape",
    "root_marker_missing": "rooted reference has no ancestor .superdex_root marker",
    "recipe_cycle": "recipe modification graph contains a cycle",
    "not_a_bundle": "no .superdex_root marker anywhere in the tree",
    "ambiguous_reference": "reference matches multiple resolution roots",
}


class SuperdexAssetError(Exception):
    """A structured inventory or verification failure with a stable code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class Provenance:
    """Where the copied asset bytes came from."""

    source_repo: str
    source_branch: str
    source_revision: str
    source_clean: bool
    source_initial_commit: str
    local_derivative_paths: tuple[str, ...]
    historical_upstream_sha: str
    generated_utc: str

    def to_json(self) -> dict[str, Any]:
        return {
            "source_repo": self.source_repo,
            "source_branch": self.source_branch,
            "source_revision": self.source_revision,
            "source_clean": self.source_clean,
            "source_initial_commit": self.source_initial_commit,
            "local_derivative_paths": list(self.local_derivative_paths),
            "historical_upstream_sha": self.historical_upstream_sha,
            "generated_utc": self.generated_utc,
        }


@dataclass(frozen=True)
class DependencyEdge:
    """One file requirement extracted from a manifest field."""

    source: str
    field: str
    role: str
    target: str
    resolved: bool = True
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "field": self.field,
            "role": self.role,
            "target": self.target,
            "resolved": self.resolved,
            "note": self.note,
        }


@dataclass(frozen=True)
class AssetEntry:
    """Per-manifest record with dependencies and capability classification."""

    entrypoint: str
    kind: str
    name: str
    disposition: str
    blockers: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    joint_counts: dict[str, int] = field(default_factory=dict)
    root_joint_type: str | None = None
    link_count: int = 0
    actuator_count: int = 0
    sensor_count: int = 0
    has_cycles: bool = False
    has_contact_overrides: bool = False
    has_world_from_root: bool = False
    dependencies: tuple[DependencyEdge, ...] = ()
    licenses: tuple[str, ...] = ()
    local_derivative: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "entrypoint": self.entrypoint,
            "kind": self.kind,
            "name": self.name,
            "disposition": self.disposition,
            "blockers": list(self.blockers),
            "required_capabilities": list(self.required_capabilities),
            "required_capability_stages": sorted(
                {_STAGE_BY_CAPABILITY[c] for c in self.required_capabilities}
            ),
            "joint_counts": self.joint_counts,
            "root_joint_type": self.root_joint_type,
            "link_count": self.link_count,
            "actuator_count": self.actuator_count,
            "sensor_count": self.sensor_count,
            "has_cycles": self.has_cycles,
            "has_contact_overrides": self.has_contact_overrides,
            "has_world_from_root": self.has_world_from_root,
            "dependencies": [edge.to_json() for edge in self.dependencies],
            "licenses": list(self.licenses),
            "local_derivative": self.local_derivative,
        }


@dataclass(frozen=True)
class Inventory:
    """The complete SDK-free survey of one asset bundle."""

    provenance: Provenance
    entries: tuple[AssetEntry, ...]
    file_count: int
    total_bytes: int
    tree_digest: str
    root_markers: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        kind_counts: dict[str, int] = {}
        disposition_counts: dict[str, int] = {}
        for entry in self.entries:
            kind_counts[entry.kind] = kind_counts.get(entry.kind, 0) + 1
            disposition_counts[entry.disposition] = (
                disposition_counts.get(entry.disposition, 0) + 1
            )
        return {
            "provenance": self.provenance.to_json(),
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "tree_digest": self.tree_digest,
            "root_markers": list(self.root_markers),
            "entry_counts_by_kind": dict(sorted(kind_counts.items())),
            "disposition_counts": dict(sorted(disposition_counts.items())),
            "entries": [entry.to_json() for entry in self.entries],
        }


def now_utc() -> str:
    """Current UTC timestamp in a stable, second-resolution format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    """Stream a file in 1 MiB chunks and return its hex SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path) -> tuple[str, int, int]:
    """Hash sorted (relative path, per-file sha256) pairs of a file tree.

    Returns ``(tree_digest, file_count, total_bytes)``; the digest is
    deterministic across re-extractions of identical trees.
    """
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        file_digest = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
        file_count += 1
        total_bytes += path.stat().st_size
    return digest.hexdigest(), file_count, total_bytes


def _iter_strings(node: Any, trail: list[str]) -> Any:
    """Yield ``(field_path, value)`` for every string under a JSON subtree."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _iter_strings(value, trail + [key])
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _iter_strings(value, trail + [str(index)])
    elif isinstance(node, str):
        yield ".".join(trail), node


def _reference_fields(node: Any) -> list[tuple[str, str]]:
    """Collect ``(field_path, value)`` pairs whose value looks like a path."""
    references: list[tuple[str, str]] = []
    for field_path, value in _iter_strings(node, []):
        leaf = field_path.rsplit(".", 1)[-1] if field_path else ""
        if leaf not in _ROLE_BY_FIELD or not value:
            continue
        if value.startswith(ROOTED_REFERENCE_PREFIX):
            references.append((field_path, value))
        elif "/" in value or value.endswith(ENTRYPOINT_SUFFIXES):
            references.append((field_path, value))
    return references


def _find_root_marker(start: Path, bundle_root: Path) -> Path | None:
    """Walk up from ``start`` looking for a .superdex_root marker directory."""
    for candidate in [start, *start.parents]:
        if candidate == bundle_root.parent:
            break
        if (candidate / ROOT_MARKER_NAME).is_file():
            return candidate
    return None


def resolve_reference(
    manifest: Path, reference: str, bundle_root: Path
) -> tuple[Path, str]:
    """Resolve one manifest reference to a path inside the bundle.

    Returns ``(resolved_path, resolution_rule)``.  Rooted ``//`` references
    resolve against the nearest ancestor ``.superdex_root`` marker;
    ``./``-prefixed and plain relative references resolve against the
    manifest's own directory, with a fallback to the bundle root for
    root-relative authored paths (benchmark scene meshes).  References that
    match neither location return the manifest-relative path unresolved, so
    the caller reports a missing dependency at the authored location.
    """
    if reference.startswith(ROOTED_REFERENCE_PREFIX):
        marker_dir = _find_root_marker(manifest.parent, bundle_root)
        if marker_dir is None:
            raise SuperdexAssetError(
                "root_marker_missing",
                f"{manifest.relative_to(bundle_root)}: {reference!r} has no"
                f" ancestor {ROOT_MARKER_NAME} marker",
            )
        target = (marker_dir / reference[len(ROOTED_REFERENCE_PREFIX) :]).resolve()
        rule = "root-marker"
    else:
        text = reference[2:] if reference.startswith("./") else reference
        direct = (manifest.parent / text).resolve()
        rooted = (bundle_root / text).resolve()
        if bundle_root not in [direct, *direct.parents]:
            raise SuperdexAssetError(
                "dependency_escape",
                f"{manifest.relative_to(bundle_root)}: {reference!r} resolves to"
                f" {direct}, outside the bundle {bundle_root}",
            )
        direct_exists = direct.is_file()
        rooted_exists = rooted.is_file() and rooted != direct
        if direct_exists and rooted_exists:
            raise SuperdexAssetError(
                "ambiguous_reference",
                f"{manifest.relative_to(bundle_root)}: {reference!r} matches both"
                f" {direct} and {rooted}",
            )
        if direct_exists:
            return direct, "manifest-relative"
        if rooted_exists:
            return rooted, "bundle-root-relative"
        return direct, "manifest-relative"
    if bundle_root not in [target, *target.parents]:
        raise SuperdexAssetError(
            "dependency_escape",
            f"{manifest.relative_to(bundle_root)}: {reference!r} resolves to"
            f" {target}, outside the bundle {bundle_root}",
        )
    return target, rule


def _collect_licenses(manifest: Path, bundle_root: Path) -> tuple[str, ...]:
    """Gather LICENSE/NOTICE file paths from the manifest dir up to the root."""
    names: list[str] = []
    for directory in [manifest.parent, *manifest.parent.parents]:
        for item in sorted(directory.iterdir()):
            if item.is_file() and item.name.upper().startswith(("LICENSE", "NOTICE")):
                names.append(item.relative_to(bundle_root).as_posix())
        if directory == bundle_root:
            break
    return tuple(names)


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(64).startswith(LFS_POINTER_PREFIX)
    except OSError:
        return False


def _actor_groups(data: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Return the actor lists of a manifest (bots store links at top level)."""
    actors = data.get("actors")
    if isinstance(actors, dict):
        return [group for group in actors.values() if isinstance(group, list)]
    return []


def _joints_of(data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(data.get("joints"), list):
        return data["joints"]
    joints: list[dict[str, Any]] = []
    for group in _actor_groups(data):
        for actor in group:
            if isinstance(actor, dict) and isinstance(actor.get("joints"), list):
                joints.extend(actor["joints"])
    return joints


def _links_of(data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(data.get("links"), list):
        return data["links"]
    links: list[dict[str, Any]] = []
    for group in _actor_groups(data):
        for actor in group:
            if isinstance(actor, dict) and isinstance(actor.get("links"), list):
                links.extend(actor["links"])
    return links


def _joint_counts(joints: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for joint in joints:
        if isinstance(joint, dict):
            kind = str(joint.get("type", "unknown"))
            counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def _root_joint_type(joints: list[dict[str, Any]]) -> str | None:
    for joint in joints:
        if isinstance(joint, dict) and "type" in joint:
            return str(joint["type"])
    return None


def _components_of(links: list[dict[str, Any]]) -> tuple[int, int, tuple[str, ...]]:
    actuator_count = 0
    sensor_count = 0
    types: set[str] = set()
    for link in links:
        if not isinstance(link, dict):
            continue
        for actuator in link.get("actuators") or []:
            actuator_count += 1
            if isinstance(actuator, dict) and "type" in actuator:
                types.add(f"actuator:{actuator['type']}")
        for sensor in link.get("sensors") or []:
            sensor_count += 1
            if isinstance(sensor, dict) and "type" in sensor:
                types.add(f"sensor:{sensor['type']}")
    return actuator_count, sensor_count, tuple(sorted(types))


def _bot_capabilities(
    data: dict[str, Any],
    root_joint: str | None,
    component_types: tuple[str, ...],
    unsupported_joint_types: list[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(required_capabilities, blockers)`` for a bot manifest."""
    capabilities: list[str] = []
    blockers: list[str] = []
    if root_joint == "Free":
        capabilities.append(CAPABILITY_NATIVE_FLOATING_ROOT)
        blockers.append(f"native floating root ({CAPABILITY_NATIVE_FLOATING_ROOT})")
    if component_types:
        capabilities.append(CAPABILITY_ACTUATOR_SENSOR_COMPONENTS)
        blockers.append(
            f"actuator/sensor components ({CAPABILITY_ACTUATOR_SENSOR_COMPONENTS})"
        )
    if data.get("cycles"):
        capabilities.append(CAPABILITY_MECHANICAL_CYCLES)
        blockers.append(f"mechanical cycles ({CAPABILITY_MECHANICAL_CYCLES})")
    if unsupported_joint_types:
        blockers.append(f"unsupported joint types: {', '.join(unsupported_joint_types)}")
    return tuple(capabilities), tuple(blockers)


def _classify_bot(
    entrypoint: Path,
    data: dict[str, Any],
    bundle_root: Path,
    local_derivative: bool,
    licenses: tuple[str, ...],
    dependencies: tuple[DependencyEdge, ...],
) -> AssetEntry:
    is_recipe = "base" in data
    joints = _joints_of(data)
    joint_counts = _joint_counts(joints)
    links = _links_of(data)
    actuator_count, sensor_count, component_types = _components_of(links)
    capabilities, feature_blockers = _bot_capabilities(
        data,
        _root_joint_type(joints),
        component_types,
        sorted(set(joint_counts) - SUPPORTED_JOINT_TYPES),
    )
    if any(not edge.resolved for edge in dependencies):
        disposition = DISPOSITION_UNRESOLVED_DEPENDENCY
    elif is_recipe:
        disposition = DISPOSITION_RECIPE_CANDIDATE
    elif feature_blockers:
        disposition = DISPOSITION_UNSUPPORTED_FEATURES
    else:
        disposition = DISPOSITION_PROFILE_CANDIDATE
    required = list(capabilities)
    if is_recipe:
        required.append(CAPABILITY_RECIPE_COMPOSITION)
    return AssetEntry(
        entrypoint=entrypoint.relative_to(bundle_root).as_posix(),
        kind="bot",
        name=str(data.get("name", entrypoint.stem)),
        disposition=disposition,
        blockers=feature_blockers,
        required_capabilities=tuple(required),
        joint_counts=joint_counts,
        root_joint_type=_root_joint_type(joints),
        link_count=len(links),
        actuator_count=actuator_count,
        sensor_count=sensor_count,
        has_cycles=bool(data.get("cycles")),
        has_contact_overrides=bool(data.get("contactOverrides")),
        has_world_from_root=bool(data.get("worldFromRoot")),
        dependencies=dependencies,
        licenses=licenses,
        local_derivative=local_derivative,
    )


def _classify_prefab(
    entrypoint: Path,
    data: dict[str, Any],
    bundle_root: Path,
    local_derivative: bool,
    licenses: tuple[str, ...],
    dependencies: tuple[DependencyEdge, ...],
) -> AssetEntry:
    actors = data.get("actors") or {}
    rigid = actors.get("rigid") or []
    articulated = actors.get("articulated") or []
    capabilities: list[str] = []
    if rigid:
        capabilities.append(CAPABILITY_PREFAB_RIGID_ACTORS)
    if articulated:
        capabilities.append(CAPABILITY_PREFAB_ARTICULATED_ACTORS)
    if any(not edge.resolved for edge in dependencies):
        disposition = DISPOSITION_UNRESOLVED_DEPENDENCY
    else:
        disposition = DISPOSITION_UNSUPPORTED_FEATURES
    blockers = [f"prefab assembly pending ({', '.join(capabilities)})"]
    joints = _joints_of(data)
    return AssetEntry(
        entrypoint=entrypoint.relative_to(bundle_root).as_posix(),
        kind="prefab",
        name=str(data.get("name", entrypoint.stem)),
        disposition=disposition,
        blockers=tuple(blockers),
        required_capabilities=tuple(capabilities),
        joint_counts=_joint_counts(joints),
        root_joint_type=_root_joint_type(joints),
        link_count=len(_links_of(data)),
        dependencies=dependencies,
        licenses=licenses,
        local_derivative=local_derivative,
    )


def _classify_scene(
    entrypoint: Path,
    data: dict[str, Any],
    bundle_root: Path,
    local_derivative: bool,
    licenses: tuple[str, ...],
    dependencies: tuple[DependencyEdge, ...],
) -> AssetEntry:
    capabilities = [CAPABILITY_SCENE_ASSEMBLY, CAPABILITY_PREFAB_RIGID_ACTORS]
    if _joints_of(data):
        capabilities.append(CAPABILITY_PREFAB_ARTICULATED_ACTORS)
    if data.get("controllers"):
        capabilities.append(CAPABILITY_SCENE_CONTROLLERS)
    if any(not edge.resolved for edge in dependencies):
        disposition = DISPOSITION_UNRESOLVED_DEPENDENCY
    else:
        disposition = DISPOSITION_UNSUPPORTED_FEATURES
    joints = _joints_of(data)
    return AssetEntry(
        entrypoint=entrypoint.relative_to(bundle_root).as_posix(),
        kind="scene",
        name=str(data.get("name", entrypoint.stem)),
        disposition=disposition,
        blockers=(f"scene assembly pending ({CAPABILITY_SCENE_ASSEMBLY})",),
        required_capabilities=tuple(sorted(set(capabilities))),
        joint_counts=_joint_counts(joints),
        root_joint_type=_root_joint_type(joints),
        link_count=len(_links_of(data)),
        dependencies=dependencies,
        licenses=licenses,
        local_derivative=local_derivative,
    )


def _classify_controller(
    entrypoint: Path,
    data: dict[str, Any],
    bundle_root: Path,
    local_derivative: bool,
    licenses: tuple[str, ...],
    dependencies: tuple[DependencyEdge, ...],
) -> AssetEntry:
    params = data.get("poseControllerParams") or {}
    joint_tracking = params.get("jointTracking") or []
    return AssetEntry(
        entrypoint=entrypoint.relative_to(bundle_root).as_posix(),
        kind="controller",
        name=str(data.get("name", entrypoint.stem)),
        disposition=DISPOSITION_CONTROL_PROFILE,
        required_capabilities=(CAPABILITY_POSE_CONTROLLER_PROFILE,),
        joint_counts={"PoseTrackingJoint": len(joint_tracking)},
        dependencies=dependencies,
        licenses=licenses,
        local_derivative=local_derivative,
    )


_CLASSIFIERS = {
    BOT_SUFFIX: _classify_bot,
    PREFAB_SUFFIX: _classify_prefab,
    SCENE_SUFFIX: _classify_scene,
    CONTROLLER_SUFFIX: _classify_controller,
}


def _dependencies_of(
    manifest: Path, data: dict[str, Any], bundle_root: Path
) -> tuple[DependencyEdge, ...]:
    edges: list[DependencyEdge] = []
    source = manifest.relative_to(bundle_root).as_posix()
    for field_path, reference in _reference_fields(data):
        leaf = field_path.rsplit(".", 1)[-1]
        role = _ROLE_BY_FIELD[leaf]
        try:
            target, rule = resolve_reference(manifest, reference, bundle_root)
        except SuperdexAssetError as exc:
            edges.append(
                DependencyEdge(
                    source=source,
                    field=field_path,
                    role=role,
                    target=reference,
                    resolved=False,
                    note=exc.code,
                )
            )
            continue
        note = rule
        resolved = target.is_file()
        if resolved and _is_lfs_pointer(target):
            note = "lfs_pointer"
            resolved = False
        target_text = (
            target.relative_to(bundle_root).as_posix()
            if resolved or note != "manifest-relative"
            else reference
        )
        edges.append(
            DependencyEdge(
                source=source,
                field=field_path,
                role=role,
                target=target_text,
                resolved=resolved,
                note=note,
            )
        )
    return tuple(edges)


def _detect_recipe_cycles(parsed: dict[Path, dict[str, Any]], bundle_root: Path) -> None:
    """Raise ``recipe_cycle`` if the base/attachment graph contains a cycle."""
    bot_manifests = {
        path: data for path, data in parsed.items() if path.suffix == BOT_SUFFIX
    }
    visited: set[Path] = set()

    def recipe_references(data: dict[str, Any]) -> list[str]:
        values: list[str] = []
        if isinstance(data.get("base"), str):
            values.append(data["base"])
        for modification in data.get("modifications") or []:
            if isinstance(modification, dict):
                for operation in modification.values():
                    if isinstance(operation, dict):
                        path = operation.get("path")
                        if isinstance(path, str):
                            values.append(path)
        return values

    def visit(manifest: Path, chain: tuple[Path, ...]) -> None:
        if manifest in chain:
            cycle = " -> ".join(
                str(p.relative_to(bundle_root)) for p in (*chain, manifest)
            )
            raise SuperdexAssetError("recipe_cycle", cycle)
        if manifest in visited:
            return
        data = bot_manifests.get(manifest)
        if data is None:
            visited.add(manifest)
            return
        for reference in recipe_references(data):
            try:
                target, _ = resolve_reference(manifest, reference, bundle_root)
            except SuperdexAssetError:
                continue  # unresolvable refs are reported as dependency edges
            if target in bot_manifests:
                visit(target, (*chain, manifest))
        visited.add(manifest)

    for manifest in sorted(bot_manifests):
        visit(manifest, ())


def build_inventory(
    bundle_root: Path,
    provenance: Provenance,
    local_derivative_paths: frozenset[str] | set[str] = frozenset(),
) -> Inventory:
    """Survey one asset bundle and classify every entrypoint manifest."""
    bundle_root = Path(bundle_root).resolve()
    markers = sorted(
        path.relative_to(bundle_root).parent.as_posix()
        for path in bundle_root.rglob(ROOT_MARKER_NAME)
        if path.is_file()
    )
    if not markers:
        raise SuperdexAssetError("not_a_bundle", f"no {ROOT_MARKER_NAME} under {bundle_root}")

    manifests = sorted(
        path
        for path in bundle_root.rglob("*")
        if path.is_file() and path.suffix in ENTRYPOINT_SUFFIXES
    )
    parsed: dict[Path, dict[str, Any]] = {}
    for manifest in manifests:
        try:
            parsed[manifest] = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SuperdexAssetError(
                "schema_unexpected",
                f"{manifest.relative_to(bundle_root)}: {exc}",
            ) from exc

    _detect_recipe_cycles(parsed, bundle_root)

    entries: list[AssetEntry] = []
    for manifest, data in parsed.items():
        relative = manifest.relative_to(bundle_root).as_posix()
        dependencies = _dependencies_of(manifest, data, bundle_root)
        licenses = _collect_licenses(manifest, bundle_root)
        classifier = _CLASSIFIERS[manifest.suffix]
        entries.append(
            classifier(
                manifest,
                data,
                bundle_root,
                relative in local_derivative_paths,
                licenses,
                dependencies,
            )
        )

    tree_digest, file_count, total_bytes = hash_tree(bundle_root)
    return Inventory(
        provenance=provenance,
        entries=tuple(sorted(entries, key=lambda entry: entry.entrypoint)),
        file_count=file_count,
        total_bytes=total_bytes,
        tree_digest=tree_digest,
        root_markers=tuple(markers),
    )


def verify_bundle(inventory: Inventory, bundle_root: Path) -> tuple[list[str], list[str]]:
    """Check every dependency edge of an inventory against the bundle on disk.

    Returns ``(errors, warnings)``: errors are missing, escaped or LFS-pointer
    target files; warnings cover soft findings such as missing attribution.
    """
    bundle_root = Path(bundle_root).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    for entry in inventory.entries:
        for edge in entry.dependencies:
            if edge.resolved:
                target = bundle_root / edge.target
                if not target.is_file():
                    errors.append(f"{entry.entrypoint}: {edge.field} -> {edge.target}: missing")
                elif _is_lfs_pointer(target):
                    errors.append(
                        f"{entry.entrypoint}: {edge.field} -> {edge.target}: lfs_pointer"
                    )
            else:
                errors.append(
                    f"{entry.entrypoint}: {edge.field} -> {edge.target}: {edge.note}"
                )
        if not entry.licenses:
            warnings.append(
                f"{entry.entrypoint}: no LICENSE/NOTICE found in directory chain"
            )
    return errors, warnings
