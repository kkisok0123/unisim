# Superdex asset integration plan

Status: revised on 2026-09-14. The initial local copy and inventory exist; Git LFS
distribution and runtime qualification are planned. This document describes
implementation work; it does not establish support for additional assets.

## Objective and approach

Distribute the Superdex assets under `assets/superdex/` through Git LFS (Git
Large File Storage, interpreting the requested “git-LTS” as Git LFS). Preserve
the source directory structure and asset bytes. A contributor must be able to
download the assets from a pinned UniSim checkout without access to
`/home/pc829/UniFamily/project_superdex/`. Make the downloaded assets usable
through UniSim's Superdex adapter, including geometry, controls, state and reset.

Expand by simulator capability. Prove each capability with one representative
asset, then apply the same qualification checks to the remaining candidates.
Each stage should produce a usable result and fit into a small set of reviewable
pull requests.

The first milestone ends after stages 1–3: verified Git LFS asset downloads, an
unchanged FR3 running correctly, and a tested compatibility table for other bots.

## Contribution workflow

Follow the applicable practices in the
[UniLab contributing guide](https://github.com/Motphys/UniLab/blob/main/CONTRIBUTING.md)
(reviewed on 2026-09-14), using UniSim's own commands and ownership boundaries:

- Record the driving issue, intended PR base, scope and acceptance criteria
  before implementation. Work on a focused branch and use Conventional Commits.
- Use `uv run` for Python commands, `make sync` for the UniSim development
  environment, and focused checks followed by `make check` on the final head.
  UniLab's `make test-all` is specific to that repository; UniSim's full local
  gate is `make check`, plus `make package` for packaging changes.
- Keep responsibility-based module names, English code comments and docstrings,
  and temporary exports or backup files out of the source tree. Synchronize
  user-facing setup changes across `README.md`, `README_zh.md`,
  `CONTRIBUTING.md` and relevant guides when implementing the download workflow.
- Include the issue, exact validation commands and results, backend/platform
  impact, asset provenance and remaining limitations in every PR. Complete
  review and verify applicable CI for the final head before merging to `main`.

Before implementation, consult UniLab's linked
[architecture standard](https://github.com/Motphys/UniLab/blob/main/docs/sphinx/source/zh_CN/4-developer_guide/0-index.md)
for backend changes and
[collaboration workflow](https://github.com/Motphys/UniLab/blob/main/docs/sphinx/source/en/4-developer_guide/5-contributing_workflow.md)
for issue, milestone and ADR decisions. Those deeper references were unavailable
during this revision; no additional rules from them are assumed here. UniSim's
[AGENTS.md](../../AGENTS.md) and [contributing guide](../../CONTRIBUTING.md)
define this repository's validation and release requirements. Git LFS is the
requested distribution design for this plan, not a requirement attributed to
UniLab's contributing guide.

## Baseline and scope

The initial local survey found approximately 248 MiB across 891 asset files:

| Entrypoint format | Count |
| --- | ---: |
| `.superdex_bot` | 37 |
| `.mochi_prefab` | 15 |
| `.mochi_scene` | 2 |
| `.superdex_controller` | 1 |

These figures describe the surveyed local checkout at
`6b0541bd41adb39afe4e13fd6c45d9e9288f3021`. Earlier acquisition work pinned
`ed30ce16361329cbbed956173d6a4f5842815d24`. The local revision includes actuated
Wuji hands and removes some previously surveyed assets. Preserve this source
revision and its local derivatives as the migration baseline. The historical
pin is reference information; it does not replace that baseline.

UniSim commit `475f0e4` contains the initial copy, acquisition script and
SDK-free inventory tooling. The checked-in
[inventory report](../../docs/superdex-assets-inventory.md) records 892 files,
248.3 MiB and the same entrypoint counts above. Reuse its per-file hashes and
source provenance to verify the LFS migration. Inventory evidence establishes
dependency completeness, not any robot's runtime behavior. The repository has
no `.gitattributes` yet, so the existing copy does not establish LFS distribution.

The existing loader accepts native bots with a fixed HARD root and
fixed/revolute/prismatic joints. Native free roots, sensor/actuator components,
mechanical cycles and general scene assembly need additional adapter work.
See the [current Superdex profile](../../docs/superdex.md) and
[materialization implementation](../../src/unisim/backend/superdex/materialization.py).

The downloaded assets live in UniSim's repository-level `assets/superdex/`
directory as adapter qualification fixtures. UniSim owns asset resolution,
compatibility records, cold materialization, adapter lifecycle and validation.
UniLab continues to own task-facing robot assets, task configuration,
observations, rewards, training and policy evaluation. Cross-engine conversion
is a separate scope from native Superdex support. Keep this asset collection
outside the published Python package.

## Implementation stages

### 1. Migrate the existing assets to Git LFS and verify downloads

Deliverables:

- Add a root `.gitattributes` with explicit patterns scoped to
  `assets/superdex/` for binary and large payloads: meshes, collision data, CAD,
  textures and other binary formats discovered by the inventory. Each selected
  pattern uses `filter=lfs diff=lfs merge=lfs -text`. Keep small text bot/prefab/
  scene/controller definitions, `.superdex_root` markers, metadata, this plan,
  README files and LICENSE/NOTICE files in ordinary Git for reviewability.
- Audit all 892 inventoried files against the selected patterns so binary
  formats are not missed. Stage existing matching files through the LFS clean
  filter in a new commit; verify their Git blobs are pointers and their working
  files retain the recorded hashes. Merely adding attributes is insufficient.
- Preserve source revision, working-tree provenance, file hashes and local
  derivatives from the existing inventory. Record LFS OIDs and sizes for tracked
  payloads and the UniSim commit used for each download verification result.
- Reuse SDK-free inventory and verification tooling for bots, prefabs, scenes,
  controllers and dependencies. Add verification against the committed manifest
  without requiring the original checkout. Retain the copy script only as a
  maintainer import utility; contributors acquire payloads with `git lfs pull`.
- Preserve original directory structure, `.superdex_root` markers, collision
  geometry, render geometry, recipe dependencies and LICENSE/NOTICE files.
- Record required capabilities and a support status for each entry.
- Detect missing payloads, unexpanded LFS pointers, hash/size mismatches and
  references escaping the asset root before SDK loading. Give an actionable
  download command for missing assets; do not attempt an implicit download.
- Upload the LFS objects with the implementation commit and verify access from
  the documented remote in a fresh clone with an empty LFS cache. Keep ordinary
  Git clone access and LFS object access as separate checks.
- Verify that all required references resolve inside the downloaded tree and
  loading does not require the source checkout or network access after download.
- Keep downloads out of imports, backend construction and the normal test suite.

Required destination layout (unchanged by LFS):

```text
/home/pc829/UniFamily/unisim/assets/superdex/
  bots/
  prefabs/
  benchmarks/
  cube/
  test/
  ...
```

For example, source `assets/bots/arms/fr3_v2/fr3_v2.superdex_bot` maps to
`/home/pc829/UniFamily/unisim/assets/superdex/bots/arms/fr3_v2/fr3_v2.superdex_bot`.
There is no additional `assets/assets/`, `superdex/superdex/` or revision
directory.

Planned contributor download workflow, after the LFS migration is published
(POSIX shell; install Git LFS first):

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/unilabsim/unisim.git
cd unisim
git lfs install --local
# Replace the placeholder with the full commit published for qualification.
GIT_LFS_SKIP_SMUDGE=1 git checkout --detach <asset-commit-sha>
git lfs pull origin --include="assets/superdex/**" --exclude=""
git lfs fsck
export SUPERDEX_ASSETS_PATH="$PWD/assets/superdex"
```

The commands use the current checkout's LFS pointers to select the payload
versions; see the [Git LFS pull documentation](https://github.com/git-lfs/git-lfs/blob/main/docs/man/git-lfs-pull.adoc).
Publish a concrete full commit in the implemented loading guide. For an existing
checkout at that commit, run the install, pull and verification steps from its
repository root. Add equivalent PowerShell setup instructions when implementing
the guide. Run the manifest verifier after `git lfs fsck`; LFS object integrity
alone does not establish dependency completeness. Keep reports under `docs/`.

The current asset commit already contains ordinary Git payload blobs. Converting
the current tree in a new commit leaves those blobs in history, so it does not
make the historical clone small. Any history rewrite is a separate maintainer
decision; it is not part of this stage. Git LFS also does not install the SDK.

**Done when:** a fresh clone downloads every required payload through Git LFS,
all asset paths and hydrated file hashes match the committed inventory, no
required pointers remain unexpanded, and dependencies resolve locally. The
offline core suite must also pass with LFS downloads disabled.

### 2. Prove one unchanged robot: FR3 v2

Use the downloaded
`assets/superdex/bots/arms/fr3_v2/fr3_v2.superdex_bot`
through the existing native loader and
`create_backend("superdex", SceneCfg(bot_path), ...)`.

- Verify body/joint inventories, initial pose, control ordering, effort limits
  and collision geometry against the original asset and direct SDK loading.
- Exercise bounded movement, reset, two independent environments and cleanup.
- Supply explicit effort limits when required and record them in the fixture.
- Fix demonstrated adapter issues while preserving the original asset bytes.
- Keep the existing regression that modifies mass metadata as a separate test.
- Provide one runnable loading example and a repeatable qualification result.

**Done when:** the unchanged FR3 passes the physics and lifecycle checks below
through UniSim, with visual verification reported separately.

### 3. Expand compatible robots and compositions

- Qualify other fixed-base arms and simple bots using a parameterized runner.
- Resolve Mod Bot recipes through the SDK and inspect the compiled robot,
  accounting for every base, attachment, link, joint and geometry dependency.
- Apply the same checks as FR3 to every supported candidate.
- Record precise blockers for candidates requiring later capabilities. Preserve
  unsupported components in the inventory instead of dropping them to pass.

**Done when:** every candidate in this stage either passes qualification or has
a specific blocker, and the first milestone's compatibility table is available.

### 4. Support native floating roots

Extend native bot state translation for free-root robots. Begin with one hand
whose remaining features fit the adapter, then expand to other candidates.

- Verify root position and quaternion conventions, linear/angular velocity
  frames, joint offsets and state dimensions against UniSim's existing contract.
- Test initial state, movement, state round trips and full/selective reset.
- Check two-environment isolation and cleanup.

**Done when:** a representative standalone hand works with correct root and
joint state, and existing fixed-base behavior remains covered.

### 5. Add rigid objects and prefab assembly

Extend scene ownership beyond one articulation. Start with a sphere or block
beside a qualified robot, then add a peg/board or cup fixture.

- Resolve nested prefab dependencies and track every actor, collision shape,
  transform and initial state.
- Add ownership and cleanup for independent rigid objects.
- Define how complete scene state is captured and restored. Extend the public
  contract only where existing interfaces cannot express the required behavior.
- Exercise robot–object contact and reset after objects have moved.

**Done when:** actor inventories match, representative contact works, whole-scene
reset restores moved objects, and selective reset leaves other environments
unchanged.

### 6. Add actuator and sensor components

Use the local actuated Wuji hands as a concrete target after native floating-root
support is available.

- Audit component registration and reject missing or silently skipped types.
- Define command units, gains, effort limits, update frequency and execution
  order. Preserve the adapter's control semantics.
- Define sensor identity, output frames and shapes, and reset of component state.
- Account for all 20 actuators and 5 sensors per hand in the surveyed assets.
- Test bounded target response, saturation, sensor response and repeatable reset.

**Done when:** all expected components instantiate and behave correctly. An
integrated hand–object fixture also requires stage 5.

### 7. Load complete native scenes

- Add `.mochi_scene` dispatch after scene assembly is available.
- Define explicit precedence for authored gravity, timestep and solver settings
  versus caller configuration; reject unresolved conflicts.
- Bind robot actions and retain all scene actors and supported components.
- Qualify benchmark scenes individually against direct SDK execution.

**Done when:** actor inventories, effective settings, controls and whole-scene
reset match the intended SDK behavior. Benchmark task rewards and training
remain with UniLab.

### 8. Extend advanced capabilities individually

Create separate work packages for closed mechanisms, spherical joints, multiple
articulations, specialized controllers and soft bodies. Add archive loading and
offline CAD/URDF conversion when a concrete use case requires them.

Select each package using the stage 1 inventory and user priorities. Keep these
features outside the first milestone's critical path.

**Done when:** each promoted capability has a representative fixture, explicit
state/control/reset semantics, repeatable tests and documented limits. Other
assets retain precise unsupported statuses.

## Dependencies and first pull requests

Stages 1–3 form the initial sequence. Stage 4 extends robot state support; stage 5
can build on the qualified fixed-base route independently of floating roots.
The Wuji target in stage 6 depends on stage 4, and its object interaction checks
also depend on stage 5. Stage 7 depends on stage 5 and any component capabilities
required by its selected scenes. Stage 8 is scheduled per capability.

The first three pull requests are:

1. **Git LFS distribution and inventory verification:** migrate existing payloads,
   document explicit downloads, preserve provenance, verify fresh-clone hashes
   and dependencies, and test diagnostics with unexpanded pointers.
2. **Untouched FR3 qualification:** focused adapter fixes, runtime checks and a
   runnable loading example.
3. **Reusable bot qualification:** parameterized checks and a compatibility
   table for direct bots and recipes.

## Validation and reporting

Track dependency completeness, loading, physics verification and visual
verification separately. A successful load or short finite run alone does not
qualify an asset.

| Check | Required evidence |
| --- | --- |
| Provenance and dependencies | Source revision and local changes recorded; hydrated file hashes match; required references resolve inside `assets/superdex/`. |
| Git LFS distribution | Fresh clone at the recorded commit retrieves all required LFS objects; OIDs/sizes and manifest hashes match; no unresolved pointers remain. |
| Offline boundary | Core/import tests pass with downloads disabled; hydrated fixtures load without network access or the original checkout. |
| Structure | Authored/compiled body, joint, actor and component inventories agree. |
| State | Correct dimensions, ordering, frames and finite initial values. |
| Control | Small bounded commands affect the intended joints; limits are enforced. |
| Stability | At least 1,000 steps under a recorded timestep/control profile without invalid state or unexplained divergence. |
| Geometry and contact | Expected colliders instantiate and a representative contact probe behaves correctly. |
| Reset | State round trips and whole-scene/selective reset restore the intended state. |
| Isolation | Stepping or resetting one environment leaves another unchanged. |
| Lifecycle | Repeated creation, cleanup and recreation succeed. |
| Visuals | Representative captures show complete geometry, correct scale and link alignment. |

Use direct Superdex execution with the same inputs as the adapter reference.
Record the asset source revision, UniSim/LFS pointer commit, manifest digest,
SDK version, platform, precision,
timestep, control profile and numerical tolerances with every result. Separate
checks performed through UniSim from any checks requiring a direct SDK fixture.

Run native viewer checks in serial mode with one environment; validate headless
batch execution separately. Keep missing-runtime diagnostics and unsupported
feature rejection covered by tests.

For implementation changes, run focused tests and `make check`. Run `make package`
when packaging changes and `uv lock --check` when dependency metadata changes.
Asset/runtime tests may be optional in the normal suite, but an explicit
qualification run must fail if its required assets or runtime are missing.

Keep the existing three cross-platform CI test jobs and package dependency gate.
Core CI should explicitly skip LFS smudging and use small synthetic fixtures for
inventory and pointer-diagnostic tests. Put full download/SDK qualification in
an explicit opt-in job or workflow; pull LFS payloads before that run and fail
on acquisition or verification errors. Record both its commit and results in
the PR, separately from normal CI. Do not add asset or SDK downloads to `make check`.

Keep NumPy as the sole base dependency and preserve lazy SDK imports. Inspect
package contents when packaging changes to ensure the repository-level asset
collection, including LFS pointer stubs, does not enter `unisim-core`
distributions. Caches and
native SDKs remain outside the base package.

## Deliverable for each completed stage

Leave a working example, repeatable tests and a compatibility report listing
asset paths, tested profiles, separate verification results and remaining
blockers. Update the [Superdex guide](../../docs/superdex.md) alongside implementation.
Public contract or factory changes also require focused documentation and a
`CHANGELOG.md` entry; keep both READMEs synchronized if they change.
