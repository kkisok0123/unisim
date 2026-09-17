# Superdex asset integration plan

Status: revised on 2026-09-17. Stages 1–7 are complete for their recorded
profiles. Stage 7 qualifies unchanged Cart Pole and Half Cheetah `.mochi_scene`
files, including authored contact filters and Half Cheetah rest springs.
Stage 6's floating OSC SDK limitation remains. Stage 5 and Stage 7 native
renderer smoke passed; manual visual inspection remains outstanding.
Qualification applies only to the recorded fixtures, not every asset in the bundle.

## Objective and approach

Keep the existing Superdex assets local under `assets/superdex/` and preserve
the existing `assets/` ignore rule. Preserve the source directory structure and
asset bytes. Once the local copy is available, running it must not require access
to `/home/pc829/UniFamily/project_superdex/`. Make those assets usable through
UniSim's Superdex adapter, including geometry, controls, state and reset. Asset
distribution is outside this plan; a Git clone does not provide the ignored files.

Start by making the unchanged FR3 visible and controllable through UniSim so
adapter changes can be inspected as they are developed. Then prove its numerical
and lifecycle behavior. Expand by simulator capability: implement or fix the
adapter path, inspect one representative asset in the viewer, run qualification
checks, and only then apply those checks to other candidates. Each stage should
produce a usable result and fit into a small set of reviewable pull requests.

The first milestone ends after stages 1–3: the existing asset baseline, a runnable
FR3 viewer demonstration, a qualified FR3 adapter path, and a tested compatibility
table for other bots. Stage 2 is split into visualization (2A, complete) and full
adapter qualification (2B, complete).

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
  `CONTRIBUTING.md` and relevant guides when documenting local asset/viewer setup.
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
define this repository's validation and release requirements.

## Baseline and scope

The local bundle contains approximately 248 MiB across 888 files:

| Entrypoint format | Count |
| --- | ---: |
| `.superdex_bot` | 35 |
| `.mochi_prefab` | 15 |
| `.mochi_scene` | 2 |
| `.superdex_controller` | 1 |

Source provenance is recorded in the inventory. The historical upstream pin
`ed30ce16361329cbbed956173d6a4f5842815d24` is reference information.

UniSim commit `475f0e4` contains the initial copy, acquisition script and
SDK-free inventory tooling. The checked-in
[inventory report](../../docs/superdex-assets-inventory.md) records 888 files,
248.3 MiB and the same entrypoint counts above. Reuse its per-file hashes and
source provenance for subsequent local asset verification. Inventory
evidence establishes dependency completeness, not any robot's runtime behavior.
This historical import record does not change the current decision to keep
`assets/` ignored; do not force-add payloads or rewrite history as part of this plan.

The current loader accepts native bots with HARD or FREE roots and
fixed/revolute/prismatic child joints. Stage 5 adds independent rigid actors
and nested `.mochi_prefab` fragments beside one native bot. Stage 6 supports the
qualified built-in camera/controller profiles. Stage 7 adds complete native
scenes containing one fixed/hinge/slide articulation and rigid actors. Custom
components, mechanical cycles and multiple articulations still need additional work.
See the [current Superdex profile](../../docs/superdex.md) and
[materialization implementation](../../src/unisim/backend/superdex/materialization.py).

The adapter already connects native interactive playback to the Superdex
Polyscope viewer through `run_playback`. It requires
`superdex_execution_mode="serial"`, `num_envs=1`, and the optional SDK/viewer
runtime. Reuse and validate this path with the repository FR3 asset. Its presence
in the code does not establish that the unchanged FR3 has passed visual checks.

The local assets live in UniSim's repository-level `assets/superdex/`
directory as adapter qualification fixtures. UniSim owns asset resolution,
compatibility records, cold materialization, adapter lifecycle and validation.
UniLab continues to own task-facing robot assets, task configuration,
observations, rewards, training and policy evaluation. Cross-engine conversion
is a separate scope from native Superdex support. Keep this asset collection
outside the published Python package.

## Implementation stages

### 1. Establish the local asset baseline — complete

The initial copy, provenance record and SDK-free inventory are already present.
Retain them as the starting point; stage 2A builds on them.

- Keep assets local and ignored at `assets/superdex/`, preserving directories,
  `.superdex_root` markers, collision and render geometry, recipe dependencies,
  local derivatives and LICENSE/NOTICE files.
- Reuse the recorded inventory and per-file hashes when checking the local copy.
  The copy script remains a maintainer import utility; running an example must
  not require the original source checkout.
- Keep asset acquisition out of imports, backend construction and normal tests.
  Installing the Python package does not install this collection or the SDK.

The FR3 entrypoint is
`assets/superdex/bots/arms/fr3_v2/fr3_v2.superdex_bot`; there is no extra
`assets/assets/`, `superdex/superdex/` or revision directory.

From the repository root, with the local assets already present, select the
fixture root with:

```bash
export SUPERDEX_ASSETS_PATH="$PWD/assets/superdex"
```

Verify local file hashes and dependency completeness against the recorded
inventory without using the source checkout. Document the local asset prerequisite
and how to configure its path. If assets are absent, stop with a clear setup
diagnostic; do not imply that cloning UniSim supplies them. Keep provenance and
qualification reports under `docs/`; no asset publication is required for the
next stage.

**Completed result:** the local asset copy and inventory are available for viewer
and adapter development. Runtime qualification remains in the following stages.

### 2A. Visualize the unchanged FR3 through the existing adapter — complete

Built as `scripts/superdex_bot_viewer.py` with a numerical run report
under `docs/superdex-fr3-viewer/`; see the FR3 viewer demonstration section
in `docs/superdex.md`. Asset pre-verification uses
`verify_asset_bundle()` against the recorded inventory JSON. Demonstrated
control profile: joint-position targets in radians, pre-step PD converter,
explicit 87/87/87/87/12/12/12 N·m effort limits; zero effort does not hold
the pose. Proceed to stage 2B.

Build one standalone UniSim example around
`assets/superdex/bots/arms/fr3_v2/fr3_v2.superdex_bot`.
Load it with `create_backend("superdex", SceneCfg(bot_path), ...)` and use the
adapter's native `run_playback` path. The example must work without UniLab.

- Document the Superdex SDK, Polyscope and graphical-session prerequisites.
  Use `superdex_execution_mode="serial"`, `num_envs=1`, interactive playback
  and no video recording. Keep viewer imports optional.
- Validate asset paths and dependencies before SDK loading. Report missing files,
  hash/size mismatches and references escaping the asset root clearly; do not
  download files implicitly. Add recorded-manifest verification where needed.
- First display the initial pose, frame the camera and inspect complete geometry,
  scale and link alignment. Render the actual Superdex scene simulated by UniSim.
- Add a repeatable bounded movement demonstration using UniSim's control API.
  Record control ordering, command units, timestep and explicit effort limits.
  Do not assume that zero effort holds the initial pose.
- Demonstrate reset through UniSim and show the restored state in the viewer.
  Make initial-pose inspection, movement and reset distinct, observable phases.
- Fix demonstrated adapter or viewer problems in asset materialization, scene
  binding, stepping, state synchronization, camera framing and cleanup. Preserve
  the original asset bytes and reject unsupported features explicitly.
- Supply one documented `uv run` command and a numerical run report. Verify
  that closing the window releases the viewer and backend, including error paths.
  Retain focused regression coverage for any adapter/viewer fixes.

**Done when:** one command opens the unchanged FR3 through UniSim, lets the user
inspect its initial geometry, demonstrates bounded movement and visible reset,
and exits cleanly. Record this as visual verification; it does not yet qualify
the robot's physics or batch behavior.

### 2B. Complete and qualify the FR3 adapter path — complete

Built as `scripts/superdex_fr3_qualify.py` with the recorded report under
`docs/superdex-fr3-qualification/` and the pytest regression subset in
`tests/test_superdex_fr3_qualification.py` (opt-in via
`SUPERDEX_ASSETS_PATH`); see the FR3 qualification section in
`docs/superdex.md`. All adapter-side results were checked against a direct
SDK scene driven with identical inputs; every adapter-vs-SDK deviation was
exactly 0.0 and no adapter defect was demonstrated, so no adapter fix was
needed. The pre-existing mass-metadata regression
(`test_native_fr3_when_registered_assets_are_available`) remains a separate
test. Proceed to stage 3.

Use the local
`assets/superdex/bots/arms/fr3_v2/fr3_v2.superdex_bot`
through the existing native loader and
`create_backend("superdex", SceneCfg(bot_path), ...)`.

- Verify body/joint inventories, initial pose, control ordering, effort limits
  and collision geometry against the original asset and direct SDK loading.
- Exercise bounded movement, reset, two independent environments and cleanup.
- Supply explicit effort limits when required and record them in the fixture.
- Fix demonstrated adapter issues while preserving the original asset bytes.
- Keep the existing regression that modifies mass metadata as a separate test.
- Reuse the stage 2A example and record repeatable numerical qualification.
- Use direct SDK execution with identical inputs to distinguish asset/SDK
  behavior from adapter translation issues. Add focused regression tests for
  each demonstrated adapter defect.

**Done when:** the unchanged FR3 passes the physics and lifecycle checks below
through UniSim, with visual verification reported separately.

### 3. Reuse the adapter and viewer for compatible robots and compositions — complete

Built as `scripts/superdex_bot_qualify.py` with `scripts/superdex_bot_profiles.py`
(shared candidate/control registry), reports plus the compatibility table under
`docs/superdex-bots-qualification/`, and the pytest subset in
`tests/test_superdex_bot_qualification.py` (opt-in via
`SUPERDEX_ASSETS_PATH`); the stage-2A viewer gained `--bot <key>`. See the
compatible-bots section in `docs/superdex.md`. Eleven bots qualified with every
adapter-vs-SDK deviation exactly 0.0; no adapter extension was needed because
the SDK compiles recipes into the native loader's supported profile. Every
remaining bot has a precise recorded blocker. Stages 4 and 5 extend this baseline below.

- Qualify other fixed-base arms and simple bots using a parameterized runner.
- Resolve Mod Bot recipes through the SDK and inspect the compiled robot,
  accounting for every base, attachment, link, joint and geometry dependency.
- Extend adapter recipe resolution and compiled-structure validation where
  needed; do not bypass UniSim to mark a candidate supported.
- Reuse the stage 2A viewer example with a selectable asset path and explicit
  per-asset control profile. Inspect geometry, movement and reset before promotion.
- Apply the same numerical and lifecycle checks as FR3 to every supported candidate.
- Record precise blockers for candidates requiring later capabilities. Preserve
  unsupported components in the inventory instead of dropping them to pass.

**Done when:** every candidate in this stage either passes qualification or has
a specific blocker, and the first milestone's compatibility table is available.

### 4. Extend the adapter for native floating roots — complete

Extend native bot state translation for free-root robots. Begin with one hand
whose remaining features fit the adapter, then expand to other candidates.

- Verify root position and quaternion conventions, linear/angular velocity
  frames, joint offsets and state dimensions against UniSim's existing contract.
- Test initial state, movement, state round trips and full/selective reset.
- Check two-environment isolation and cleanup.

**Done when:** a representative standalone hand works with correct root and
joint state, and existing fixed-base behavior remains covered.

### 5. Extend the adapter for rigid objects and prefab assembly — complete

**Completed result:** `SceneCfg.fragment_files` now composes rigid
`.mochi_prefab` files, including nested dependencies and transforms, beside a
native bot. Dynamic objects append canonical free-body coordinates to the
existing `qpos`/`qvel` state; robot controls keep their indices. Body queries,
object root layouts, forces, full/selective reset and scene-owned cleanup use
the existing public interfaces. Static fixtures retain authored transforms.
State round trips restore kinematics, not hidden solver history. Unsupported
prefab components and scene settings are rejected rather than dropped.

`scripts/superdex_prefab_qualify.py` qualifies unchanged FR3 plus a sphere and
nine-hole peg board: 11 standalone rigid actors, including nine dynamic pegs,
one sphere and one static board. Both serial and batch execution passed 1,000
steps at 0.002 s with two environments and zero measured deviation from direct
SDK execution. Actor inventories, robot–sphere contact, moved-object state
round trips, full/selective reset, isolation and cleanup/recreation passed.
Tests also cover partial-instantiation failure cleanup. No additional bot or
prefab is promoted by these results.

```bash
SUPERDEX_ASSETS_PATH="$PWD/assets/superdex" uv run --no-sync scripts/superdex_prefab_qualify.py
SUPERDEX_ASSETS_PATH="$PWD/assets/superdex" uv run --no-sync pytest -q tests/test_superdex_prefabs.py
SUPERDEX_ASSETS_PATH="$PWD/assets/superdex" uv run --no-sync scripts/superdex_prefab_qualify.py --viewer
```

The separate `--viewer --frames 420` smoke completed initial pose, contact and
whole-scene reset with zero reset error and clean teardown, without saving
images. **Manual visual inspection remains outstanding.** The focused suite
passed 15 tests; `make check` passed Ruff and 329 tests with 19 skips;
`make package` built the sdist and wheel with the ignored assets excluded.
See the [Stage 5 report](../../docs/superdex-prefab-qualification/README.md),
[numerical results](../../docs/superdex-prefab-qualification/report.json),
[viewer smoke](../../docs/superdex-prefab-qualification/viewer.json) and
[current guide](../../docs/superdex.md). These reports preserve the source and
asset digests of their original runs; later documentation-only inventory
refreshes do not rewrite that evidence.

The stage's acceptance criteria are retained below:

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

### 6. Built-in cameras/controllers and universal qualification - complete

Support link-attached `SENSOR_CAMERA` settings and world poses, plus explicit
`BASIC_JSC_PD`, `BASIC_OSC_PD` and `MOCHI_ARTICULATED_POSE` controller execution.
Use one controller type at a time, with independent instances per environment,
substep evaluation, selective reset and complete cleanup. Keep these APIs
SuperDex-specific; image rendering and full scene dispatch remain separate.

Include ordinary `wuji_hand2_beta1` torque qualification.
The universal runner accepts `--bots`/`--all`, rigid scene fragments, explicit
effort limits and an optional `--controller-config` using SDK settings.

Acceptance evidence is recorded under `docs/superdex-component-qualification/`
and in `tests/test_superdex_component_qualification.py`: compare mounted camera
metadata/poses and all three controllers with direct SDK execution; test both
execution modes, fixed/floating roots, effort clipping, reset, isolation and
lifecycle. Unsupported components must fail explicitly, while models without
cameras report that check as skipped. The current SDK's floating OSC
initialization has a bot/actor effort-index mismatch; verify and report that
blocker instead of claiming floating OSC support.

**Done when:** the supported profiles pass qualification and documentation
records the exact coverage and remaining SDK limitations. A complete scene
or arbitrary custom component is not implied by this stage.

### 7. Extend adapter dispatch to complete native scenes — complete

`SceneCfg.model_file` now dispatches `.mochi_scene` through the native SDK
prefab loader. The profile supports one root-file articulation with fixed,
revolute and prismatic joints (including a prismatic first joint), standalone
rigid actors, nested rigid prefabs and additional rigid `fragment_files`.
Audits reject unsupported content before loading; asset bytes remain unchanged.

- Callers provide ordered `superdex_controlled_joints` and matching finite
  positive `superdex_effort_limits`. Inputs are physical efforts. Cart Pole
  controls `Cart` (limit 3); Half Cheetah controls six leg joints (limits
  120, 90, 60, 120, 60, 30). These are tooling profiles, not loader special cases.
- Authored gravity and solver settings take effect; absent settings retain SDK
  defaults. `sim_dt` belongs to the caller. Timestep fields and conflicting
  nested settings fail explicitly. Preserve negative-Y SDK gravity in these
  scenes; do not add a ground plane or convert coordinate systems.
- Preserve actor names, geometry, transforms, contact filters and Half Cheetah's
  authored joint-tracking rest springs. Full/selective reset restores initial
  state and spring targets without changing controller ownership.
- Whole-scene body/state/force interfaces and serial/batch execution are reused.
  Multiple articulations, soft bodies, scene camera/plugin components, non-rigid
  nested prefabs and advanced mechanisms remain unsupported.

Both unchanged benchmarks passed 1,000 steps at 0.002 s with two environments
in each of serial and batch execution. State and body-position deviation from
independently loaded SDK scenes was zero. Checks cover initial state, actor
inventories, effective settings, action routing, clipping, moved-state round
trips, reset, isolation, controller preservation and cleanup/recreation.
Synthetic regressions exercise authored settings, rigid contact/filtering,
additional fragments and partial-instantiation cleanup.

```bash
SUPERDEX_ASSETS_PATH="$PWD/assets/superdex" uv run --no-sync scripts/superdex_scene_qualify.py
SUPERDEX_ASSETS_PATH="$PWD/assets/superdex" uv run --no-sync pytest -q tests/test_superdex_scenes.py
uv run --no-sync scripts/superdex_scene_qualify.py --viewer --frames 240
```

Each scene completed 240 offscreen frames covering initial pose, controlled
motion and reset, with zero reset error. Manual visual inspection remains
outstanding and is separate from numerical qualification. See the
[Stage 7 report](../../docs/superdex-scene-qualification/README.md),
[numerical evidence](../../docs/superdex-scene-qualification/report.json) and
[viewer smoke](../../docs/superdex-scene-qualification/viewer.json).

**Done when:** actor inventories, effective settings, controls and whole-scene
reset match direct SDK behavior for both selected benchmarks. This numerical
and lifecycle gate passed. Benchmark rewards and training remain with UniLab.

### 8. Extend advanced adapter capabilities individually

Create separate work packages for closed mechanisms, spherical joints, multiple
articulations, specialized controllers and soft bodies. Add archive loading and
offline CAD/URDF conversion when a concrete use case requires them.

Select each package using the stage 1 inventory and user priorities. Keep these
features outside the first milestone's critical path.

**Done when:** each promoted capability has a representative fixture, explicit
state/control/reset semantics, repeatable tests and documented limits. Other
assets retain precise unsupported statuses.

## Adapter development process for stages 4–8

For each new capability, follow the same process:

1. Select one representative asset from the inventory and record its required
   features and current blockers. Inspect direct SDK behavior as the reference.
2. Define how loading, scene ownership, state, controls and reset map to UniSim.
   Keep validation and materialization on cold paths. Extend the public contract
   only if its existing interfaces cannot express the capability.
3. Implement the smallest adapter extension needed for that asset. Preserve
   original asset bytes, account for every component and reject unsupported
   features instead of silently dropping them.
4. Extend the viewer example to inspect the new capability: root movement,
   robot–object contact, actuated motion or complete-scene behavior as applicable.
   Provide numerical readouts for sensor/state details that geometry cannot show.
5. Run numerical and lifecycle qualification, including headless execution,
   selective reset and environment isolation. Keep visual results separate.
6. Publish the example, regression tests and compatibility status before applying
   the capability to other assets. A visual pass alone does not grant support.

## Dependencies and next pull requests

The immediate sequence is **stage 1 complete → stage 2A visualization
complete → stage 2B FR3 adapter qualification complete → stage 3 compatible
robots complete → stage 4 native floating roots complete → stage 5 rigid
prefabs qualified → stage 6 built-in cameras/controllers qualified → stage 7
complete native scenes qualified**. Manual visual inspection of the stage-5 fixture
remains a separate follow-up.

Stage 4 supplies floating robot state; stage 5 supplies independent rigid
objects and whole-scene state/reset on the qualified fixed-base route. Stage 6
reuses these capabilities for component qualification. Stage 7 reuses
stage 5 state/ownership and qualifies both selected scenes, including native rest springs.
Stage 8 is scheduled per capability. All extensions reuse the viewer workflow.

The first milestone's completed work packages were:

1. **FR3 visualization through UniSim (2A): complete** — see
   `scripts/superdex_bot_viewer.py` and `docs/superdex-fr3-viewer/`.
2. **Untouched FR3 adapter qualification (2B): complete** — numerical,
   contact, reset, isolation and lifecycle checks against direct SDK
   execution; see `scripts/superdex_fr3_qualify.py`,
   `docs/superdex-fr3-qualification/` and
   `tests/test_superdex_fr3_qualification.py`. No adapter fix was required.
3. **Reusable bot visualization and qualification (3): complete** — a
   selectable-asset viewer example (`--bot <key>`), parameterized checks over
   shared profiles, no recipe adapter work needed (the SDK compiles recipes
   inside the native profile) and a compatibility table for direct bots and
   recipes; see `scripts/superdex_bot_qualify.py`,
   `scripts/superdex_bot_profiles.py`, `docs/superdex-bots-qualification/`
   and `tests/test_superdex_bot_qualification.py`.

## Validation and reporting

Track dependency completeness, loading, physics verification and visual
verification separately. A successful load or short finite run alone does not
qualify an asset.

| Check | Required evidence |
| --- | --- |
| Provenance and dependencies | Source revision and local changes recorded; file hashes match; required references resolve inside `assets/superdex/`. |
| Local asset reproducibility | The configured local asset tree matches recorded file sizes and hashes, and all required dependencies resolve without the source checkout. |
| Offline boundary | Core/import tests pass with downloads disabled; local fixtures load without network access or the original checkout. |
| Structure | Authored/compiled body, joint, actor and component inventories agree. |
| State | Correct dimensions, ordering, frames and finite initial values. |
| Control | Small bounded commands affect the intended joints; limits are enforced. |
| Stability | At least 1,000 steps under a recorded timestep/control profile without invalid state or unexplained divergence. |
| Geometry and contact | Expected colliders instantiate and a representative contact probe behaves correctly. |
| Reset | State round trips and whole-scene/selective reset restore the intended state. |
| Isolation | Stepping or resetting one environment leaves another unchanged. |
| Lifecycle | Repeated creation, cleanup and recreation succeed. |
| Visuals | Native UniSim playback shows complete geometry, correct scale and link alignment, bounded movement and visible reset; numerical results are recorded without saving images. |

Use direct Superdex execution with the same inputs as the adapter reference.
Record the asset source revision and local derivatives, UniSim code commit, manifest digest,
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
Core CI should use small synthetic fixtures for inventory, asset diagnostics and
viewer lifecycle tests. Put full asset/SDK qualification in an explicit opt-in
job or workflow with explicitly provisioned local assets and runtime; verify the
asset tree before that run and fail on missing files or verification errors.
Record its code commit, asset manifest digest and results separately
from normal CI. Document graphical viewer checks separately from headless
qualification. Do not add asset or SDK downloads to `make check`.

Keep NumPy as the sole base dependency and preserve lazy SDK/viewer imports.
Inspect package contents when packaging changes to ensure the repository-level
asset collection does not enter `unisim-core` distributions. Caches and native
SDKs remain outside the base package.

## Deliverable for each completed stage

Leave a working example, repeatable tests and a compatibility report listing
asset paths, tested profiles, separate verification results and remaining
blockers. Update the [Superdex guide](../../docs/superdex.md) alongside implementation.
Public contract or factory changes also require focused documentation and a
`CHANGELOG.md` entry; keep both READMEs synchronized if they change.
