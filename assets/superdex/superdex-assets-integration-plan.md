# Superdex asset integration plan

Status: planned. Prepared on 2026-09-14. This document describes implementation
work; it does not establish support for additional assets.

## Objective and approach

Make assets from `/home/pc829/UniFamily/project_superdex` usable through UniSim's
Superdex adapter, including their geometry, controls, state and reset behavior.

Expand by simulator capability. Prove each capability with one representative
asset, then apply the same qualification checks to the remaining candidates.
Each stage should produce a usable result.

The first milestone ends after stages 1–3: a reproducible bundle, an unchanged
FR3 running correctly, and a tested compatibility table for other bots.

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
Wuji hands and removes some previously surveyed assets. Stage 1 must reconcile
these baselines and regenerate the inventory for the selected revision.

The current UniSim worktree starts at
`1ef3bb6b04a2cd76f20be72c49cd4d15f528e51b`. It includes the native bot loader but
does not contain the earlier draft acquisition script, inventory module or
integration plans. The earlier assessment ran 15 inventory tests successfully
in the previous working tree; that result does not certify this worktree or any
robot's runtime behavior. Recover and review reusable work during stage 1 if
available, otherwise implement it here.

The existing loader accepts native bots with a fixed HARD root and
fixed/revolute/prismatic joints. Native free roots, sensor/actuator components,
mechanical cycles and general scene assembly need additional adapter work.
See the [current Superdex profile](superdex.md) and
[materialization implementation](../src/unisim/backend/superdex/materialization.py).

UniSim owns asset resolution, compatibility records, cold materialization,
adapter lifecycle and validation. Large asset bundles remain in an external
store. UniLab continues to own asset deployment for tasks, observations,
rewards, training and policy evaluation. Cross-engine conversion is a separate
scope from native Superdex support.

## Implementation stages

### 1. Establish a reproducible asset bundle

Deliverables:

- Record the exact source revision, file hashes and local derivatives. Preserve
  the distinction between upstream assets and locally modified assets.
- Finish explicit acquisition and SDK-free inventory tooling. Cover bots,
  prefabs, scenes, controllers and their referenced dependencies.
- Preserve original directory structure, `.superdex_root` markers, collision
  geometry, render geometry, recipe dependencies and LICENSE/NOTICE files.
- Record required capabilities and a support status for each entry.
- Verify first-run preparation, existing-bundle reuse, changed content, missing
  dependencies and relocation. Publish a verified bundle atomically.
- Keep downloads out of imports, backend construction and the normal test suite.

Proposed storage layout:

```text
<asset-store>/superdex/<source-revision>/
  inventory.json
  assets/
    bots/
    prefabs/
    benchmarks/
    ...
```

Point `SUPERDEX_ASSETS_PATH` at the bundle's `assets` directory. Record local
derivatives with their own provenance instead of overwriting the source bundle.

**Done when:** the bundle can be relocated and verified, and every entry has
provenance, dependency completeness and required capabilities recorded.

### 2. Prove one unchanged robot: FR3 v2

Use `bots/arms/fr3_v2/fr3_v2.superdex_bot` through the existing native loader and
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

1. **Asset acquisition and inventory:** source pinning, dependency records,
   verification and first-run/reuse/relocation checks.
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
| Provenance and dependencies | Exact source revision/hashes; all required references resolve after relocation. |
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
Record the asset revision, UniSim revision, SDK version, platform, precision,
timestep, control profile and numerical tolerances with every result. Separate
checks performed through UniSim from any checks requiring a direct SDK fixture.

Run native viewer checks in serial mode with one environment; validate headless
batch execution separately. Keep missing-runtime diagnostics and unsupported
feature rejection covered by tests.

For implementation changes, run focused tests and `make check`. Run `make package`
when packaging changes and `uv lock --check` when dependency metadata changes.
External-asset/runtime tests may be optional in the normal suite, but an explicit
qualification run must fail if its required assets or runtime are missing.

Keep NumPy as the sole base dependency and preserve lazy SDK imports. Inspect
package contents when packaging changes to ensure large assets, caches and
native SDKs stay outside `unisim-core`.

## Deliverable for each completed stage

Leave a working example, repeatable tests and a compatibility report listing
asset paths, tested profiles, separate verification results and remaining
blockers. Update the [Superdex guide](superdex.md) alongside implementation.
Public contract or factory changes also require focused documentation and a
`CHANGELOG.md` entry; keep both READMEs synchronized if they change.
