# Stage 5: rigid objects and prefab assembly

Stage 5's numerical acceptance checks pass for FR3 with a sphere and a nine-hole
peg board. The implementation extends the existing scene/state interfaces;
there is no new cross-backend API or dependency. Work is based on the current
`superdex-assets-import` branch and remains uncommitted. No GitHub issue or PR
was created by this task.

## Qualified fixture

| Asset under `assets/superdex/` | Role | Result |
| --- | --- | --- |
| `bots/arms/fr3_v2/fr3_v2.superdex_bot` | Existing qualified robot; seven effort inputs | Direct-SDK parity retained |
| `prefabs/sphere/sphere.mochi_prefab` | Independent dynamic object above robot base | Robot–object contact observed |
| `prefabs/nine_hole_peg_test/nine_hole_peg_test.mochi_prefab` | Static board and nine dynamic pegs | Complete inventory/state/reset qualified |

The runner builds temporary two-level prefab wrappers. The outer assembly is
translated by 0.01 m in z and rotated 0.2 rad about z. Inside it, the sphere is
translated by (-0.09, 0, 0.23) m and the board by (0.55, 0, 0) m. Original asset
bytes are preserved. Native nesting prefixes all object names with `assembly/`.
There are 22 native actors: the articulation and its ten links, plus eleven
standalone rigid actors. The canonical inventory has world, ten robot bodies,
and eleven rigid bodies. Ten objects are dynamic, giving `nq=77`, `nv=67` and
seven action entries. Each rigid actor retains its authored collision shape;
nested geometry and transforms are delegated to the SDK without conversion.

## Evidence

[`report.json`](report.json) records source commit plus hashes of the uncommitted
implementation, asset manifest digest/provenance, SDK, precision, platform,
controls and per-mode results. Asset verification runs before construction and
fails on missing files, changed hashes or unresolved dependencies. The verified
local copy supplies all dependencies; the source checkout is not used.

Both serial and batch modes ran 1,000 steps at 0.002 s with two environments.
The robot uses PD effort control (`kp=80`, `kd=8`) around its initial pose and
87/87/87/87/12/12/12 N·m limits. An independently constructed SDK scene receives
the same control law. The measured robot qpos/qvel and every object's world
pose/angular velocity match exactly (maximum deviation 0.0; tolerance 3e-5).
The sphere contacts a robot link on two steps, with the same filtered actor-pair
contact observed through the adapter scene. This proves a contact interaction,
not sustained grasping or a manipulation task; the sphere subsequently falls
away because the fixture has no ground plane.

After motion, full/selective resets restore initial coordinates and authored
velocities. State round trips exercise rotated objects, translated positions and
nonzero linear/angular velocities. Resetting or setting environment zero leaves
environment one bitwise unchanged. A direct single-scene step separately checks
scene ownership isolation (the public `step` advances every environment).
Repeated close and recreation pass. Synthetic tests additionally cover a
floating robot beside objects, nonzero COM offsets, initial object velocity,
force application, invalid quaternions, missing dependencies, recursive cycles,
and rejection of unsupported prefab contents.

[`viewer.json`](viewer.json) records the separate 420-frame native renderer smoke
run: initial geometry, contact simulation, whole-scene reset, and successful
viewer/backend cleanup. The restored qpos error is 0.0. This is automated rendering
evidence, **not a claimed human visual inspection** of geometry, scale or alignment.
No screenshots were saved. The interactive command remains available for that
inspection.

## Reproduce

```bash
SUPERDEX_ASSETS_PATH="$PWD/assets/superdex" uv run scripts/superdex_prefab_qualify.py
SUPERDEX_ASSETS_PATH="$PWD/assets/superdex" uv run scripts/superdex_prefab_qualify.py --viewer --frames 420
SUPERDEX_ASSETS_PATH="$PWD/assets/superdex" uv run pytest -q tests/test_superdex_prefabs.py
make check
make package
```

The ordinary suite's synthetic tests do not require ignored assets; runtime tests
skip when the optional SDK is absent. Explicit qualification never substitutes a
skip for a missing runtime or bundle. No asset download is added to CI.

## Remaining limits

Other local prefabs have not been promoted by this report. Standalone prefab-only
models, full scene files, multiple articulations, soft bodies, constraints,
controllers, sensor/actuator components, authored scene settings and contact
filter overrides remain outside the supported stage-5 profile. Static fixtures
have fixed authored transforms. State round trips restore kinematics, not hidden
solver history. Viewer imports remain optional, and original assets remain
ignored and excluded from the distribution.

## Local validation result

On 2026-09-16, `make check` passed Ruff and **329 tests**, with 19 optional tests
skipped. The explicit asset-enabled prefab test file passed **15 tests**,
including cleanup after an injected partial-instantiation failure. `make package`
built the sdist and wheel; archive inspection confirmed the prefab implementation
is included and the ignored asset collection is excluded. Both recorded
qualification modes and the 420-frame renderer smoke passed. Changes are left
unstaged and uncommitted for review.
