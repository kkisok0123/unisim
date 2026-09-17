# Stage 7 native scene qualification

Stage 7 is complete for the unchanged Cart Pole and Half Cheetah scenes in the
local SuperDex bundle. These results qualify scene loading and physics/control
behavior through UniSim; they do not implement benchmark rewards or training.

| Scene | Articulation state | Effort inputs | Serial | Batch |
| --- | --- | --- | --- | --- |
| Cart Pole | 2 qpos / 2 qvel | Cart force, limit 3 | Passed | Passed |
| Half Cheetah | 9 qpos / 9 qvel | Six leg torques, limits 120/90/60/120/60/30 | Passed | Passed |

Both modes use two environments, 1,000 steps and `sim_dt=0.002`. Independently
loaded SDK scenes receive the same clipped physical efforts. Maximum measured
qpos, qvel and body-position error is **0.0** in all four runs (comparison
atol/rtol `3e-5`). The SDK default gravity is `[0, -9.8, 0]`; both files omit
gravity and solver settings. No ground is inserted. Half Cheetah falls under
gravity when no ground is supplied; this is the authored scene's behavior, and
its six authored rest springs remain active.

Checks cover complete actor inventories, effective gravity/solver settings,
initial state, independent action-routing probes, effort clipping, trajectories,
body positions, moved-state round trips, whole-scene and selective reset,
environment isolation, spring targets/gains, repeated close and recreation.
Synthetic pytest fixtures additionally cover custom gravity/solver settings,
initial velocities, nested rigid objects, explicit extra fragments, contact and
contact-filter behavior, invalid inputs, conflicts, and partial-load cleanup.

- [Numerical report](report.json): both execution modes, settings, actor inventories,
  controller gains/targets, SDK/precision, scene hashes and adapter source hashes.
- [Viewer report](viewer.json): 240 frames per scene, initial pose, controlled
  movement and whole-scene reset, with zero reset error and clean teardown.
- Manual visual inspection is outstanding. Renderer smoke is recorded separately;
  Stage 5's manual inspection also remains outstanding.

## Reproduce

Use CPython 3.12 or 3.13 with the optional SuperDex runtime and the local asset
bundle; no SDK source checkout is needed. From the repository root:

```bash
export SUPERDEX_ASSETS_PATH="$PWD/assets/superdex"
uv run --no-sync scripts/superdex_scene_qualify.py
uv run --no-sync pytest -q tests/test_superdex_scenes.py tests/test_superdex_assets.py tests/test_superdex_prefabs.py tests/test_superdex_component_qualification.py
uv run --no-sync scripts/superdex_scene_qualify.py --viewer --frames 240
```

For interactive inspection, omit `--frames` and select `--scenes cart_pole` or
`--scenes half_cheetah`. Close after the reset phase. Synthetic tests run without
asset downloads; unchanged-scene pytest checks opt in with `SUPERDEX_ASSETS_PATH`.

Repository validation: `UV_NO_SYNC=1 make check` passes Ruff and 380 tests, with
21 skips for optional runtimes/local fixtures. `make package` builds the source
distribution and wheel. The no-sync setting preserves the installed native SDK.

## Exact supported boundary

The adapter accepts one root-file fixed/revolute/prismatic articulation and
standalone rigid actors, nested rigid prefabs and extra rigid scene fragments.
It requires an explicit ordered joint selection and finite positive effort
limits. Authored gravity, solver settings, layer-contact filters and native
joint-tracking controllers are retained. Absent settings retain SDK defaults;
`sim_dt` is caller-owned. Unknown fields and conflicting nested settings fail.

Multiple articulations, FREE/spherical scene joints, advanced mechanisms,
soft bodies, scene cameras/plugins, arbitrary constraints and other controller
profiles remain unsupported. The existing native-bot FREE-root and built-in
camera/controller support is unchanged. Reset restores initial controller
targets without removing snapshot-owned entities. State round trips restore
kinematics, not hidden solver history.

The [inventory](../superdex-assets-inventory.md) now lists these scenes as
structural profile candidates without the obsolete blanket scene-loading
blocker. This report provides their runtime qualification; candidates elsewhere
are not automatically promoted. Historical stage reports have not been rewritten.
