# SuperDex shared-interface qualification

This report records the 2026-09-18 shared-interface change independently of
historical Stages 1–8. Asset payloads and historical reports remain unchanged.
The shared contract additions do not change other backends' physics behavior.

## What was verified

- Every public SuperDex method/property is declared in `SimBackend`; importing
  the shared target/metadata types does not import engine runtimes.
- Unsupported optional operations fail explicitly on a base-only backend;
  default close releases materialization artifacts idempotently.
- Model metadata is detached and exposes coordinate units, groups and ownership.
  Scalar efforts expand after coordinate selection, including spherical groups
  and zero-action profiles. Invalid scalar/vector limits are rejected.
- Gravity overrides affect every environment in both existing serial and batch
  profiles, persist across selective/full reset, and agree with measured motion.
- Joint, Cartesian and articulation-pose targets convert inside SuperDex. Complete
  batches are validated before stepping. Shared and legacy targets agree; legacy
  objects emit deprecation warnings. Floating OSC remains explicitly SDK-blocked.
- Normal bot/rigid viewers have no SDK imports or private backend/model accesses.
  The synchronized adapter viewer uses shared targets and `run_playback()`.

## Root-frame correction

The expanded public metadata enabled an additional check of native bot root state
against world body poses. It exposed a pre-existing omission of bot-level
`worldFromRoot` placement in floating-bot state conversion (the four Oculus
hands). Materialization now composes that placement with authored joint/link
frames. Synthetic translated/rotated fixtures verify pose, velocity, state
round-trip and reset in both existing serial and batch execution modes. Native
physics remains the reference; historical reports are not rewritten.

## Numerical evidence

The [rigid summary](rigid/summary.json) and [fixture manifest](rigid/fixture-manifest.json)
record the full Stage 8 matrix with fresh source and installed binary hashes.
Qualification uses 1,000 steps per model profile and an independent SDK reference.
The original authored FR3 controller file is exercised separately through the
shared-target component runner; see [its report](controller/fr3_v2.json).

The matrix passed 65 profiles: 54 native assets and 11 synthetic fixtures. Nine
custom-component/deformable profiles remain deferred. The separate authored
controller run covers the remaining controller file in the 55-file native target
set and passed 1,040 physics steps. Implementation hashes identify the tested
working tree. A subsequent CLI-only correction to `superdex_rigid_viewer.py`
rejects standalone `.mochi.h5` shapes with an explanatory error and passes
controlled-joint selection only to scenes/prefabs; the recorded viewer hash
predates that correction. Adapter implementation hashes remain unchanged.

Across the matrix, maximum native-state, body-pose and sampled contact-force
deviations from the SDK were zero. Maximum state round-trip error was
`3.9935112e-6`; maximum reset error was `3.9907084e-8`, within qualification
tolerances. Every passing profile also passed cleanup/recreation and control
routing checks.

The numerical runners retain explicit white-box native audits for compiled actor
inventories, contact samples and native-state fidelity. Those are qualification
checks, not application APIs. Normal application code uses public methods.

## Test and renderer results

- Focused shared-interface suite: 19 passed.
- Full asset-enabled SuperDex suite: 216 passed.
- `make check`: Ruff passed; 418 tests passed and 21 skipped.
- Both test runs emitted 26 expected deprecation warnings from legacy-target
  compatibility coverage.
- `make package`: source distribution and wheel built; both exclude asset
  payloads, and the wheel contains the new shared types and metadata module.
- Refreshed inventory: 888 files, 260,255,576 bytes, no dependency errors;
  tree digest `3f896fd4a0c30516f4538dfcafdb1bcdd4a35eea18e7ddd9ab9504bfe619b6cf`.

Offscreen native renderer smoke passed for the
[2F85 rigid viewer with gravity disabled](viewer-rigid.json) (120 frames),
[FR3 bot viewer](viewer-fr3/report.json) (570 frames), and
[floating OpenArm gripper viewer](viewer-floating/report.json) (570 frames).
The latter two exercise public metadata, controls, state, reset and `CameraCfg`
playback. All three reported zero immediate reset error. These are automated
renderer runs, not completed human visual inspections.

## Reproduce

```bash
uv run --no-sync pytest -q tests/test_superdex_shared_api.py tests/test_superdex_rigid.py tests/test_superdex_component_qualification.py
uv run --no-sync scripts/superdex_rigid_qualify.py --out docs/superdex-interface-qualification/rigid
SUPERDEX_ASSETS_PATH="$PWD/assets/superdex" uv run --no-sync scripts/superdex_component_qualify.py \
  --bots bots/arms/fr3_v2/fr3_v2.superdex_bot --effort-limit 1 \
  --controller-config docs/superdex-rigid-qualification/configs/fr3-authored-pose.json \
  --out docs/superdex-interface-qualification/controller
SUPERDEX_ASSETS_PATH="$PWD/assets/superdex" uv run --no-sync pytest -q tests/test_superdex*.py
make check
make package
```

No new batch profiles, native image capture, deformable/custom-component support,
URDF fidelity or solver changes are claimed. Native renderer smoke is separate
from manual inspection; the outstanding Stage 5/7/8 inspections remain outstanding.
