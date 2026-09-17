# SuperDex assets

Stages 1–6 are implemented for the recorded profiles in this local bundle.
**10 floating models and 11 fixed-base models qualify**, and FR3 with a sphere
and nine-hole peg board passes rigid-prefab qualification. Stage 6 adds
built-in camera metadata/poses, explicit controllers and universal
qualification against the direct SDK. Its bundle audit records **21 passed,
14 blocked and no failed checks**; it does not qualify every component or asset.
Stage 5's native viewer smoke passed; manual visual inspection remains
outstanding. Complete `.mochi_scene` dispatch is planned for Stage 7.

This directory is local and ignored by Git. The tracked qualification code
and reports live under `scripts/`, `tests/`, and `docs/` in the repository.
Run the commands below from the repository root.

UniSim does not require SuperDex assets to live in this directory. Set
`SUPERDEX_ASSETS_PATH` to the root of your own SuperDex asset bundle. The root
must contain the `bots`, `prefabs`, or `test` directories and their
`.superdex_root` markers. For the FR3 qualification test, this file must exist:

```text
$SUPERDEX_ASSETS_PATH/bots/arms/fr3_v2/fr3_v2.superdex_bot
```

Use an absolute path so the setting works from any working directory:

```sh
export SUPERDEX_ASSETS_PATH="/absolute/path/to/project_superdex/assets"
```

The export applies to the current shell. To keep it for future shells, add the
same line to `~/.bashrc` or `~/.zshrc`, then open a new shell or source that
file. If the assets are stored in this UniSim checkout, run the following from
the repository root instead:

```sh
export SUPERDEX_ASSETS_PATH="$PWD/assets/superdex"
```

## Run the FR3 qualification tests

The SuperDex runtime supports CPython 3.12 and 3.13. Install the optional
dependencies once, then run the opt-in test suite:

```sh
uv sync --python 3.12 --extra superdex --extra mujoco
uv run --no-sync pytest -q -rs tests/test_superdex_fr3_qualification.py
```

The suite contains eight regression tests. It loads the unchanged FR3 through
the UniSim SuperDex adapter and compares it with a direct SuperDex SDK scene
driven with the same inputs. It checks:

- robot structure, joint order, masses, transforms, bounds, and default pose;
- control ordering and effort-limit clipping;
- batch and serial trajectories over more than 1,000 physics steps;
- contact recovery from an out-of-range pose;
- reset and state round trips;
- isolation between environments;
- repeated create, step, reset, and close cycles; and
- the recorded qualification report schema and required results.

A passing run means the adapter still matches the direct SDK within the test
tolerances for this FR3 profile. The tests skip when the optional SuperDex
runtime, a supported Python version, or the configured asset bundle is missing.
The `-rs` option prints the reason for every skip.

## Run the stage-3 bot qualification tests

Stage 3 extends the same checks to other fixed-base bots and to recipe
compositions (an arm with an attached hand). The candidate list and the
per-bot control settings live in `scripts/superdex_bot_profiles.py` at the
repository root. Run:

```sh
uv run --no-sync pytest -q -rs tests/test_superdex_bot_qualification.py
```

Eleven bots are qualified this way, from the one-joint `googly_eyes` to the
54-joint `openarm_v20_wuji` recipe. The recorded results and the
compatibility table are in `docs/superdex-bots-qualification/` at the
repository root. One guard test compares the candidate lists against the
inventory, so a bot can never silently disappear from testing: it must be
either qualified or listed with a precise blocker.

## Run the Stage 4 floating-model checks

The ten qualified floating models are:

| Family | Qualified variants |
| --- | --- |
| Allegro V5 | Left and right |
| DG5F Short | Left and right |
| DG5F Long | Left and right |
| Wuji Hand2 Beta1 | Unactuated left and right |
| OpenArm V20 grippers | Left and right |

Run the focused tests, audit all 22 floating models, and repeat the eleven
fixed-base regression profiles:

```sh
uv run --no-sync pytest -q -rs tests/test_superdex_native_floating.py
uv run --no-sync scripts/superdex_floating_qualify.py --all-floating --out docs/superdex-floating-qualification/all-models
uv run --no-sync scripts/superdex_bot_qualify.py --out docs/superdex-floating-qualification/fixed-base-regression
```

The check code is in `scripts/superdex_floating_qualify.py` and
`tests/test_superdex_native_floating.py`. It checks each model against a direct
SDK scene in serial and batch modes: default pose, geometry, joint ordering,
root-state conversion, 1,040-step trajectories, state round trips, effort
clipping, full/selective reset, environment isolation, and cleanup. Separate
tests cover translated/rotated root frames and viewer exit/error paths.

Canonical root state uses world xyz and a wxyz quaternion. Linear velocity is
measured at the root-link origin in world axes; angular velocity uses root-link
axes. Scalar joint positions follow the seven root position entries, and
scalar joint velocities follow the six root velocity entries.

The all-floating audit records **10 passes and 12 explicit blockers**: seven
models have unsupported sensor/actuator components, four Oculus XR models have
spherical child joints, and the 2f_85 gripper has mechanical cycles. Stage 6's
built-in support does not remove these blockers; they require further component
or Stage 8 capability work. Expected blockers do not fail the Stage 4 audit;
unexpected or numerical failures do. To require a particular model to pass,
select it explicitly:

```sh
uv run --no-sync scripts/superdex_floating_qualify.py --bots wuji_hand2_beta1_left --out /tmp/superdex-wuji-qualification
```

All 35 bot models are accounted for: 21 qualify and 14 have recorded blockers,
including four fixed-base models. Stage 5 adds independent objects and object
contact alongside a qualified robot; it does not promote additional bot models.

## Run floating-model viewer smoke checks

Open one interactive native viewer, or run movement and reset through the
native viewer for all ten qualified floating models:

```sh
uv run --no-sync scripts/superdex_bot_viewer.py --bot wuji_hand2_beta1_left
uv run --no-sync scripts/superdex_floating_viewers.py
```

Viewer profiles use gravity compensation, bounded joint motion, and slow root
motion to keep each model visible. Numerical qualification uses ordinary
gravity. The commands record numerical reports and logs without saving images.
A real interactive window-close check passed.

See the [Stage 4 completion report](../../docs/superdex-floating-qualification/README.md),
[floating compatibility table](../../docs/superdex-floating-qualification/all-models/compatibility-table.md),
and [viewer measurements](../../docs/superdex-floating-qualification/viewers/summary.json).
The recorded completion run had **23 focused tests passed**, **315 passed and
18 skipped** in the full suite, and a successful package build.

## Run the Stage 5 rigid-prefab checks

With `SUPERDEX_ASSETS_PATH` configured as above, run:

```sh
uv run --no-sync scripts/superdex_prefab_qualify.py
uv run --no-sync pytest -q -rs tests/test_superdex_prefabs.py
uv run --no-sync scripts/superdex_prefab_qualify.py --viewer
```

The fixture combines unchanged FR3, sphere and nine-hole peg-board assets
through temporary, translated and rotated nested prefab wrappers. It contains
11 independent rigid actors: one sphere, one static board and nine dynamic
pegs. Both serial and batch modes passed 1,000 steps at 0.002 s with two
environments and zero measured deviation from independently constructed SDK
scenes. Robot–sphere contact, moved-object state round trips, whole-scene and
selective reset, environment isolation, cleanup and recreation passed. The
sphere later falls away because this fixture has no ground plane; this is a
contact check, not a sustained grasp or manipulation task.

Use `SceneCfg.fragment_files` to add `.mochi_prefab` files to a native bot.
Dynamic objects append world xyz/wxyz pose and world origin/body angular
velocity coordinates to the existing robot `qpos`/`qvel` arrays. Robot action
indices stay unchanged. `get_root_state_layout(body_name)` exposes object
indices; static fixtures keep their authored transforms and have no state
coordinates. State round trips restore pose and velocity, not solver history.

The viewer holds the initial pose, simulates contact, and resets the complete
scene for inspection. For the repeatable offscreen smoke check, run:

```sh
uv run --no-sync scripts/superdex_prefab_qualify.py --viewer --frames 420
```

The 420-frame smoke completed all phases with zero reset error and successful
cleanup; no images were saved. It does not replace manual visual inspection of
geometry, scale and alignment. The recorded validation passed **15 focused
tests**, **329 passed and 19 skipped** in the full suite, and the package build.

See the [Stage 5 report](../../docs/superdex-prefab-qualification/README.md),
[numerical results](../../docs/superdex-prefab-qualification/report.json),
[viewer smoke results](../../docs/superdex-prefab-qualification/viewer.json),
and [adapter guide](../../docs/superdex.md#rigid-objects-and-nested-prefabs-stage-5).
Other prefabs are not automatically qualified. Scene settings, contact-filter
overrides, constraints, controllers, sensor/actuator components, extra
articulations and soft bodies remain outside this rigid-prefab profile.
Stage 6 adds built-in cameras/controllers below; full `.mochi_scene` dispatch
remains stage 7.

## Run the Stage 6 universal qualification checks

Stage 6 is complete for the supported profiles. The qualification tool works
with fixed/floating bots and optional rigid prefab fragments, comparing UniSim
with direct SDK execution in serial and batch modes.

| Capability | Verified coverage and limits |
| --- | --- |
| `SENSOR_CAMERA` | Link-attached settings and world poses, including translated/rotated mounts; no image rendering |
| `BASIC_JSC_PD` | Explicit controller execution on fixed and floating roots |
| `BASIC_OSC_PD` | Fixed-base execution; the current SDK rejects initialization on the tested floating-base bot because of effort-limit indexing |
| `MOCHI_ARTICULATED_POSE` | Fixed/floating execution and both scalar-joint and per-link-transform target forms |

Controllers have independent instances per environment, selective reset and
cleanup. Automatic controller selection, arbitrary custom components and full
`.mochi_scene` dispatch are outside this stage.

Models including `wuji_hand2_beta1` support ordinary joint-torque control.
Custom actuator/sensor components are reported as unsupported.

### Compare the Stage 6 adapter and SDK viewers

Use `--absolute-target` with its matching controller configuration to hold an
explicit command. Add `--visualize` to open synchronized `UniSim adapter` and
`Native SuperDex SDK` windows for side-by-side inspection:

```sh
uv run --no-sync scripts/superdex_component_qualify.py \
  --bots bots/arms/fr3_v2/fr3_v2.superdex_bot \
  --effort-limit 87,87,87,87,12,12,12 \
  --controller-config docs/superdex-component-qualification/configs/fr3-jsc.json \
  --absolute-target docs/superdex-component-qualification/targets/fr3-jsc-absolute.json \
  --visualize
```

This requires the optional SuperDex viewer with Polyscope >= 2.5.0 and exactly
one bot. Close either window or press Ctrl+C to stop both. For OSC or
articulated-pose control, replace `fr3-jsc.json` and `fr3-jsc-absolute.json` with
the matching `fr3-osc` or `fr3-pose` files in the same directories.

Visual mode writes neither a qualification report nor a GIF. Remove
`--visualize` and supply a separate `--out` directory to record numerical
qualification with the same absolute target. Visual inspection and numerical
qualification remain separate evidence.

### Run numerical qualification and regression tests

```sh
uv run --no-sync scripts/superdex_component_qualify.py \
    --bots bots/arms/fr3_v2/fr3_v2.superdex_bot --effort-limit 87,87,87,87,12,12,12
uv run --no-sync scripts/superdex_component_qualify.py \
    --bots bots/hands/wuji_hand2_beta1/left/wuji_hand2_beta1_left.superdex_bot \
    --effort-limit 1 --scene prefabs/sphere/sphere.mochi_prefab
uv run --no-sync scripts/superdex_component_qualify.py --all --effort-limit 1
uv run --no-sync scripts/superdex_component_qualify.py \
    --bots bots/arms/fr3_v2/fr3_v2.superdex_bot --effort-limit 87,87,87,87,12,12,12 \
    --controller-config docs/superdex-component-qualification/configs/fr3-jsc.json \
    --absolute-target docs/superdex-component-qualification/targets/fr3-jsc-absolute.json \
    --out /tmp/fr3-jsc-qualification
uv run --no-sync pytest -q tests/test_superdex_component_qualification.py
```

Supply `--bots` or `--all`; models with missing authored effort limits require
`--effort-limit`. This fallback is a qualification input, not hardware calibration.
Without `--controller-config`, the runner uses bounded PD-generated torque
commands. The tool checks direct-SDK equivalence in both execution
modes, cameras, scene fragments, clipping, reset, isolation and lifecycle.
Unsupported models are recorded as blocked and absent cameras as skipped;
`--all` continues through the bundle and returns nonzero for failures/blockers.
The recorded bundle audit used an effort-limit fallback of 1 and reports
**21 passed, 14 blocked and no failed checks** across all 35 bots. Separate
reports cover fixed/floating torque control, both non-actuated Wuji hands,
each built-in controller on FR3, and a floating Wuji hand with a sphere.
Camera mount and fixed/floating controller coverage also use synthetic test
fixtures; a bot report with no cameras does not establish camera support.

Reports and their exact coverage are listed in the
[Stage 6 report](../../docs/superdex-component-qualification/README.md).
Use separate `--out` directories for different controller configurations of the
same bot, because report filenames are based on the bot filename. See the
[adapter guide](../../docs/superdex.md#built-in-cameras-controllers-and-universal-qualification-stage-6)
for the API, target conventions and all three controller configurations.

## Bundle verification

Qualification runners verify the local tree against
`docs/superdex-assets-inventory.json` before loading models. That checksum
includes local documentation such as this README. Updating documentation
therefore requires refreshing the recorded inventory, even when model files
are unchanged. Qualification reports retain the bundle digest from their
original run; the inventory describes the current local bundle.

Refresh the local inventory without consulting the source checkout:

```sh
uv run --no-sync scripts/copy_superdex_assets.py --inventory-only
```

## Known SuperDex SDK phenomena observed during qualification

These are behaviors of the SDK or the authored assets, not UniSim adapter
bugs. Each was isolated against a direct SDK scene driven with inputs
identical to the adapter's, so the adapter was ruled out as the cause.

### Single-joint dominance does not generalize

The FR3 check "torque one joint; that joint reacts the most" relies on the
FR3's inertia distribution, not on a general law of the adapter. On the
openarm arms the distal links are so light that torquing a proximal joint
moves the wrist faster than torquing the wrist itself: after five 2 ms
substeps with 3.5 N·m on openarm left joint 5, the wrist velocity response is
larger than the response to torquing the wrist directly. The stage-3
replacement check compares the adapter's full per-joint velocity response
matrix (torque each joint, record every joint's response, subtract a
zero-torque baseline) against the direct SDK matrix. Exact equality proves
the control wiring; a nonzero diagonal proves each control column reaches its
own joint.

### Asking for contact points on a scene with a rough history halts the SDK

The contact check asks the SDK "where does the robot touch itself?". To get
an answer, that question must first be registered on the robot's links
(`register_query(CONTACT_POINTS)`). The SDK also keeps an internal account
of contact forces for every pair of touching links. When the question is
registered on a scene that has already simulated the robot jamming its own
links deeply into each other — which the movement stress-test does — that
account no longer matches what the SDK finds on later steps, and the debug
build deliberately stops the whole program instead of continuing with
possibly wrong numbers:

```text
MOCHI ASSERTION FAILURE:
    Expected a force vector for every contact between this pair of entities
```

This is SDK strictness, not corrupted UniSim state: the same
register-after-movement sequence halts a plain SDK scene with no adapter
involved, and the adapter's own scenes replay the identical trajectory
without halting. The fix is procedural — every contact check in the stage-3
runner builds a fresh reference scene whose only history is the check
itself, which is stable.

### Robots that declare zero joint inertia need a minimum damping setting

The openarm asset files declare each joint's internal inertia — how much
the joint resists being sped up — as 0. The viewer demo drives joints with
a PD controller: think of a spring (`kp`) pulling each joint toward its
target and a shock absorber (`kd`) keeping it from oscillating. The shock
absorber is sized from the declared inertia; with 0 declared, it falls back
to a small minimum. That minimum (0.001, giving kd 0.49 at kp 60) was too
weak: the arm oscillated harder and harder until the physics solver gave up
and crashed — always around demo frame 141, on every replay of the same
command sequence. Raising the per-robot minimum to 0.05 (kd 3.46) removed
the crash with margin; 0.01 also passed. The setting is `armature_floor`,
recorded per bot in `scripts/superdex_bot_profiles.py`. If another
zero-inertia robot crashes the same way, raise its floor first and compare
against a direct SDK scene before suspecting the adapter.
