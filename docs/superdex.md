# SuperDex CPU development profile

The `superdex` adapter runs SuperDex Physics/Robotics 1.0.0 directly behind
`SimBackend`. UniLab roadmap [#1533](https://github.com/Motphys/UniLab/issues/1533)
tracks this development profile. Changes remain on roadmap branches; the
version is unchanged and no PyPI release is required for local integration.

## Installation and ownership

Use CPython 3.12 or 3.13, as covered by the superdex-uni wheels. From the UniSim
checkout:

```sh
uv sync --python 3.12 --extra superdex --extra mujoco
```

`superdex-physics-uni==1.0.0` and `superdex-robotics-uni==1.0.0` are optional.
They are a temporary unilabsim build of the upstream SuperDex 1.0.0 facades
carrying the native batch executor, published from
[unilabsim/superdex-uni](https://github.com/unilabsim/superdex-uni) until the
upstream project_superdex PR merges; they install into the same `superdex/`
namespace as the upstream packages and must not be co-installed with them. The
extra also supplies MuJoCo 3.11 as a **cold MJCF parser**; SuperDex executes every
physics step. Native `.superdex_bot` and `.mochi_scene` loading do not use that parser.
Importing `unisim` or its `SuperDexBackend` class does not load either engine.
SuperDex Lab, Gymnasium and a learner are not adapter dependencies.

For a sibling UniLab checkout, keep both versions unchanged and install local
editable projects together, for example `uv pip install -e './[superdex,mujoco]'
-e ../UniLab`. Use `uv run --no-sync` (or `UV_NO_SYNC=1 make check`) while
testing editable overrides so normal project synchronization does not replace
them with index distributions. UniLab's local provenance test profile uses
`UNILAB_LOCAL_UNISIM` pointing at the exact UniSim checkout. The UniLab backend
guide describes its task and registered asset setup.

The verified platform is Linux x86_64, CPU FP32. Upstream also provides
Windows x86_64 and macOS ARM wheels, but this integration has not established
those platforms. The default x86 build requires AVX2 and related instructions.
The upstream source exposes optional CUDA linear solvers, but the tested wheel
rejects them as not built with CUDA. This adapter does not enable GPU solvers.
FP64 upstream packages require a process-wide precision choice before import;
the integration's numerical validation currently targets FP32.

Each environment owns an independent native scene. The adapter reference
counts the process-global engine: closing one instance leaves other instances
alive. The source-built SuperDex `SceneBatchExecutor` batches force writes,
stepping, articulated state, link state, contact sensors and solver status in
persistent C++ workers. `superdex_num_workers=0` uses the physical cores visible
to the process (Linux topology or macOS `sysctl`);
SDK-internal workers are disabled. Runtime initialization must belong to UniSim,
and live backends cannot be transferred between processes. Call the public
`cleanup_scene_assets()` hook or `close()` before interpreter shutdown. UniLab's
`env.close()` calls that public hook.

## Native debugger and serial execution

A SuperDex scene's `DebugDraw` object is thread-affine. When the native SuperDex
debugger is connected, its sync callbacks gather debug-draw data from the
scene's step thread, so stepping scenes on `SceneBatchExecutor` workers with an
attached debugger violates that affinity and traps natively. The default
`batch` execution mode therefore fails closed: constructing or stepping the
backend while a debugger client is connected raises an actionable `RuntimeError`.

Attach the debugger only with the serial execution mode, which never constructs
the executor and steps every scene on the environment thread:

```sh
create_backend("superdex", scene, num_envs, sim_dt, superdex_execution_mode="serial")
```

In UniLab pass `env.superdex_execution_mode=serial` on the Hydra command line.
`superdex_num_workers` has no effect in serial mode. The mode is a debugging
profile, not a performance configuration: prefer `batch` for training.

Serial mode also unlocks the native Polyscope viewer
(`superdex.physics.viewer`) for `run_playback` in `interactive` render mode:
the viewer shares the scene's thread with stepping, so interactive playback
requires serial mode and exactly one environment, and fails closed with an
actionable error otherwise. `record`/`auto` playback still uses the shared
MuJoCo offline renderer and works in both modes. UniLab's interactive superdex
eval injects both settings (`serial` + `training.play_env_num=1`).

## Native robots

Preprocessed SuperDex assets stay outside the code repositories. The FR3 example
uses the upstream `assets/bots/arms/fr3_v2` directory including its HDF5 collision
and GLB render files. Preserve its LICENSE/NOTICE. The native bot must have a
HARD or FREE root. Stage 8 also supports spherical child joints, closed loops,
linear transmissions and spatial tendons in one serial environment. Custom
components remain deferred; see the complete native rigid profile below.

The local asset copy lives at `assets/superdex` (ignored by git; cloning UniSim
does not provide it). Before an SDK load, `verify_asset_bundle()` from
`unisim.backend.superdex.assets` re-hashes the tree against the recorded
inventory (`docs/superdex-assets-inventory.json`) and re-resolves every
dependency edge, so a qualification run fails on missing or modified assets
without consulting the original checkout.

```python
import numpy as np
from unisim import create_backend
from unisim.scene import SceneCfg

backend = create_backend(
    "superdex",
    SceneCfg("/path/to/project_superdex/assets/bots/arms/fr3_v2/fr3_v2.superdex_bot"),
    num_envs=2,
    sim_dt=0.002,
    base_name="fr3_link0",
    superdex_num_workers=0,
    superdex_effort_limits=[20, 20, 20, 20, 5, 5, 5],
)
try:
    backend.step(np.zeros((2, 7)), nsteps=5)
    state = backend.get_state()
finally:
    backend.cleanup_scene_assets()
```

The native control vector names and ordering follow the single-DoF joint names.
Positive finite effort limits must be present in the asset or supplied
explicitly. The example values define a research control profile, not verified
FR3 hardware ratings. Fixed-base `get_state()` contains only joint coordinates;
requesting a floating-root layout for a fixed body is rejected. A named keyframe
must actually exist in the scene; the adapter does not invent `home` for bots.

## FR3 viewer demonstration (stage 2A)

One documented command visualizes the unchanged FR3 through the adapter's
native `run_playback` path against the local asset copy. It requires the
optional runtime (`uv sync --python 3.12 --extra superdex --extra mujoco`),
Polyscope >= 2.5.0 and a graphical session:

```sh
export SUPERDEX_ASSETS_PATH="$PWD/assets/superdex"
uv run scripts/superdex_bot_viewer.py
```

The script verifies the asset tree against the recorded inventory (use
`--skip-verification` to bypass), constructs the backend with
`superdex_execution_mode="serial"`, `num_envs=1` and no recording, and runs
three distinct, observable phases in the native Polyscope viewer:

1. **Initial pose** (2 s): PD position control holds the authored default pose
   so geometry, scale and link alignment can be inspected.
2. **Bounded movement** (6 s): a phase-shifted sine sweep per joint, bounded
   inside the authored joint ranges.
3. **Reset** (1 s settle): `backend.reset()` visibly restores the default
   pose; the restored joint state is checked numerically (exact by
   construction) before the settle phase.

Control profile (recorded in `docs/superdex-fr3-viewer/report.json`): commands
are joint-position targets in radians following the authored actuator order
`fr3_joint1..fr3_joint7`; a pre-step PD converter (`kp` 400/400/400/400/20/20/20,
critically damped `kd`) turns them into motor torques clipped to explicit
effort limits of 87/87/87/87/12/12/12 N·m (Franka's SRMS rating shape, a
demonstration profile, not verified hardware ratings); `sim_dt` 0.002 s with
8 substeps per 60 Hz render. Zero effort does **not** hold the arm: without
torque the FR3 collapses under gravity within half a second, and the earlier
`[20,20,20,20,5,5,5]` research profile visibly sags at the elbow.

Closing the window releases the viewer and backend (verified for window close,
Ctrl-C and error paths); the run report lands in
`docs/superdex-fr3-viewer/`. Use `--frames N` for an offscreen smoke run on a
headless host. The viewer does not save images. Physics and lifecycle
qualification of the unchanged FR3 follows below.

## FR3 adapter qualification (stage 2B)

One command runs the numerical and lifecycle qualification of the unchanged
FR3 against direct SDK execution; no viewer or graphical session is needed:

```sh
export SUPERDEX_ASSETS_PATH="$PWD/assets/superdex"
uv run scripts/superdex_fr3_qualify.py
```

The script verifies the asset tree against the recorded inventory, then loads
the FR3 through `create_backend("superdex", ...)` while a direct SuperDex SDK
scene in the same process is driven with identical inputs, separating adapter
translation from asset/SDK behavior. Seven checks run, each recorded with
evidence in `docs/superdex-fr3-qualification/report.json`:

1. **Structure** — body/joint inventories, actuator control order, default
   pose, dynamic-link masses, world link transforms, world AABBs and authored
   joint ranges against the direct SDK actor and authored prefab.
2. **Control** — per-joint ordering by rollout differencing (the torqued joint
   is the most affected DoF for every joint index) and effort-limit clipping
   (a saturated command reproduces per-joint capped dynamics exactly).
3. **Trajectory equivalence** — batch and serial execution modes each match
   the direct SDK rollout under the recorded PD sweep profile; this is also
   the 1,000+ step stability check.
4. **Contact recovery** — starting from a pose beyond joint 2's authored range
   (`-2.6` rad < `-1.784`), `set_state` stores the pose unclamped and the
   contact push-out matches the SDK rollout through 14 contact steps.
5. **Reset** — whole-scene and selective restore are exact, and
   `set_state`/`get_state` round-trip within float32 noise.
6. **Isolation** — each environment of a two-environment backend reproduces
   the matching single-environment backend exactly while the other runs
   different commands.
7. **Lifecycle** — repeated create/step/reset/close cycles.

On the qualification host (SuperDex 1.0.0, float32, `sim_dt` 0.002) every
adapter-vs-SDK deviation measured exactly 0.0 — batch, serial and direct SDK
integrate identically — and no adapter defect was demonstrated, so stage 2B
required no adapter fix. Known SDK behavior recorded alongside: authored joint
limits are soft (an all-limits pose overshoots ~0.11 rad under 87 N·m), and
`set_state` does not clamp to authored ranges by design. The report records
code commit, asset tree digest, SDK precision, timestep, control profile and
tolerances.

The regression subset is `tests/test_superdex_fr3_qualification.py`
(8 tests, opt-in via `SUPERDEX_ASSETS_PATH`; skips cleanly without the local
assets or runtime). The pre-existing mass-metadata regression in
`tests/test_superdex.py` stays a separate test.

## Compatible bots and recipe compositions (stage 3)

One parameterized runner applies the stage-2B check set to every registered
candidate — native fixed-base bots and Mod Bot recipe compositions — against
direct SDK execution; the stage-2A viewer demonstration gained `--bot <key>`
for the same assets:

```sh
export SUPERDEX_ASSETS_PATH="$PWD/assets/superdex"
uv run scripts/superdex_bot_qualify.py                  # all candidates + blocked probes
uv run scripts/superdex_bot_qualify.py --bots openarm_v20_wuji,googly_eyes
uv run scripts/superdex_bot_viewer.py --bot openarm_v20 # visual check, serial + 1 env
```

Candidate selection and per-asset control profiles live in
`scripts/superdex_bot_profiles.py` (explicit effort limits where authored
limits are unlimited, `kp` with critically damped `kd` from authored armature,
per-joint sweep amplitudes bounded by the authored ranges). Eleven bots
qualified with every adapter-vs-SDK deviation exactly 0.0: the fr3 and fr3_v2
arms, both openarm_v20 arms, googly_eyes, fr3_v2_with_eyes, and the recipe
compositions `openarm_v20` (18 joints), `openarm_v20_wuji` (54),
`fr3_dg5f_short` left/right (27) and `fr3_v2_allegro_v5_right` (23). Reports
and the compatibility table land in `docs/superdex-bots-qualification/`.

Recipe compositions need no adapter extension for this set: the SDK resolves
`base` plus `AttachBot`/`ReplaceLinkWithBot` references (including `//`-rooted
paths and prefixes) into one compiled prefab at load time, and the compiled
results stay inside the native loader's profile (HARD root, fixed/revolute
joints, no components). The runner adds a recipe-accounting check that every
base/attachment reference resolves inside the verified bundle, and structure
checks compare against the *compiled* SDK actor, not a sum of parts
(`ReplaceLinkWithBot` merges links: openarm_v20's components sum to 28 links,
the compiled robot has 26).

Two stage-2B check generalizations were required. The FR3 control-ordering
probe (torqued joint = most affected DoF) does not hold for light distal
links — openarm's wrist responds more to a proximal torque than to its own —
so control now verifies the adapter's per-joint velocity *response matrix*
matches the direct SDK matrix exactly plus a nonzero diagonal (each control
column reaches its own joint). The FR3 10 rad/s sweep-velocity guard is kept
only on the pure fr3 arms; near-massless distal joints (googly_eyes,
hand/finger combos) legitimately oscillate far faster while staying
adapter-vs-SDK exact. Recorded SDK behaviors, not adapter defects: the debug
SDK build asserts natively if CONTACT_POINTS queries are registered on a
scene whose earlier rollout produced deep self-collisions (the runner uses a
fresh reference scene per contact probe), and the openarm arms' authored
armature of 0 needs a damping floor (`armature_floor=0.05` in the profile) —
with the default floor the viewer sweep diverges the solver deterministically
around frame 141 of the demo.

The historical stage-3 report records the blockers before floating-root support (see
`docs/superdex-bots-qualification/blocked-probes.json`): floating FREE roots
(superseded for the qualified hands below), actuator/sensor components (stage 6; the dg5f *seed* variants and
`fr3_dg5f_short_seed` carry 5 each), mechanical cycles (`fr3_v2_2f_85`
compiles 2 from its 2f_85 attachment; `2f_85` also has a FREE root),
SPHERICAL joints (oculus_xr hands), and the 0-DoF `openarm_v20_torso`, which
loads in serial mode but is rejected by the default batch executor
(`SceneBatchExecutor requires articulated actors with DoFs`). The pytest
regression subset is `tests/test_superdex_bot_qualification.py` (6 tests,
opt-in via `SUPERDEX_ASSETS_PATH`); it also guards that every stage-1
recipe-candidate is either qualified or precisely blocked, so silent drops
fail CI.

## Audited MJCF profile

The cold importer accepts one articulation tree, one optional free root,
hinge/slide joints, scalar stateless motor or linear position actuators and
authored static planes. Existing scene fragments and named keyframes are
materialized before stepping. Joint and actuator ordering remain distinct.
Mass, inertial frame/COM, joint frames/axes, armature, joint friction and
control/force limits are mapped explicitly.

Dynamic primitive collision geometry is triangulated and baked to SDF once
during materialization. Separate welded geometry links retain authored
geom-pair contact sensor identity; their mass/inertia parts sum to the original
body's inertial properties. Mesh collision, arbitrary multiple joints per body,
multiple articulations, equality/tendon/flex/mocap/hfield/plugin features and
unsupported actuator/sensor semantics are rejected. Visual mesh files still
need to be present for the source MJCF parser even though this adapter is
headless. No model parsing or SDF baking occurs during reset, step or getters.

SuperDex contact and its implicit integration are not numerically equivalent to
MuJoCo. Primitive SDFs approximate analytic surfaces, and solver settings have
different meanings. Torsional/rolling friction requires the explicit
`superdex_allow_contact_approximation=True` experimental profile, which warns
that only the sliding Coulomb component is preserved. The default rejects that
loss of semantics. Go2's task owner opts into this profile; a finite rollout is
not evidence of locomotion quality or equivalent contacts.

The 1.0.0 wheel lacks the newer source tree's per-pair friction override API.
The importer therefore factors authored sliding-friction pairs into native
actor coefficients so their geometric-mean mixing reproduces the selected
MuJoCo pair coefficient. Incompatible friction graphs are rejected; no private
engine API or silently changed mixing rule is used.

## State, controls and sensors

Free-root public qpos is world xyz + **wxyz**, followed by single-DoF joints.
Public reset qvel is world body-origin linear velocity + **body-frame angular
velocity**, followed by joint velocity. Native SuperDex free qpos stores a
rotation vector, but its free rotational velocity is **not** the ordinary
derivative of that vector. With an identity native reference transform, native
free qvel uses world origin linear velocity and world angular velocity. The
adapter rotates the angular component at the state barrier and verifies body
origin/COM velocity against authored MuJoCo kinematics at nontrivial poses.
Native bots may also author parent-joint and joint-link reference transforms.
For those roots the adapter composes both transforms, rotates native velocities
from the parent joint's axes, and accounts for the root-origin velocity induced
by angular motion around a translated joint. Both translations and rotations
are covered by independent SDK transform/Jacobian checks.

The pre-step control callback runs once per physics substep. Motor/position
controls respect authored order, gains, gear and limits. Pending body forces
are accumulated as generalized forces and submitted together with control;
one native external-force write cannot erase a separate control contribution.

Named joint position/velocity, frame pose/axis/velocity, gyro and velocimeter
signals are reconstructed from native state into NumPy caches. Supported
plane/geom `contact data="found" num="1"` signals use native contact points and
the actual actor pair, not a nonzero-force proxy. Contacts represent the last
completed physics solve. A reset clears the solved contact state; `step(0)` does
not rebuild the contact manifold after teleportation, so the first positive
physics step supplies fresh contact results. Do not use reset-time contact
flags as a geometric-overlap test.

Authored accelerometers are recognized but unavailable: requesting/binding one
raises `NotImplementedError`, because the public runtime does not supply
instantaneous point acceleration. Unused accelerometers do not prevent loading
an otherwise supported asset; no zero or finite-difference substitute is
presented as the authored sensor. Native bot sensor components, cameras,
arbitrary force/touch sensors and site Jacobians are outside this profile.

Reset restores a private initial dynamic snapshot, writes selected qpos/qvel,
clears controls/external forces and refreshes kinematic caches. Other rows are
unchanged. Snapshot bytes are not exposed as portable checkpoints. Model DR,
rendering/video, ROM/soft/tactile state and GPU batched physics are unsupported
and must not be advertised by callers. Playback uses the shared offline MuJoCo
renderer when a visual MJCF model is available.

## Validation

```sh
uv run --no-sync pytest -q tests/test_superdex_contract.py tests/test_superdex.py \
  tests/test_superdex_materialization.py tests/test_superdex_fr3_qualification.py
UV_NO_SYNC=1 make check
uv lock --check
make package
```

Set `SUPERDEX_ASSETS_PATH` to the repository-local asset copy at
`assets/superdex` to include the native FR3 fixture. The copy is a verified
byte-for-byte duplicate of the local SuperDex asset checkout; provenance,
dependency resolution and per-entry capability records live in
`docs/superdex-assets-inventory.md`, generated by
`scripts/copy_superdex_assets.py` (use `--report-only` to rebuild the report
and re-verify without re-copying), and loading it never requires the source
checkout. Other numerical tests use small authored models
and require the optional Python 3.12 runtime. Contract/import tests also run
without it. UniLab owns task rollouts, training checkpoints and sim2sim policy
I/O validation; those outcomes are tracked in the roadmap's integration child.


## Native floating hands (stage 4)

The native loader supports a FREE root with authored `parentLinkFromJoint`
and `parentJointFromLink` translations and rotations, followed by fixed, revolute
or prismatic joints. Root position is world xyz; orientation is a wxyz
quaternion. Root velocity stores world linear velocity and body-frame angular
velocity. Scalar joint positions start at index 7, velocities at index 6;
controls exclude the six unactuated root degrees of freedom. Additional free
joints remain unsupported. Stage 8 extends this profile with spherical joints
and cycles in one serial environment; Stage 6 covers the built-in components.

Run the local, manifest-verified qualification (SDK required):

```bash
uv run scripts/superdex_floating_qualify.py --all-floating \
  --out docs/superdex-floating-qualification/all-models
SUPERDEX_ASSETS_PATH="$PWD/assets/superdex" uv run pytest -q tests/test_superdex_native_floating.py
```

Allegro V5 right (16 joints) and DG5F Short left (20 joints) passed 1,040
steps in each of serial and batch modes against independent SDK scenes,
including nonidentity root orientation, nonzero root/joint velocities, state
round trips, full/selective reset, two-environment isolation and recreation.
The command fails if the runtime, asset tree or any check is missing/failing.
These checks qualify floating state and lifecycle; they do not qualify
robot–object contact or sensor/actuator components.
See [the report and current compatibility limits](superdex-floating-qualification/README.md).

The expanded audit (`--all-floating`) discovers all 22 compiled FREE-root
models in the local bundle. Ten pass: Allegro V5 left/right, DG5F Short and
Long left/right, unactuated Wuji left/right, and OpenArm V20 left/right grippers.
Twelve retain explicit later-stage
blockers; see the [full table](superdex-floating-qualification/all-models/compatibility-table.md).
The pytest qualification is parameterized over all ten passing models. Separate
offscreen viewer smoke checks cover all ten; the standalone runner is
`uv run scripts/superdex_floating_viewers.py`. SDK-free tests also cover root
frame math and viewer cleanup on window-close events and initialization errors.

The existing viewer accepts all ten floating profiles, for example:

```bash
uv run scripts/superdex_bot_viewer.py --bot allegro_v5_right
uv run scripts/superdex_bot_viewer.py --bot wuji_hand2_beta1_left
```

Its viewer profile applies per-link gravity compensation through UniSim's
force API to keep the hand in frame, then starts a slow root translation and
rotation alongside a bounded finger sweep. Reset restores the full root and
joint state. The numerical qualification uses ordinary gravity without this
viewer compensation. The offscreen runner records numerical movement/reset
reports and logs without saving images.

## Rigid objects and nested prefabs (stage 5)

A native `.superdex_bot` can now own independent rigid objects alongside its
articulation. Pass `.mochi_prefab` files in `SceneCfg.fragment_files`; their path
resolution follows the existing scene-fragment rule. The SDK resolves nested
prefabs, names, transforms, collision shapes and render models on the cold path.
Use named nested instances to disambiguate repeated actors. Body names must be
unique across the robot and all prefab instances.

```python
scene = SceneCfg(
    str(assets / "bots/arms/fr3_v2/fr3_v2.superdex_bot"),
    fragment_files=[str(assets / "prefabs/sphere/sphere.mochi_prefab")],
)
```

That minimal example preserves the sphere's authored origin. For a placed
contact fixture, use the runnable example below, which writes temporary nested
wrappers around the unchanged FR3, sphere and nine-hole peg-board assets.

The existing public state interfaces cover the whole scene:

- Robot coordinates and action indices retain their existing order. Each dynamic
  rigid body then appends seven `qpos` entries (world xyz and wxyz quaternion)
  and six `qvel` entries (world body-origin velocity and body-frame angular
  velocity). Use `get_root_state_layout(body_name)` to obtain the indices.
- `get_state`, `set_state`, `get_default_qpos`, `get_init_qvel` and
  `get_physics_state` include all dynamic objects. Authored object velocities
  are preserved. Body getters include static and dynamic prefab bodies; static
  objects have no generalized coordinates and report mass/COM offset as zero.
  Joint/actuator getters continue to describe only the robot.
- `reset()` restores every actor, including the static fixtures. `reset(env_ids)`
  and `set_state(env_ids, qpos, qvel)` touch only the selected scenes. A state
  round trip restores pose and velocity, not the solver's internal history:
  `set_state` first restores the initial native snapshot, then applies the
  supplied coordinates and clears controls/pending forces, as for robot-only
  scenes. Static transforms remain authored and are not independently mutable.
- `apply_body_force` accepts dynamic rigid objects with world-frame COM forces
  and torques. Static prefab targets are rejected. Each scene owns its rigid
  actors, including partially constructed instances on failure; closing destroys
  the scenes and releases the loaded prefab resources before runtime shutdown.

Both serial and batch execution are supported. Object readback happens after the
native worker barrier; each environment has independent actor handles. Native
interactive playback renders the complete scene in serial mode with one
environment. MuJoCo offline video requires a separately authored visual twin
whose generalized coordinates match this expanded state layout.

Run the numerical qualification and the viewer with the verified local bundle:

```bash
SUPERDEX_ASSETS_PATH="$PWD/assets/superdex" uv run scripts/superdex_prefab_qualify.py
SUPERDEX_ASSETS_PATH="$PWD/assets/superdex" uv run scripts/superdex_prefab_qualify.py --viewer
SUPERDEX_ASSETS_PATH="$PWD/assets/superdex" uv run pytest -q tests/test_superdex_prefabs.py
```

The viewer holds the initial state for two seconds, simulates contact for four
seconds, and restores the whole scene for inspection. Closing before reset is
reported as an incomplete demonstration. `--viewer --frames 420` runs the same
phases offscreen and records a renderer smoke report without saving images.
Reports and exact qualification limits are in
[`superdex-prefab-qualification/README.md`](superdex-prefab-qualification/README.md).

This profile accepts only rigid actors and nested prefab references. Scene
settings, authored contact-filter overrides, constraints, controllers, sensors,
actuators, extra articulations and soft bodies are rejected instead of dropped.
Full `.mochi_scene` dispatch is covered separately by Stage 7 below. SDK support
for another prefab does not qualify it automatically; the stage-5 evidence covers the sphere and peg
board listed in the report. No assets or SDK downloads enter the normal tests.

## Built-in cameras, controllers and universal qualification (stage 6)

The native adapter supports link-attached `SENSOR_CAMERA` metadata and world
poses, plus explicitly configured `BASIC_JSC_PD`, `BASIC_OSC_PD` and
`MOCHI_ARTICULATED_POSE` controllers. These are SuperDex-specific methods;
`SimBackend` and the base package's optional-import boundary are unchanged.

Models including `wuji_hand2_beta1` support ordinary joint-torque control.
Unsupported actuator/sensor components are rejected before spawning and are
never silently dropped.

### Camera metadata and poses

```python
names = backend.get_camera_names()
settings = backend.get_camera_parameters(names[0])
poses = backend.get_camera_poses(names[0])          # (num_envs, 7): xyz + wxyz
poses = backend.get_camera_poses(names[0], env_ids) # selected rows
```

Settings are detached dictionaries with SDK snake_case fields: `name`,
`image_width`, `image_height`, `fov_vertical_deg`, `near_clip`, `far_clip`,
`forward_axis`, `up_axis_local`, `offset_local`, `look_at`, `look_distance`.
The returned pose is the mounted sensor frame, not an optical view matrix;
renderers must interpret the axis/offset settings separately. Translated and
rotated mounts are supported. Poses reflect stepping, state assignment and
selective reset. `get_sensor_data(camera_name)` directs callers to these
accessors. Image generation and scene-level camera authoring are out of scope.
Every spawn verifies authored camera inventory, mounts, attachments and settings.

### Explicit controller execution

```python
import json
import superdex.robotics as robotics

# A fixed-base robot: every array contains one entry per native actor DOF.
n = backend.num_actuators
backend.configure_controller(
    "BASIC_JSC_PD",
    param_args=json.dumps({"Kp": [10.] * n, "Kd": [1.] * n,
                           "saturation": [2.] * n, "deadband": [0.] * n}),
)
targets = [robotics.ControllerBasicJscPdTarget(target_pose=q)
           for q in backend.get_dof_pos()]
backend.step_controller(targets, nsteps=8)
backend.clear_controller()
```

`param_args` and `init_args` accept SDK inline JSON or parameter-file strings.
File paths passed to this API follow the SDK's working-directory rules; the
qualification CLI resolves its config-file references relative to that file.
Controller instances and histories are independent per environment. One type
is configured at a time; clear it before switching. While configured, ordinary
`step(ctrl)` and user pre-step callbacks are rejected. Clearing restores ordinary
torque stepping and removes any solver-side pose controller.

| Type | SDK target and coordinate convention |
| --- | --- |
| `BASIC_JSC_PD` | `ControllerBasicJscPdTarget.target_pose`: native actor DOF order; radians for hinges, metres for slides. A free root contributes translation XYZ and rotation-vector XYZ before joint coordinates (three per spherical joint). Gains, saturation and deadband arrays cover all native DOFs; use zero root gains for an unactuated base. |
| `BASIC_OSC_PD` | `ControllerBasicOscPdTarget.root_from_target_ee`: end-effector pose relative to the root frame reported by SDK observations. `init_args` uses unprefixed `baseLinkName` and `eeLinkName`. |
| `MOCHI_ARTICULATED_POSE` | `ControllerMochiArticulatedPoseTarget.world_from_root` plus exactly one of `pose_dofs` (all non-root native joint DOFs) or `local_to_parent_transforms` (one per robot link). |

SDK `TransformRT` quaternions use **xyzw**, unlike UniSim's camera-pose **wxyz**.
JSC/OSC read fresh native observations each physics substep and their joint
outputs are clipped to the adapter's effort limits. Nonzero free-root effort
is rejected. The pose controller uses the native implicit solver: its
saturation limits the elastic contribution, not total torque. No gravity
compensation, IK, controller composition or policy logic is added.

The current SDK cannot initialize OSC on the tested floating-base bot because
its effort-limit lookup mixes bot and actor DOF indices. The adapter reports
this as unsupported with the SDK cause; it does not change the model or solver.
JSC and articulated-pose execution are tested on fixed and floating roots.

Both execution modes compute controllers between completed physics steps;
Python callbacks do not execute inside native workers. Reset/state assignment
resets only selected controller instances, and targets are supplied afresh on
every `step_controller` call. Parameter/target errors are validated before
advancing physics; failed initialization releases partial controller instances.

### Universal qualification

Select explicit bots or the whole bundle. Optional rigid fragments and explicit
effort limits work independently of camera/controller support:

```bash
uv run scripts/superdex_component_qualify.py \
    --bots bots/arms/fr3_v2/fr3_v2.superdex_bot --effort-limit 87,87,87,87,12,12,12
uv run scripts/superdex_component_qualify.py \
    --bots bots/hands/wuji_hand2_beta1/left/wuji_hand2_beta1_left.superdex_bot \
    --effort-limit 1 --scene prefabs/sphere/sphere.mochi_prefab
uv run scripts/superdex_component_qualify.py --all
uv run scripts/superdex_component_qualify.py \
    --bots bots/arms/fr3_v2/fr3_v2.superdex_bot --effort-limit 87,87,87,87,12,12,12 \
    --controller-config docs/superdex-component-qualification/configs/fr3-jsc.json \
    --out /tmp/fr3-jsc-qualification
uv run pytest -q tests/test_superdex_component_qualification.py
```

Controller configuration files contain `type_name`, `param_args` and `init_args`;
the latter two are SDK JSON/file strings. Examples for all three controllers
live in `superdex-component-qualification/configs/`. The runner builds its own
SDK reference, compares 1,040 physics steps in batch and serial modes, checks
camera metadata/poses, rigid fragments, effort clipping, reset, isolation and
lifecycle. It reports absent cameras as skipped. Unsupported models are blocked;
other exceptions and numerical mismatches fail. `--all` continues after either
and exits nonzero if any model fails or is blocked. Qualification is evidence
for the recorded fixtures, not a guarantee for every asset.


## Native scenes (stage 7)

`SceneCfg.model_file` accepts `.mochi_scene` files with one root-file articulation
containing fixed/revolute/prismatic joints, including a prismatic first joint,
plus independent rigid actors and nested rigid prefabs. Additional rigid
`SceneCfg.fragment_files` use the existing path-resolution rule. Set
`SUPERDEX_ASSETS_PATH` to the local bundle root for bundle-relative mesh references.

```python
import os
from pathlib import Path

from unisim import create_backend
from unisim.scene import SceneCfg

assets = Path(os.environ["SUPERDEX_ASSETS_PATH"])
backend = create_backend(
    "superdex",
    SceneCfg(str(assets / "benchmarks/cart_pole/cart_pole.mochi_scene")),
    num_envs=2,
    sim_dt=0.002,
    superdex_controlled_joints=["Cart"],
    superdex_effort_limits=[3.0],
)
try:
    import numpy as np

    backend.step(np.zeros((2, 1)))  # Physical force along the Cart joint axis.
    backend.reset()
finally:
    backend.close()
```

`superdex_controlled_joints` is required for scene inputs and determines action
order. Names must uniquely select active authored hinge/slide joints; fixed or
unknown joints are rejected. `superdex_effort_limits` supplies one finite
positive limit per input. Controls are clipped physical forces/torques; passive
joints remain in state but receive no selected effort. These options do not
change native bot or MJCF controls; `controlled_joints` is rejected for those inputs.

| Qualified scene | Physical effort inputs | Limits | State dimensions |
| --- | --- | --- | --- |
| Cart Pole | `Cart` | 3 | nq=nv=2 |
| Half Cheetah | `BackThigh`, `BackShin`, `BackFoot`, `FrontThigh`, `FrontShin`, `FrontFoot` | 120, 90, 60, 120, 60, 30 | nq=nv=9 |

Scene gravity and solver settings are preserved; absent values use SDK defaults.
These two scenes use gravity `[0, -9.8, 0]`, not the bot loader's negative-Z
convention. `sim_dt` remains caller-owned: the SDK scene schema has no timestep
field, and such fields are rejected. Nested scene settings must match the root's
explicit values; conflicting or otherwise unresolved settings fail. No gravity
or solver override options are added. Effective settings appear in the report.

The SDK loader retains actor names, shapes, transforms and authored layer-contact
filters. Half Cheetah's joint-tracking controller supplies rest springs while
`step()` applies external efforts. Reset restores the initial spring targets and
velocities without destroying snapshot-owned controller entities. The explicit
bot controller API cannot replace an authored scene controller. Native coordinates
are preserved, and there is no implicit ground plane: the benchmark application
adds ground separately. UniLab owns action normalization, rewards and training.

All articulation DoFs precede dynamic rigid-object coordinates. Each dynamic
rigid object adds world xyz/wxyz qpos and world-origin linear/body-angular qvel;
static fixtures add body entries but no coordinates. Existing body, force,
full/selective reset and state-roundtrip APIs apply to the complete scene.
State round trips restore kinematics, not hidden solver history. Each environment
owns its actors/controllers; failed partial instantiation is cleaned up.

The historical Stage 7 profile excluded multiple articulations, FREE/spherical scene joints,
mechanical cycles, soft bodies, scene cameras/plugins, arbitrary constraints,
non-rigid nested prefabs and controllers other than joint-tracking pose springs.
Stage 8 extends those rigid capabilities below. Deformables and custom
components remain deferred, and unsupported fields fail explicitly.

```bash
SUPERDEX_ASSETS_PATH="$PWD/assets/superdex" uv run --no-sync scripts/superdex_scene_qualify.py
SUPERDEX_ASSETS_PATH="$PWD/assets/superdex" uv run --no-sync pytest -q tests/test_superdex_scenes.py
uv run --no-sync scripts/superdex_scene_qualify.py --viewer --scenes half_cheetah
uv run --no-sync scripts/superdex_scene_qualify.py --viewer --frames 240
```

The [Stage 7 report](superdex-scene-qualification/README.md) records 1,000 steps
per scene in each execution mode, with two environments, zero measured SDK
state/body-position deviation and passing reset/isolation/control/lifecycle checks.
Both 240-frame renderer smokes passed with zero reset error. Manual visual
inspection remains outstanding; no other scene is qualified by these results.


## Complete native rigid-body profile (Stage 8)

Use externally authored assets through `SceneCfg` and `create_backend`; no scene
builder is introduced. Inputs include native bots, bot archives, standalone
rigid/articulated prefabs and scenes with zero, one or multiple articulations.
New profiles require `num_envs=1` and `superdex_execution_mode="serial"`.
Previously qualified batch profiles retain their regression coverage.

```python
backend = create_backend(
    "superdex", SceneCfg("/path/to/authored_scene.mochi_scene"), 1, 0.002,
    superdex_execution_mode="serial",
    superdex_controlled_joints=["arm/shoulder", "hand/ball"],
    superdex_effort_limits=[10, 1, 1, 1],  # hinge plus three spherical DOFs
)
```

Spherical qpos uses native joint-frame rotation-vector XYZ, with three native
velocity and effort coordinates. The bare joint name expands to its three DOFs;
explicit `/x`, `/y`, `/z` names select individual coordinates. Metadata records
actual joint ownership, rather than guessing from these suffixes. Closed-loop
constraints and transmissions do not add action coordinates. Limits stay in the
native solver and effort limits apply per selected control coordinate.

`backend.model.articulations` records each articulation's qpos/qvel and link
slices. Each free root adds seven qpos and six qvel values; roots retain the
world-position/wxyz and world-origin-linear/body-angular convention, including
nested rotated/translated reference frames. Dynamic rigid objects follow all
articulations. `get_root_state_layout()` exposes each root. Multiple articulations
use actor-qualified names, and ambiguous repeated actor names gain stable
`#index` metadata suffixes without renaming native actors. Whole-joint index
queries expand to all coordinates. `model.joint_coordinate_groups` exposes the
mapping. Body forces use the owning articulation's Jacobian and every scene
advances exactly once per substep.

For articulated scenes/prefabs, explicitly select controls or provide `[]` for
passive operation. Object-only scenes have zero actions. Authored constraints,
tracking controllers (joint, link position and link rotation), contact filters,
articulated skin, settings and initial velocities remain SDK-owned. State round
trips restore kinematics; reset restores authored controller targets and scene
state. No ground plane, coordinate conversion or scene assembly is implicit.

Bot archives and tagged external bot dependencies use the SDK resolver. For
physics prefabs/scenes, use the nearest `.superdex_root`, or set
`SUPERDEX_ASSETS_PATH` for root-relative dependencies; `./` paths resolve beside
the containing file. External roots do not need the historical bundle layout or
checksum. Qualification fixtures remain local and independent of the SDK checkout.

See the [Stage 8 report](superdex-rigid-qualification/README.md) for all 55 native
target files, 11 synthetic fixtures, SDK/build fingerprints and exact evidence.
The passive torso runner uses one serial environment. Runtime URDF's documented
primitive-geometry loss and the tested floating OSC limitation are SDK blockers,
not adapter support claims. Deformables, custom components, new batch execution,
image rendering, conversion and solver work remain deferred.
