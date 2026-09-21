# SuperDex adapter guide

[中文版](superdex_zh.md)

Use this guide to load local models, run SuperDex through UniSim, inspect a model,
and check adapter-versus-SDK agreement. **SuperDex computes the physics; UniSim
provides the shared interface and translates model, control and state data.**
UniLab remains responsible for tasks, action normalization, rewards, observations,
training and policy rollouts.

The English guide and its Chinese translation describe the same implementation.
Keep their examples, support boundaries and limitation reports synchronized.

## Contents

1. [Install and run a first model](#quick-start)
2. [Choose an input and execution mode](#inputs-and-modes)
3. [Use the Python API](#python-api)
4. [Understand state and control values](#state-and-control)
5. [Use cameras and controllers](#cameras-and-controllers)
6. [View and compare models](#tools)
7. [Resolve and verify assets](#assets)
8. [Understand the code and public interface](#architecture)
9. [Check current limitations and unsupported assets](#limitations)
10. [Troubleshoot common failures](#troubleshooting)
11. [Validate changes and track remaining work](#validation)

<a id="quick-start"></a>
## 1. Install and run a first model

Run commands from the UniSim repository root. The examples require the local asset
bundle; cloning the repository does not download its ignored asset payloads.

```bash
uv sync --python 3.12 --extra superdex --extra mujoco
export SUPERDEX_ASSETS_PATH="$PWD/assets/superdex"

uv run --no-sync scripts/superdex_viewer.py \
    benchmarks/cart_pole/cart_pole.mochi_scene \
    --controlled-joints Cart --effort-limit 3.0
```

This opens one native viewer through the UniSim adapter. `Cart` selects the cart's
prismatic joint; `3.0` is its force limit in newtons, **not a constant applied force**.
The viewer supplies a small demonstration effort. Close the window or press Ctrl+C
to release the viewer and backend.

For a bounded renderer check without an interactive window:

```bash
uv run --no-sync scripts/superdex_viewer.py \
    benchmarks/cart_pole/cart_pole.mochi_scene \
    --controlled-joints Cart --effort-limit 3.0 --frames 9
```

No images, videos or JSON reports are saved by default. The viewer can find the
repository's default bundle without an environment-variable export; the Python
examples and asset qualification commands below explicitly set it for clarity.

### Runtime requirements

| Item | Current integration boundary |
| --- | --- |
| Python | CPython 3.12 or 3.13 for the SuperDex wheels |
| SDK packages | `superdex-physics-uni==1.0.0`, `superdex-robotics-uni==1.0.0` |
| Verified configuration | Linux x86_64, CPU, FP32 |
| Other platforms / FP64 | Not qualified by this integration |
| GPU physics | Not enabled; the tested wheel does not provide CUDA solvers |
| Base UniSim import | Remains usable without importing an engine SDK |

The `superdex-uni` packages are the integration builds carrying the native batch
executor. They share the `superdex/` namespace with upstream packages; do not
co-install the two sets. The x86 build requires AVX2 and related instructions.
MuJoCo 3.11 is a model-loading parser for audited MJCF inputs and an optional
playback renderer; **it does not perform the SuperDex simulation steps**.
Native bot/scene loading does not use the MJCF parser. SuperDex Lab, Gymnasium and
a learner are not adapter dependencies.

`uv run --no-sync` preserves the installed environment, including editable SDK
or project overrides. For sibling UniLab development, an editable setup is
`uv pip install -e './[superdex,mujoco]' -e ../UniLab`; `UNILAB_LOCAL_UNISIM` identifies
the exact checkout in UniLab's local provenance checks. A PyPI release is not
required for this local integration. Historical project context is tracked in
[UniLab roadmap #1533](https://github.com/Motphys/UniLab/issues/1533).

<a id="inputs-and-modes"></a>
## 2. Choose an input and execution mode

**Support depends on the file's contents and the execution mode, not just its suffix.**

| Input | Role and current support |
| --- | --- |
| `.superdex_bot` | Robot with HARD (fixed) or FREE (floating) root; supported joint/component profile only |
| `.superdex_bot_archive` | Packed bot plus dependencies; serial mode, one environment |
| `.mochi_scene`, `.mochi_prefab` | Direct scene inputs: rigid actors, articulations, nested prefabs, supported constraints, contact filters and authored pose controllers |
| MJCF `.xml` | Audited subset described below; not an arbitrary MuJoCo model |
| `.urdf` | Direct loading is unsupported |
| `.mochi.h5`, meshes, textures, CAD files | Supporting payloads; not standalone simulation inputs |
| `.superdex_controller`, sensor configuration files | Configuration used alongside a compatible model; not standalone models |

`rods/helix_with_visual.mochi.h5`, for example, is shape data for a deformable-rod
example, not a rigid model that the viewer can directly simulate.

### Serial versus batch

| Mode | Behavior | Use |
| --- | --- | --- |
| `serial` | Steps scenes on the calling thread | Native viewer/debugger; advanced profiles |
| `batch` (default) | Uses the native `SceneBatchExecutor` and persistent C++ workers | Supported multi-environment simulation |

Ordinary supported bots can use either mode. Archives, spherical joints, closed
loops, coupled actuation, multiple articulations, constraints and advanced
link-tracking controllers require **serial mode and `num_envs=1`** under the
current audited profiles. Interactive native playback also requires that combination.
A feature being supported in serial mode does not imply it is supported in batch mode.

The native debugger's `DebugDraw` is thread-affine. Attaching it to a batch backend
is rejected to prevent native thread-affinity failures. Use
`superdex_execution_mode="serial"` before attaching. In UniLab the setting is
`env.superdex_execution_mode=serial`; interactive evaluation also uses one play environment.

In batch mode, `superdex_num_workers=0` selects the physical cores visible to the
process. Leave that outer scene-worker setting at zero in serial mode; a nonzero
value is rejected.

`superdex_num_worker_threads` controls the process-wide SDK-internal solver pool in
serial mode: `-1` lets SuperDex choose, `0` (the default) is single-threaded, and a
positive value requests that many threads. Nonzero values are rejected in batch mode
to avoid competing thread pools. All live SuperDex backends in one process must use
the same value because they share the SDK runtime.

### Audited MJCF boundary

Supported: one articulation tree, an optional free root, hinge/slide joints,
scalar stateless motor or linear position actuators, static planes, named keyframes
and supported fragments. Mass, inertia/COM, frames/axes, armature, joint friction
and actuator limits are translated at load time. Dynamic primitive collision
geometry is triangulated and baked to SDF during loading.

Contact is not numerically equivalent to MuJoCo. Torsional/rolling friction needs
`superdex_allow_contact_approximation=True`; this experimental option warns that
only sliding Coulomb friction is retained. The SDK lacks per-pair friction overrides:
the importer factors sliding-friction pairs into actor coefficients and rejects
incompatible graphs instead of silently changing them.

<a id="python-api"></a>
## 3. Use the Python API

Export `SUPERDEX_ASSETS_PATH` as in section 1 before running these examples.
The Cart Pole, FR3 and controller examples are **independent programs**. Do not
create a second backend by overwriting an existing variable without closing it.

### Load, step, read state and reset a scene

```python
import os
from pathlib import Path

import numpy as np

from unisim import create_backend
from unisim.scene import SceneCfg

assets = Path(os.environ["SUPERDEX_ASSETS_PATH"]).expanduser().resolve()
backend = create_backend(
    "superdex",
    SceneCfg(str(assets / "benchmarks/cart_pole/cart_pole.mochi_scene")),
    num_envs=1,
    sim_dt=0.002,
    superdex_execution_mode="serial",
    superdex_controlled_joints=["Cart"],
    superdex_effort_limits=[3.0],
)
try:
    print(backend.get_actuator_names())  # ("Cart",)
    print(backend.get_model_info())
    command = np.array([[1.0]])  # One environment, one actuator; force in N.
    backend.step(command, nsteps=8)
    state = backend.get_state()
    print(state["qpos"], state["qvel"])
    backend.reset()
finally:
    backend.close()
```

`command` has shape `(num_envs, num_actuators)`. Cart Pole has two generalized
coordinates but only one selected actuator. With `sim_dt=0.002`, eight physics
steps advance **0.016 seconds of simulation time**, regardless of wall-clock speed.
Inputs are clipped to the selected effort limits.

For native scenes/prefabs, supply ordered `superdex_controlled_joints` and finite,
positive `superdex_effort_limits` for active controls. Use `[]` to make a scene
passive; its command array then has shape `(num_envs, 0)`.

### Load two independent robot environments

```python
import os
from pathlib import Path

import numpy as np

from unisim import create_backend
from unisim.scene import SceneCfg

assets = Path(os.environ["SUPERDEX_ASSETS_PATH"]).expanduser().resolve()
backend = create_backend(
    "superdex",
    SceneCfg(str(assets / "bots/arms/fr3_v2/fr3_v2.superdex_bot")),
    num_envs=2,
    sim_dt=0.002,
    superdex_effort_limits=[87, 87, 87, 87, 12, 12, 12],
)
try:
    backend.step(np.zeros((2, backend.num_actuators)), nsteps=8)
    state = backend.get_state()
    backend.set_state(np.array([0]), state["qpos"][[0]], state["qvel"][[0]])
    backend.reset(np.array([0]))  # Environment 1 is unchanged.
finally:
    backend.close()
```

The effort limits follow actuator order. A zero effort command does not hold a
robot in place: gravity and passive dynamics still act. This example uses the
default batch mode. `set_state()` writes the selected rows; `reset([0])` restores
environment 0 to its authored initial state while leaving environment 1 unchanged.

### Compose a robot with rigid objects

Using the imports and `assets` variable from the FR3 example, run this as a
separate block after closing any previous backend:

```python
scene = SceneCfg(
    str(assets / "bots/arms/fr3_v2/fr3_v2.superdex_bot"),
    fragment_files=[str(assets / "prefabs/sphere/sphere.mochi_prefab")],
)
backend = create_backend(
    "superdex", scene, num_envs=1, sim_dt=0.002,
    superdex_execution_mode="serial",
    superdex_effort_limits=[87, 87, 87, 87, 12, 12, 12],
)
try:
    backend.step(np.zeros((1, backend.num_actuators)))
finally:
    backend.close()
```

Fragments retain their authored transforms. This does not attach the sphere to
the robot or invent a contact arrangement; use an authored wrapper prefab when
you need specific placement.

### Metadata, gravity and cleanup

On an open backend, use `get_model_info()` to inspect articulations, coordinate
names/units/representations, coordinate groups and body ownership. Use
`get_actuator_names()` and `get_joint_state_qpos_indices()` /
`get_joint_state_qvel_indices()` instead of guessing action/state order.

`get_gravity()` reads gravity; `set_gravity([0, 0, -3])` changes it and the override
persists across resets. Native scenes preserve authored gravity and solver settings;
missing values use SDK defaults. No ground plane or coordinate rotation is inserted.
Nested explicit settings must agree with the root. `sim_dt` is caller-owned;
scene timestep fields are rejected rather than silently ignored.

| Qualified scene | Ordered controlled joints | Effort limits | Dimensions |
| --- | --- | --- | --- |
| Cart Pole | `Cart` | 3 | `nq=nv=2` |
| Half Cheetah | `BackThigh`, `BackShin`, `BackFoot`, `FrontThigh`, `FrontShin`, `FrontFoot` | 120, 90, 60, 120, 60, 30 | `nq=nv=9` |

Both benchmark scenes author gravity `[0, -9.8, 0]`; bot loading uses negative Z.

Each environment owns a native scene. UniSim reference-counts the shared
process-wide SDK runtime; closing one backend leaves other live backends usable.
Let UniSim own initialization, do not transfer live backends between processes,
and always call `close()` (or the public `cleanup_scene_assets()` lifecycle hook).
UniLab's `env.close()` calls that hook.

<a id="state-and-control"></a>
## 4. Understand state and control values

### State layout and coordinate frames

`get_state()` returns detached NumPy arrays, including `qpos` and `qvel`, with one
row per environment. Their column counts can differ.

| Part | Public `qpos` | Public `qvel` |
| --- | --- | --- |
| Floating root | World xyz + quaternion **wxyz** (7 values) | World body-origin linear velocity + **body-frame** angular velocity (6 values) |
| Revolute joint | Angle in radians | Angular velocity in rad/s |
| Prismatic joint | Displacement in metres | Velocity in m/s |
| Spherical joint | Native joint-frame rotation-vector XYZ (3 values) | Three native rotational velocity coordinates, not a naive rotation-vector derivative |
| Dynamic rigid object | World xyz + wxyz (7 values) | World body-origin linear velocity + body-frame angular velocity (6 values) |

A floating robot with seven scalar joints therefore has `nq=14`, `nv=13`.
Static objects have body entries but no dynamic coordinates. Dynamic objects
append their state windows; complex scenes need metadata-based indexing.
`get_root_state_layout(body_name)` returns the appropriate root indices.
Body getters ending in `_w` return world-frame quantities; do not confuse their
angular velocities with the body-frame root angular velocity stored in `qvel`.

SuperDex internally uses a rotation vector for a free root and different native
velocity conventions. The adapter converts those representations, composes authored
parent-joint/joint-link transforms, and accounts for velocity offsets caused by
rotation around translated joints. Nontrivial transform/Jacobian regressions verify
these conversions. Do not manually reinterpret native arrays as public state.

For spherical joints, a bare joint selection expands to three coordinates;
`/x`, `/y`, `/z` names select individual coordinates. Constraints and transmissions
do not add action coordinates. Use metadata to determine actual ownership.

### Efforts, position targets and substeps

| Control path | Meaning of the input |
| --- | --- |
| Native bot or selected native scene joint: `step(ctrl)` | Physical effort: N for prismatic joints, N·m for revolute joints |
| Audited MJCF motor | Authored motor input, respecting gear and limits |
| Audited MJCF position actuator | Position target converted using authored gains and limits |
| `step_controller(targets)` | Controller-specific targets; controller output is recalculated per physics substep |

Do not interpret every backend/actuator's `step()` input as the same physical
quantity. Task action normalization belongs to UniLab. `set_pre_step_control()`
also runs once per physics substep. Pending body forces and actuator forces are
combined so one native force write cannot erase another contribution.
`apply_body_force()` uses world-frame COM forces and torques for dynamic bodies;
static targets are rejected.

### Reset, contacts and sensors

Reset restores an initial native dynamic snapshot, clears controls/pending forces
and refreshes caches. State assignment restores public kinematics, not every hidden
solver-history value; snapshot bytes are not portable checkpoints.

Named joint/frame signals, gyro and velocimeter values are reconstructed from
native state. Supported MJCF plane/geom `contact data="found" num="1"` sensors use
native contact points and the actual actor pair. Contact data describes the **last
completed solve**: after teleporting or resetting, take a positive physics step
before expecting fresh contacts. A zero-duration solve does not rebuild the manifold.

The audited MJCF path recognizes accelerometers but cannot provide instantaneous
point acceleration. Requesting/binding them raises `NotImplementedError`; unused
accelerometers do not prevent loading that profile. This is separate from native
bot component validation, which currently accepts only `SENSOR_CAMERA` components.

<a id="cameras-and-controllers"></a>
## 5. Use cameras and controllers

### Camera metadata and mounted poses

On an open **native bot/archive with authored cameras**, and with NumPy imported:

```python
for name in backend.get_camera_names():
    settings = backend.get_camera_parameters(name)
    poses = backend.get_camera_poses(name)  # (num_envs, 7): xyz + wxyz
    selected = backend.get_camera_poses(name, np.array([0]))
    print(name, settings, poses, selected)
```

These APIs expose metadata and poses, **not image pixels**. Empty camera inventories
are valid, which is why the example iterates instead of indexing the first camera.
`get_sensor_data(camera_name)` directs callers to the camera accessors.

Settings are detached dictionaries with SDK snake_case fields: `name`,
`image_width`, `image_height`, `fov_vertical_deg`, `near_clip`, `far_clip`,
`forward_axis`, `up_axis_local`, `offset_local`, `look_at`, `look_distance`.
Dimensions are pixels, FOV is degrees, and distances are metres. The returned pose
is the mounted sensor frame, not a renderer's optical view matrix; interpret the
axis/offset settings separately. Poses update after stepping, state writes and
selective reset. Spawn-time validation checks camera inventory, mounts and settings.
Do not infer native scene-camera support from native bot-camera support.

### Built-in controllers

| Controller | Shared target type | Meaning |
| --- | --- | --- |
| `BASIC_JSC_PD` | `JointTarget` | Desired joint positions |
| `BASIC_OSC_PD` | `CartesianTarget` | Desired end-effector pose in the documented controller frame |
| `MOCHI_ARTICULATED_POSE` | `ArticulationPoseTarget` | Desired root/articulation pose |

Call `get_controller_descriptions()` for model-specific availability. These APIs
require a compatible native bot/archive; the Cart Pole scene example is not a
valid backend for configuring these bot controllers. Authored scene pose controllers
are a separate scene-loading feature.

This complete example uses fixed-base FR3 and holds its initial joint target:

```python
import json
import os
from pathlib import Path

from unisim import JointTarget, create_backend
from unisim.scene import SceneCfg

assets = Path(os.environ["SUPERDEX_ASSETS_PATH"]).expanduser().resolve()
backend = create_backend(
    "superdex",
    SceneCfg(str(assets / "bots/arms/fr3_v2/fr3_v2.superdex_bot")),
    num_envs=1, sim_dt=0.002,
    superdex_execution_mode="serial",
    superdex_effort_limits=[87, 87, 87, 87, 12, 12, 12],
)
try:
    print(backend.get_controller_descriptions())
    n = backend.num_actuators
    backend.configure_controller(
        "BASIC_JSC_PD",
        param_args=json.dumps({
            "Kp": [10.0] * n,
            "Kd": [1.0] * n,
            "saturation": [2.0] * n,
            "deadband": [0.0] * n,
        }),
    )
    targets = [JointTarget(q.copy()) for q in backend.get_dof_pos()]
    backend.step_controller(targets, nsteps=8)
    backend.clear_controller()
finally:
    backend.close()
```

For PD control, `Kp` multiplies position error and `Kd` provides damping; saturation
limits output. Configure one controller per environment, use `step_controller()`,
then clear it before switching control paths. A pre-step callback and a configured
controller cannot be active together. The example's parameter-array lengths are
for fixed-base FR3; native floating-root configurations may need root placeholders.
Floating-base OSC is explicitly unavailable because of the SDK indexing limitation.

Native SDK targets are deprecated; use the shared target types. Configuration
files contain `type_name`, `param_args`, `init_args`, with SDK JSON strings or file
paths. CLI configuration-file paths resolve relative to the config file.
Absolute-target JSON schemas and ready-made FR3 examples are in
[superdex-configs/](superdex-configs/); CLI targets must match the configured type.

<a id="tools"></a>
## 6. View and compare models

### Viewer: inspect geometry and behavior

```bash
uv run --no-sync scripts/superdex_viewer.py bots/arms/fr3_v2/fr3_v2.superdex_bot
uv run --no-sync scripts/superdex_viewer.py prefabs/sphere/sphere.mochi_prefab
uv run --no-sync scripts/superdex_viewer.py bots/arms/fr3_v2/fr3_v2.superdex_bot --compare
uv run --no-sync scripts/superdex_viewer.py --list-profiles
```

| Option / mode | Behavior |
| --- | --- |
| Normal viewer | Loads and steps through the adapter's public API; `run_playback()` opens the native viewer |
| `--compare` | Adds an independent direct-SDK simulation/window; both advance together and close together |
| Registered bot profile | Default-pose hold, bounded movement and reset demonstration |
| Unknown supported model | Passive stepping; no invented control profile |
| `--controlled-joints Cart --effort-limit 3.0` | Selects scene/prefab controls and their scalar limit |
| `--controller-config FILE --absolute-target FILE` | Uses a compatible bot/archive controller and matching target |
| `--fragments FILE ...` | Adds rigid prefab fragments to a bot/archive |
| `--no-gravity` | Overrides gravity with zero for this session |
| `--frames N` | Explicit headless renderer smoke; at least 6 frames |
| `--out results/viewer.json` | Opt-in JSON evidence file |

Passive does not mean frozen: gravity and authored dynamics still act. Smoke runs
exercise startup/stepping/reset/cleanup but do not establish human visual approval.
The viewer saves no images or videos. SDK comparison accepts native inputs; audited
MJCF retains the single adapter viewer and its existing pytest coverage.

The backend also has a separate **offline recording** path when a compatible
visual MJCF model is available (`SceneCfg.visual_model_file` for native assets).
That path uses the shared MuJoCo renderer and can work in serial or batch mode;
it is not camera image capture and is not the viewer script's `--frames` mode.
The native interactive path rejects debug-overlay and `on_frame` callbacks.

### Comparison: verify adapter-versus-SDK agreement

```bash
export SUPERDEX_ASSETS_PATH="$PWD/assets/superdex"
uv run --no-sync scripts/superdex_compare.py bots/arms/fr3_v2/fr3_v2.superdex_bot
uv run --no-sync scripts/superdex_compare.py --all
uv run --no-sync scripts/superdex_compare.py --all --match hand
uv run --no-sync scripts/superdex_compare.py --fixtures
uv run --no-sync scripts/superdex_compare.py --all --out results/superdex
uv run --no-sync scripts/superdex_compare.py bots/arms/fr3_v2/fr3_v2.superdex_bot \
    --controller-config docs/superdex-configs/fr3-jsc.json \
    --absolute-target docs/superdex-configs/fr3-jsc-absolute.json
```

Both sides use **SuperDex physics**. One side is loaded/stepped through UniSim;
the reference is independently materialized through the SDK. Matching conditions
include initial state, controls, timestep, gravity and solver settings. This tests
adapter fidelity, not equivalence between SuperDex and a different physics engine.

`--all` discovers models in the repository-local roots; `--roots DIR ...` chooses
explicit roots. `--fixtures` generates regression models temporarily. Each model
runs in an isolated subprocess; crashes/timeouts are recorded and later models
continue. Tests cover routing/clipping, native and public state, body poses/velocities,
contacts, cameras/controllers, resets, cleanup/recreation and qualified batch modes.
Use `--composition FRAGMENT ...` with one bot to compare rigid-object composition.

| Result | Interpretation |
| --- | --- |
| Passed | The checks executed for this profile met their tolerances |
| Blocked / unsupported | A named feature prevents this profile; never counted as passed |
| Failed | An assertion, numerical check or other execution error failed |
| Crashed / timeout | Worker terminated abnormally or exceeded its time budget |
| Unverified | The required runtime was unavailable; no successful qualification |

Expected static blockers do not make an otherwise successful sweep fail. Numerical
failures, unexpected runtime rejections, missing runtimes, crashes and timeouts
produce a nonzero exit code. Read the result counts, not only the exit code.
`--out` writes `comparison.json` with checks, limitations and reproducibility
fingerprints into the specified directory. Use ignored `results/`; never commit
those outputs. No persistent report is written without `--out`.

Default bot/controller rollouts are 1040 physics steps; scenes/compositions use
1000. Bot/controller `--steps` overrides use complete eight-step intervals and
can shorten the evidence. The nested robot-contact fixture retains 1000 steps at
1 ms. A reduced run is not the full baseline, and a zero-gravity pass is not a
qualification under gravity.

<a id="assets"></a>
## 7. Resolve and verify assets

The main local bundle is `assets/superdex`; physics examples are under
`assets/superdex-physics`. Payloads/licenses stay local and are excluded from
package distributions. Maintenance uses `scripts/copy_superdex_assets.py`;
qualification must not read the SDK source checkout or modify asset payloads.

For the viewer, `--assets ROOT` takes precedence over `SUPERDEX_ASSETS_PATH`, then
the repository default. An existing explicit model path is accepted directly;
otherwise the viewer resolves it under the selected root. It passes that root to
both adapter and comparison workers. An external model without an explicit root
or environment setting retains a local-directory fallback.

Dependency resolution is distinct from locating the model file. A nearby
`.superdex_root` defines authored root references; scene `./...` dependencies are
relative to the containing document, while ordinary relative scene references use
the resolved asset root. Native bot bundles can also declare tagged dependency
roots. Preserve authored references instead of moving individual model files away
from their dependencies. Archives need `.mochi_bot_archive_metadata` naming their
packed bot and the complete dependency closure.

The tracked [JSON inventory](superdex-assets-inventory.json) records provenance,
hashes and dependencies, including packed archives. **Inventory candidates are not
runtime qualifications.** The adapter does not re-hash the entire bundle on every load.

Verify the local bundle against the existing inventory:

```bash
uv run --no-sync python - <<'PYTHON'
from pathlib import Path
from unisim.backend.superdex.assets import verify_asset_bundle

print(verify_asset_bundle(
    Path("assets/superdex"), Path("docs/superdex-assets-inventory.json")
))
PYTHON
```

After an intentional local bundle change, refresh the manifest instead:

```bash
uv run --no-sync scripts/copy_superdex_assets.py --inventory-only
```

This refresh uses the local bundle and writes only the inventory JSON. It is not
a way to repair accidentally missing assets. Keep this guide's limitations report
current when asset coverage changes; generated Markdown compatibility tables are
no longer maintained.

<a id="architecture"></a>
## 8. Understand the code and public interface

```text
Application / UniLab / normal viewer
    -> create_backend("superdex", ...)
    -> SuperDexBackend implementing SimBackend
    -> SuperDex Physics / Robotics SDK
```

The [factory](../src/unisim/factory.py) selects the adapter;
[SimBackend](../src/unisim/backend/base.py) declares the shared public interface.
All current SuperDex public method names belong to that interface. Shared
**declaration** does not imply support in every engine: defaults can raise
`NotImplementedError`, and implementations can restrict models/modes.

| Module in `src/unisim/backend/superdex/` | Responsibility |
| --- | --- |
| `__init__.py` | Exports `SuperDexBackend` and `SuperDexDependencyError` |
| `backend.py` | Main class; lifecycle, stepping, state, forces, playback |
| `components.py` | Camera validation and inherited camera/controller APIs (`BuiltinAPI`) |
| `model.py` | Layouts, coordinate/root transformations and public metadata |
| `runtime.py` | SDK discovery, runtime ownership, CPU topology |
| `materialization.py` | Bot/archive and audited MJCF materialization, geometry |
| `scenes.py` | Scene/prefab validation, loading and composition |
| `assets.py` | Inventory, dependency closure and integrity verification |

`SuperDexBackend(BuiltinAPI, SimBackend)` inherits the camera/controller methods
from `components.py`. Therefore `backend.get_camera_names()` is a public adapter
call even though its implementation is in another file.

Three common calls illustrate where declarations and implementations differ:

| Call | Declaration | Implementation used by SuperDex |
| --- | --- | --- |
| `backend.step(...)` | `SimBackend` | Override in `backend.py` |
| `backend.reset(...)` | `SimBackend` | Inherited shared implementation, which calls SuperDex's `set_state()` |
| `backend.get_camera_names()` | `SimBackend` | Override inherited from `BuiltinAPI` in `components.py` |

Camera/controller operations can therefore belong to the shared interface even
when only SuperDex currently implements them. Internal helpers such as `_refresh()`
are not public interface methods and need not appear in `SimBackend`.

These modules are not an enforced private boundary. The comparison tool directly
uses asset/runtime/scene utilities and internal actor handles to inspect fidelity.
The viewer's comparison worker also uses runtime helpers for its direct SDK scene.
Asset-maintenance callers use `assets.py` directly. Normal simulation application
code should stay on the public backend interface; private helpers/handles are not
part of its compatibility contract.

<a id="limitations"></a>
## 9. Current limitations and unsupported assets

This maintained report lists **direct model inputs** that cannot currently load.
Supporting geometry, textures, CAD and configuration files are excluded.
All ten rows below were identified by **static validation**, including recipe
closure, not by successful SDK execution. They are adapter restrictions; claims
about SDK behavior require separate direct evidence.

| Model (relative to `assets/`) | Format | Unsupported feature | Blocker | Checked | Enabling capability |
| --- | --- | --- | --- | --- | --- |
| `superdex/test/urdf/fr3v2_1_urdf/robots/fr3v2_1_franka_hand.urdf` | URDF | direct URDF model loading | adapter restriction; historical SDK primitive-geometry loss also requires investigation | static validation | URDF ingestion or conversion to the audited MJCF profile |
| `superdex/bots/fun/example_bot_2dof/example_bot_2dof.superdex_bot` | bot | custom actuator/sensor components (`MY_VELOCITY_SERVO_ACTUATOR`, `MY_CONTACT_FORCE_SENSOR`) | adapter restriction | static validation | translation of authored custom components; only `SENSOR_CAMERA` is supported |
| `superdex/bots/sensors/dg5f_seed/dg5f_seed.superdex_bot` | bot | seed sensor component (`SENSOR_SEED_V1_MLP`) | adapter restriction | static validation | learned-sensor translation |
| `superdex/bots/hands/dg5f_long_seed/left/dg5f_long_seed_left.superdex_bot` | bot | five seed sensors inherited through `AttachBot` recipes | adapter restriction | static validation (recipe closure) | learned-sensor translation |
| `superdex/bots/hands/dg5f_long_seed/right/dg5f_long_seed_right.superdex_bot` | bot | five seed sensors inherited through `AttachBot` recipes | adapter restriction | static validation (recipe closure) | learned-sensor translation |
| `superdex/bots/hands/dg5f_short_seed/left/dg5f_short_seed_left.superdex_bot` | bot | five seed sensors inherited through `AttachBot` recipes | adapter restriction | static validation (recipe closure) | learned-sensor translation |
| `superdex/bots/hands/dg5f_short_seed/right/dg5f_short_seed_right.superdex_bot` | bot | five seed sensors inherited through `AttachBot` recipes | adapter restriction | static validation (recipe closure) | learned-sensor translation |
| `superdex/bots/arm_hand_combos/fr3_dg5f_short_seed/right/fr3_dg5f_short_seed_right.superdex_bot` | bot | seed sensors inherited transitively through the attached seed hand | adapter restriction | static validation (recipe closure) | learned-sensor translation |
| `superdex/prefabs/duck_lamp/duck_lamp_recumbent.mochi_prefab` | prefab | deformable (`soft`) actors — neoHookean material | adapter restriction | static validation | soft-body actor support in the audited scene profile |
| `superdex-physics/samples/articulations_soft_skinned_double_pendulum.mochi_scene` | scene | deformable (`softSkinned`) actors | adapter restriction | static validation | soft-skinned actor support |

A recipe can attach a bot containing an unsupported component. The blocker then
propagates to the containing hand or arm-hand assembly even when its top-level
file does not declare the component itself. Native bots currently accept
`SENSOR_CAMERA`; custom actuators and learned seed sensors remain unsupported.

### Restrictions that do not block every use of a model

- Floating-base OSC is unavailable because of the SDK's bot-space/actor-space
  effort-indexing mismatch; other supported control paths may still work.
- Near-massless spherical hands are sensitive to gravity/probe torques in the debug
  SDK. The comparison profile uses zero gravity and armature-scaled probes on both sides.
- Deep self-collision history can trigger a native contact-query assertion; the
  penetrating-contact check uses a fresh reference scene.
- Passive scenes have no selected action space but can still load and simulate.
- Model domain randomization, ROM/deformable/tactile state, GPU batched physics,
  and the base interface's standalone image-capture APIs are not provided by this
  profile. Playback support is described separately in section 6.

The ten rows are the maintained local-bundle baseline, not proof that every future
asset is supported. Fold newly discovered blockers into both language versions.
Historical URDF primitive-geometry loss in the SDK still needs independent
revalidation before planning a faithful importer.

<a id="troubleshooting"></a>
## 10. Troubleshoot common failures

| Symptom | Meaning and action |
| --- | --- |
| Missing model | Check the local bundle, working directory and `--assets` / environment setting |
| Mesh path repeats `benchmarks/cart_pole` | Model lookup and dependency loading used different roots; the viewer now propagates its selected root. For direct API use, export the correct `SUPERDEX_ASSETS_PATH` |
| Explicit finite effort limits required | Supply positive scalar/per-coordinate limits in actuator order |
| Explicit controlled joints required | Supply an ordered scene selection, or `[]` for passive loading |
| Batch incompatible with debugger / advanced profile | Use serial mode; advanced profiles and interactive viewing also need one environment |
| Controller configuration rejected | Check `get_controller_descriptions()`, model type, target type and native parameter dimensions |
| No camera images | Camera accessors provide metadata/poses, not pixels; use the supported playback path for visualization |
| Zero contact flags immediately after reset | Take a positive physics step; contacts describe the previous completed solve |
| Asset works in another loader but is blocked here | Check the unsupported fields/components; UniSim rejects content it cannot faithfully translate |

<a id="validation"></a>
## 11. Validate changes and track remaining work

```bash
export SUPERDEX_ASSETS_PATH="$PWD/assets/superdex"
uv run --no-sync scripts/superdex_compare.py --all --fixtures
UV_NO_SYNC=1 make check
make package
```


Use `uv lock --check` when dependency metadata changes. SDK-dependent tests require
the optional runtime; asset-dependent tests use `SUPERDEX_ASSETS_PATH`. Core
contract/import tests also run without native runtimes. Packaging must exclude
asset payloads and generated results. No validation command implies a release or commit.

| Check | What it establishes |
| --- | --- |
| Inventory verification | File/dependency integrity |
| `make check` | Lint and regression tests |
| Comparison tool | Direct SDK agreement for the exercised conditions |
| Viewer `--frames` | Automated native renderer smoke |
| Interactive human inspection | Visual assessment of geometry and behavior |
| `make package` | Distribution build succeeds |

The two runnable tools consolidate qualification/viewing workflows, not all unit
tests. Focused pytest coverage remains in `tests/`:

| Retained family | Comparison helper / regression location |
| --- | --- |
| FR3, bot routing/clipping, PD, contacts, reset, batch/serial | `check_bot`; bot/FR3 qualification tests |
| FREE roots and translated/rotated authored frames | `check_bot`; `test_superdex_native_floating.py` |
| Rigid scenes, multiple articulations, nested prefabs/controllers | `check_scene`; scene/rigid tests |
| Robot plus nested sphere/peg board, actual robot contact/isolation | `check_prefab_contact` via `--fixtures`; prefab tests |
| Cameras, controller substeps, forces, selective reset, link targets | `check_controller`; component/shared API tests |
| Archives, spherical joints, tendons, coupled actuation | Synthetic fixtures; rigid/shared API/inventory tests |
| Audited MJCF and lazy imports | Materialization/contract/import tests |

Remaining work: learned/custom components, soft/soft-skinned bodies, faithful URDF
loading, batch support for advanced trees, and human visual approval of qualified
scenes. UniLab owns training, task rollouts and sim2sim policy validation.
