# Pick up apple

Physical port of Dexlab's Apache-2.0 apple-stem-grasp demo at commit
`eab7e0114858fc96bfe54821590a0e3a8031e113`.

The right hand performs **stem pinch → lift → place → release → fruit grasp → lift**.
The apple is a free 0.2 kg rigid body held by contact forces. Recorded trajectories
are reference evidence only and are never used as actions or imposed object poses.

## Layout

- `superdex/`: physical runner, planning, verification, and source assets.
- `mujoco_src/`: MuJoCo exporter, runner, planner, and runtime helpers.
- `mujoco/`: ignored generated MJCF and meshes.
- `isaac/`: Isaac runner, qualification, planner bridge, and ignored USD layers.
- `evidence/`: reference trajectory and provenance.

Each engine's executable is named `run.py`; supporting files have role-based names.
Generated scenes retain their existing internal names and relative references.

## Run

Use the repository-local SuperDex asset bundle and Python 3.12. Install the optional
FP64/task dependencies into the existing UniSim environment:

```bash
uv pip install scipy==1.18.1 trimesh==5.1.0 rtree==1.4.1
uv pip install --no-deps -r benchmark/pick_up_apple/requirements.txt
uv run --no-sync python benchmark/pick_up_apple/superdex/run.py
uv run --no-sync python benchmark/pick_up_apple/superdex/run.py --viewer
```

The `--no-deps` flag preserves UniSim's patched `superdex-*-uni` facade and FP32
batch runtime while adding the official FP64 payloads. Use `--no-sync` when running
the task so project synchronization does not remove these optional dependencies.

The 40-second task executes 20,000 physics steps at 2 ms. Setup first builds the
0.2 mm apple SDF, settles the apple for three seconds without the robot, and solves
inverse kinematics in an isolated planning scene. SDF construction can take several
minutes and substantial memory. The interactive viewer renders the running task
at 20 Hz; it does not start an uncontrolled simulation after the task has finished.

Optional diagnostics:

```bash
uv run --no-sync python benchmark/pick_up_apple/superdex/run.py --seconds 0.05 --offscreen
uv run --no-sync python benchmark/pick_up_apple/superdex/run.py --output results/pick_up_apple
uv run --no-sync python benchmark/pick_up_apple/superdex/verify.py results/pick_up_apple
```

`--seconds` requests a shorter smoke run, rounded to the nearest physics step.
A short run is explicitly incomplete and cannot report a successful grasp sequence.
Closing the viewer early also leaves the sequence incomplete. `--out FILE` writes
just the summary; `--output DIR` additionally writes forces, metrics and a trajectory.
Use ignored `results/` paths for generated evidence.

## MuJoCo scene-only transform

Generate a controller-free MuJoCo MJCF from the same SuperDex robot and rigid
object fragment:

```bash
SUPERDEX_PRECISION=fp64 uv run --no-sync python benchmark/pick_up_apple/mujoco_src/export.py
```

The exporter writes ignored `benchmark/pick_up_apple/mujoco/` artifacts, exports
collision meshes, preserves the source root transform, link/joint frames and
inertials, converts disabled contact overrides to `<contact><exclude/>`, and adds
the free apple plus static table. The table uses the source `BOX` collider and the
combined apple/stem mesh uses MuJoCo's native mesh SDF. Before physics starts,
`mujoco_src/runtime.py` chooses an octree depth whose finest cells are no wider than
SuperDex's 0.2 mm target and rejects a compiled model that fails that check. For
the current asset this is depth 10 and approximately 0.115 mm maximum cell width;
compilation takes about 37 seconds and peaks near 8 GB RSS on the reference machine.
The scene intentionally defines **no actuators**. Matching spatial discretization
does not make the two engines' contact solvers or resulting forces identical.

## Isaac scene authoring and qualification

The saved USD is an authored initial definition, not a post-run snapshot:

```bash
OMNI_KIT_ACCEPT_EULA=1 \
VIRTUAL_ENV="$HOME/.unilab/isaacsim/venv" \
LD_LIBRARY_PATH="$HOME/.unilab/isaacsim/compat/libxml2" \
"$HOME/.unilab/isaacsim/venv/bin/python" \
  benchmark/pick_up_apple/isaac/qualify.py --author
```

`--author` writes the USD before stepping and never saves a live pose or velocity.
The saved apple starts at the SuperDex/MuJoCo authored pose
`(0.3, 0.4, 0.34783494)` with identity rotation and zero velocity; all authored
robot velocities are zero. The 12 collisionless virtual frames are deactivated,
and their two arm-base joints are rewired to the physical root, preserving the
79 positive-mass source links without the former 1.2 g dummy mass. The table is
static, all 140 explicit source contact exclusions remain, and the apple uses
PhysX SDF resolution 524 (spacing at or below 0.2 mm). A read-only run of the
same script rechecks those values and executes the 2-second static-hold/apple
qualification.

Full physics identity is **not** claimed. Source selective robot self-collision
is still disabled: enabling it on this imported articulation drives all robot
joint and base states to NaN by step 18 even with the required aggregate-pair
capacity raised. Contact materials and the SuperDex/MuJoCo contact solvers are
also not equivalent. Those limitations are reported explicitly rather than hidden
by another dummy mass or a rewritten saved state.

## Isaac controller-parity replay

Run the authored USD with the same online-planned joint targets and PD schedule as
SuperDex/MuJoCo. Recorded DexLab body poses are not control inputs:

```bash
VIRTUAL_ENV="$HOME/.unilab/isaacsim/venv" \
LD_LIBRARY_PATH="$HOME/.unilab/isaacsim/compat/libxml2" \
OMNI_KIT_ACCEPT_EULA=1 \
"$HOME/.unilab/isaacsim/venv/bin/python" \
  benchmark/pick_up_apple/isaac/run.py --seconds 40 --viewer --viewer-rate 5 --unpaced
```

`--viewer` starts Isaac with a GUI and renders it independently from the 2 ms
physics loop. `--unpaced` skips wall-clock sleeping while preserving the 2 ms
physics and control steps; fewer frames may be visible when physics runs quickly.
Without `--unpaced`, the viewer follows simulated time. Closing the window stops the run cleanly
and records `viewer_closed_early`; it never changes targets, gains, timestep, or
the replan. `isaac/targets.py` runs in the repository environment and invokes the shared
MuJoCo FK planner in a subprocess. `isaac/run.py` remaps those targets into
Isaac's DOF order, applies one 54-joint position target every 2 ms, verifies both
gain readbacks, and replans at 23 s from Isaac's actual apple pose. Live Isaac
gain APIs use the source per-radian values; only authored USD angular drive
attributes use per-degree values. Runtime drive effort limits are removed to
match the unsaturated SuperDex and MuJoCo controllers. The viewer calls
`world.render()` at its display rate to synchronize physics transforms.
The summary checks the same 8–9 s lift and three hold/release clearance
windows as MuJoCo, so a complete run fails when the apple is not lifted or
released as planned. These geometric checks do not establish force/contact
parity: Isaac still has selective robot self-collision disabled and a different
contact solver. `--out FILE` saves the JSON report, and the command returns a
nonzero status when any applicable check fails.

## MuJoCo physical replay

After generating the scene-only MJCF above, replay the same online-planned task
through the UniSim MuJoCo adapter:

```bash
uv run --no-sync python benchmark/pick_up_apple/mujoco_src/run.py
uv run --no-sync python benchmark/pick_up_apple/mujoco_src/run.py --viewer
uv run --no-sync python benchmark/pick_up_apple/mujoco_src/run.py --viewer-visual
uv run --no-sync python benchmark/pick_up_apple/mujoco_src/run.py --seconds 1 \
  --output results/pick_up_apple_mujoco_smoke
```

`--viewer` renders a native copy of the adapter's compiled physics model, including
the same SDF octree and the apple collider colored translucent green at compile time.
It omits visual-only geoms; press `C` in the MuJoCo viewer to show contact points. `--viewer-visual` uses a
separately compiled display model with the original visual meshes, so it is not
an inspection of the physics SDF grid.

The MuJoCo port generates an ignored runtime MJCF with one affine position servo
per robot hinge, settles the apple in an isolated adapter-owned table/apple scene,
performs the same arm/pinch/lift/regrasp planning using MuJoCo FK, and sends one
position target per 2 ms physics step. Servo gains match the unchanged SuperDex
controller, including the 23-second right-hand gain change. Actuator limits and
duplicated passive damping are disabled. The runner disables automatic bad-state
resets and records controller, integrator, gain-profile, monotonic-time, tracking,
phase-clearance, and 20 Hz transform diagnostics.

Before planning, MuJoCo settles for at least 3 seconds and requires a full
0.5-second window whose poses stay within 0.25 mm and 0.25 degrees of the window's
first pose. These tolerances allow the measured small mesh-contact jitter; they
do not imply an exactly motionless apple. Settling fails after 10 seconds if the
gate is not met. The final physical pose and velocities are preserved, and the
summary records settling duration and measured excursions. SuperDex is unchanged.

The requested MuJoCo `discrete` integrator was introduced in MuJoCo 3.13 and is
selected when the installed binding exposes it. This repository is currently pinned
to MuJoCo 3.11 because `mujoco-uni-runtime==0.5.0` requires MuJoCo `<3.12`, so the
runner prints and records an explicit `implicit` fallback. Under the requested 2 ms,
unbounded-gain settings this fallback currently reports `BADQACC`; the runner treats
that as a hard failure instead of silently changing timestep, gains, or limits.
The default viewer copies the adapter's compiled model without recompiling its SDF
and mirrors the adapter state at 20 Hz. Visual-only geoms are omitted by `discardvisual=true`; the apple's SDF
collision geom keeps its compile-time green color. `--viewer-visual` loads a separate
unstripped display model. Both viewers focus on the table and pace playback in
real time. Closing either window ends the run cleanly and marks the requested
steps incomplete; Ctrl-C closes the viewer/backend without a traceback.
Exact controller/contact parity and the SuperDex force-balance verifier are not
claimed: SuperDex uses an internal pose constraint, while the MuJoCo path uses a
native-servo emulation. Both paths now use the same combined apple/stem surface and
an SDF target spacing of at most 0.2 mm, but their contact solvers remain different.

## Implementation and validation

- Dexlab's base placement and trimmed torso; unchanged repository-local arm/hand assets.
- FP64, GMRES, 128 nonlinear iterations, Dexlab's contact settings and 0.2 mm SDF.
- Online known-pose IK, gravity compensation, and joint targets, including the gain
  change and fresh planning from the actual released apple at 23 seconds.
- `SimBackend.set_state()` initializes the robot and settled apple once. The
  unchanged SuperDex path uses `step_controller()` with its native pose controller;
  only the MuJoCo path uses `step()` with position targets for its actuator emulation.
- `superdex/native.py` is an explicit SuperDex-specific helper for solver/contact setup,
  isolated preparation, and native force/contact instrumentation. These features
  are not advertised as portable shared-API capabilities. It accesses the adapter's
  native scene handles; it does not drive or teleport the live task scene.
- The adapted Dexlab verifier checks both suspended holds, table-supported release,
  thumb/index stem contact, fruit contact, penetration, momentum/force balance,
  solver divergence, fixed base, and finite state. A full failed sequence exits nonzero.

This is a SuperDex workload, not a cross-engine timing result. It retains the
reference's known-object-pose assumption and rigid stem (no bending/fracture).
Native Polyscope renders collision/robot assets; Dexlab's textured MuJoCo presentation
is not part of this port.

## Provenance

Dexlab code and its modifications are attributed in `superdex/assets/NOTICE-Dexlab` and
licensed by `superdex/assets/LICENSE-Dexlab`. `evidence/port-provenance.json` records the
source commit and file hashes. Original historical evidence is retained separately.

Robot and hand payloads live in the ignored repository-local SuperDex bundle.
The task includes Dexlab's trimmed torso and retains OpenArm/Wuji notices. Apple
collision meshes derive from NVIDIA Isaac Sim 5.1; original URLs/hashes are in
`superdex/assets/objects/sources.json`. Third-party payloads retain their original terms and
are not relicensed by the code license.
