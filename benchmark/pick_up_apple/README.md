# Pick up apple

Physical port of Dexlab's Apache-2.0 apple-stem-grasp demo at commit
`eab7e0114858fc96bfe54821590a0e3a8031e113`.

The right hand performs **stem pinch → lift → place → release → fruit grasp → lift**.
The apple is a free 0.2 kg rigid body held by contact forces. Recorded trajectories
are reference evidence only and are never used as actions or imposed object poses.

## Run

Use the repository-local SuperDex asset bundle and Python 3.12. Install the optional
FP64/task dependencies into the existing UniSim environment:

```bash
uv pip install scipy==1.18.1 trimesh==5.1.0 rtree==1.4.1
uv pip install --no-deps -r benchmark/pick_up_apple/requirements.txt
uv run --no-sync python benchmark/pick_up_apple/run.py
uv run --no-sync python benchmark/pick_up_apple/run.py --viewer
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
uv run --no-sync python benchmark/pick_up_apple/run.py --seconds 0.05 --offscreen
uv run --no-sync python benchmark/pick_up_apple/run.py --output results/pick_up_apple
uv run --no-sync python benchmark/pick_up_apple/verify.py results/pick_up_apple
```

`--seconds` requests a shorter smoke run, rounded to the nearest physics step.
A short run is explicitly incomplete and cannot report a successful grasp sequence.
Closing the viewer early also leaves the sequence incomplete. `--out FILE` writes
just the summary; `--output DIR` additionally writes forces, metrics and a trajectory.
Use ignored `results/` paths for generated evidence.

## Implementation and validation

- Dexlab's base placement and trimmed torso; unchanged repository-local arm/hand assets.
- FP64, GMRES, 128 nonlinear iterations, Dexlab's contact settings and 0.2 mm SDF.
- Online known-pose IK, gravity compensation, and joint targets, including the gain
  change and fresh planning from the actual released apple at 23 seconds.
- `SimBackend.set_state()` initializes the robot and settled apple once; every task
  physics step uses `SimBackend.step_controller()` and `ArticulationPoseTarget`.
- `native.py` is an explicit SuperDex-specific helper for solver/contact setup,
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

Dexlab code and its modifications are attributed in `assets/NOTICE-Dexlab` and
licensed by `assets/LICENSE-Dexlab`. `evidence/port-provenance.json` records the
source commit and file hashes. Original historical evidence is retained separately.

Robot and hand payloads live in the ignored repository-local SuperDex bundle.
The task includes Dexlab's trimmed torso and retains OpenArm/Wuji notices. Apple
collision meshes derive from NVIDIA Isaac Sim 5.1; original URLs/hashes are in
`assets/objects/sources.json`. Third-party payloads retain their original terms and
are not relicensed by the code license.
