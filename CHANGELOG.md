# Changelog

## Unreleased

- Expose `superdex_num_worker_threads` for serial SuperDex backends, forwarding
  `-1`, `0`, or a positive worker count to the process-wide SDK runtime while
  rejecting nested batch/runtime thread pools and conflicting live configurations.
- Pass the viewer's selected asset root to dependency loading and comparison
  workers, so default-root and `--assets` scenes work without a prior
  `SUPERDEX_ASSETS_PATH` export.
- Consolidate the SuperDex backend from 17 modules to 8 (`model`, `runtime`,
  `materialization`, `scenes`, `components`, `backend`, `assets`,
  `__init__`), preserving public exports, shared API signatures, lazy SDK
  loading and numerical behavior. Fold root frames, joint layouts,
  articulations and metadata into `model`; dependency loading, runtime
  ownership and CPU topology into `runtime`; MJCF geometry into
  `materialization`; prefab audit/composition into `scenes`; controller APIs
  into `components`.
- Replace the twelve per-stage qualification/viewer scripts with two tools:
  `scripts/superdex_compare.py` (numerical adapter-vs-SDK comparison over
  every native model in isolated subprocesses, with format discovery,
  limitation classification and opt-in JSON reports into the ignored
  `results/` directory) and `scripts/superdex_viewer.py` (interactive viewing
  for all four native formats plus MJCF, passive fallback, controller
  configuration and synchronized `--compare` dual windows). Preserve the
  historical tolerances, rollout lengths and check families; the synthetic
  fixture generator is ported into the comparison tool.
- Remove the generated qualification reports, compatibility tables,
  screenshots and logs under `docs/superdex-*`; keep the JSON integrity
  inventory and move the authored controller config/target examples to
  `docs/superdex-configs/`. `docs/superdex.md` is now the single
  authoritative guide, including the maintained "Current limitations and
  unsupported assets" report covering the ten known blocked models (URDF
  input, custom actuator/sensor components, seed sensors — including recipe
  inheritance — and deformable actors).
- Include packed bot archives and their dependencies in the integrity inventory;
  stop generating the redundant Markdown inventory.
- Restore full multi-articulation state checks, per-substep SDK controller comparison,
  camera parity, transformed-root probes, and robot–prefab contact regressions.
  Viewer comparison propagates options and failures across both windows; registered
  bots use PD effort control and unknown models remain passive.
- Inventory classification now propagates custom-component blockers through
  recipe dependency closures (seed-sensor hands record `unsupported-features`
  instead of `recipe-candidate`), accepts both `type` and `typeName` component
  spellings, and prefers feature blockers over the recipe disposition.
- Rewire the SuperDex pytest suites onto the consolidated tools (fresh
  temporary-output checks replace historical-report tests); add coverage for
  format discovery, archive handling, limitation classification and
  subprocess failure isolation.

- Fix rigid-viewer format routing: pass controlled-joint selection only for scenes
  and prefabs, and explain that `.mochi.h5` shape files are not standalone models.

- Align SuperDex with the shared `SimBackend` interface: optional model/controller
  metadata, camera/controller methods and gravity override, plus idempotent close.
  Export SDK-independent controller targets; retain native inputs with deprecation
  warnings. Resolve scalar effort limits inside the adapter and migrate viewers
  to public APIs. Include bot-level placement in floating world-state conversion
  (notably Oculus hands); preserve native physics and existing batch profiles.

- Complete Stage 8 native rigid-body adaptation: closed loops, spherical DOF
  metadata, bot archives, external roots, standalone/nested prefabs, object-only
  scenes and multiple articulations. Preserve rigid constraints, transmissions,
  tendons, tracking controllers and articulated skin. Add per-actor state/force
  ownership, passive torso qualification, direct-SDK reports and native viewer
  smoke. New profiles target one serial environment; preserve existing batch
  regressions and document deferred features and SDK blockers.

- Add Stage 7 native `.mochi_scene` dispatch for one fixed/hinge/slide articulation
  plus rigid objects and nested prefabs. Preserve authored scene settings,
  contact filters and joint-tracking rest springs; keep the timestep caller-owned.
  Require explicit `superdex_controlled_joints` and physical effort limits.
  Add complete state/reset and partial-load cleanup coverage, Cart Pole and
  Half Cheetah direct-SDK qualification, native viewer smoke and updated guidance.
  Multiple articulations, soft bodies and scene camera/plugin components remain
  unsupported. Asset bytes and existing bot/MJCF control behavior are preserved.

- Add `--inventory-only` to refresh the local SuperDex asset inventory without
  consulting the source checkout, while validating all asset dependencies.

- Add SuperDex-specific built-in camera metadata/pose access and explicit JSC PD,
  OSC PD and articulated-pose controller APIs, with per-environment ownership,
  substep evaluation and reset/cleanup. Report the SDK's floating OSC indexing
  limitation explicitly. Preserve ordinary torque and MJCF control behavior.
- Keep universal bot/scene qualification with direct SDK references, explicit
  bot selection, camera checks and optional controller configuration.

- Add stage-5 native SuperDex rigid prefab composition through
  `SceneCfg.fragment_files`, including nested transforms, complete robot/object
  qpos/qvel state, per-object floating-root layouts, body queries and forces,
  whole-scene/selective reset, and scene-owned cleanup. Reject unsupported
  prefab components before SDK loading. Add synthetic regression fixtures,
  FR3/sphere/peg-board direct-SDK qualification and a native viewer example.

- Complete the native floating-model audit across all 22 FREE-root bots: ten
  hands/grippers pass direct-SDK state/lifecycle qualification in serial and
  batch modes; twelve retain later-stage component/joint/cycle blockers.
  Add `--all-floating`, per-model reports, a ten-model pytest set and native
  viewer demonstrations for every qualified floating model.

- Add native SuperDex FREE roots with authored reference transforms, canonical
  world-position/wxyz-orientation state, body-frame angular velocity, and
  corrected joint/control offsets. Add SDK-free metadata/rejection tests,
  direct-SDK transform/Jacobian qualification, translated/rotated reference
  regression fixtures, and floating viewer profiles with full-state reset.
  Close native viewers on scene-binding, camera and initialization failures.

- Add the stage-3 compatible-bots qualification:
  `uv run scripts/superdex_bot_qualify.py` runs the stage-2B check set for
  eleven registered candidates — the fr3/fr3_v2 arms, both openarm_v20 arms,
  googly_eyes, fr3_v2_with_eyes and the recipe compositions openarm_v20,
  openarm_v20_wuji, fr3_dg5f_short (left/right) and fr3_v2_allegro_v5_right —
  each against a direct SuperDex SDK scene driven with identical inputs, with
  a recipe-accounting check that every base/attachment reference resolves
  inside the verified bundle. Every adapter-vs-SDK deviation was exactly 0.0;
  no adapter extension was needed because the SDK compiles recipes into the
  native loader's supported profile. Two checks were generalized for light
  distal links: control verifies the adapter's per-joint response matrix
  matches the SDK matrix (FR3's single-joint dominance does not hold for the
  openarm wrist), and the 10 rad/s sweep-velocity guard now applies only to
  the pure fr3 arms. Reports and the compatibility table land in
  `docs/superdex-bots-qualification/`, including precise blockers for every
  remaining bot (floating roots, sensor components, mechanical cycles,
  spherical joints, the 0-DoF torso's batch rejection). The stage-2A viewer
  gained `--bot <key>` with shared per-asset control profiles
  (`scripts/superdex_bot_profiles.py`); the pytest subset is
  `tests/test_superdex_bot_qualification.py` (opt-in via
  `SUPERDEX_ASSETS_PATH`). See `docs/superdex.md`.
- Add the stage-2B FR3 adapter qualification:
  `uv run scripts/superdex_fr3_qualify.py` loads the unchanged
  `bots/arms/fr3_v2/fr3_v2.superdex_bot` through the SuperDex adapter and runs
  seven numerical/lifecycle checks — structure, control ordering and effort
  clipping, batch/serial trajectory equivalence (1,000+ steps, also the
  stability evidence), contact recovery from a pose beyond an authored joint
  range, exact whole/selective reset and state round trips, two-environment
  isolation, and repeated create/close cycles — each compared against a direct
  SuperDex SDK scene driven with identical inputs in the same process. On the
  qualification host every adapter-vs-SDK deviation was exactly 0.0 and no
  adapter defect was demonstrated, so no adapter fix was required. The run
  report lands in `docs/superdex-fr3-qualification/`; the pytest regression
  subset is `tests/test_superdex_fr3_qualification.py` (opt-in via
  `SUPERDEX_ASSETS_PATH`). See `docs/superdex.md`.
- Add the stage-2A FR3 viewer demonstration:
  `uv run scripts/superdex_bot_viewer.py` opens the unchanged
  `bots/arms/fr3_v2/fr3_v2.superdex_bot` from the local asset copy through the
  SuperDex adapter's native `run_playback` interactive path (serial mode, one
  environment) and runs three observable phases — authored initial-pose hold,
  bounded joint-space sine movement inside the authored ranges, and a visible
  `reset()` restore verified numerically. Commands are joint-position targets
  in radians (`fr3_joint1..7`); a pre-step PD converter clips to explicit
  87/87/87/87/12/12/12 N·m effort limits (demonstration profile, not hardware
  ratings). Window close, Ctrl-C and error paths release the viewer and
  backend; the run report is written to
  `docs/superdex-fr3-viewer/`. See `docs/superdex.md`.
- Add `verify_asset_bundle()` to `unisim.backend.superdex.assets`: it
  re-hashes a local asset tree against the recorded inventory JSON and
  re-resolves every dependency edge, so qualification runs fail closed on
  missing, modified or escaped assets without the source checkout.
  `scripts/copy_superdex_assets.py --report-only` rebuilds the inventory and
  verifies without copying; the recorded digest was refreshed for the current
  tree (asset bytes unchanged).
- Add the verified local SuperDex asset copy under `assets/superdex` (891
  files, ~248 MiB) with SDK-free inventory tooling:
  `unisim.backend.superdex.assets` (module stays outside the public import
  boundary) and `scripts/copy_superdex_assets.py`, which records source
  provenance (revision `6b0541b`, 5 local-derivative files), verifies
  source/destination SHA-256 equality and dependency resolution, and writes
  `docs/superdex-assets-inventory.{md,json}`. Every entry carries a
  disposition and required-capability record; unsupported entries are
  preserved with precise blockers. The asset tree is excluded from
  `unisim-core` distributions.

## 1.2.0 - 2026-09-10

- Promote the current contract and adapter surface to the `1.2.x` line. No
  functional changes since 1.1.6; the public import boundary
  (`SimBackend`, `create_backend`, `ADAPTER_SPECS`, adapter classes, and
  `unisim.backend.subprocess_ipc`) is unchanged.

## 1.1.6 - 2026-09-10

- **Fix:** SuperDex serial-mode native interactive playback now frames the
  scene (`viewer.frame_scene()`) immediately after `set_scene`, before the
  first `frame_tick`. Polyscope's camera view matrix is uninitialized (NaN)
  until the first explicit camera placement, and the viewer's navigation
  gizmo reads it while building the first ImGui frame, so on-screen
  interactive playback crashed on the first frame with `ValueError: cannot
  convert float NaN to integer` (UniLab `eval --sim superdex --render-mode
  interactive`).

## 1.1.5 - 2026-09-10

- **Breaking (snapshot layout):** `mjwarp` `get_physics_state` snapshots now
  append `[mocap_pos(nmocap*3), mocap_quat(nmocap*4)]` after
  `[time, qpos, qvel]` when the model has mocap bodies, and
  `run_playback_mode` declares the extended `snapshot_shape`. The offline
  render workers (`render_many`) replay the recorded mocap pose instead of
  resetting mocap bodies to the model defaults, fixing record-mode videos
  where mocap-driven geometry (e.g. the Wuji mocap palm, whose wrist pitch is
  randomized at reset) rendered misaligned with — and interpenetrating — the
  free-joint objects. Legacy `[time, qpos, qvel]` snapshots keep the previous
  defaults-plus-grid-offset fallback. `validate_offline_visual_model` now also
  requires `nmocap` parity between the physics and visual models.

- **Fix:** `ghost_geom` debug overlays now inherit the material (with mesh UV
  texturing) of the model geom that renders the same mesh, in both the
  offline render workers and the mjwarp interactive viewer, matching the
  source task's textured goal indicator. `append_debug_primitives` gains an
  optional `mesh_materials` mapping; assets without a textured model geom
  keep the flat primitive rgba.

- **Fix:** multi-env grid recording without an explicit `cam_lookat` widens
  `cam_distance` so every grid cell fits the frame (`render_many`
  `_grid_fit_distance`, from the grid span, fovy, and frame aspect ratio).
  An explicit `cam_lookat` still pins the camera to a single env.
- Add a SuperDex `execution_mode` option (`superdex_execution_mode` factory
  kwarg, `"batch"` default or `"serial"`). Serial mode never constructs the
  `SceneBatchExecutor` and steps every scene on the environment thread so the
  native SuperDex debugger can attach without violating the scene's
  thread-affine `DebugDraw`. Batch mode now fails closed with an actionable
  `RuntimeError` naming the serial mode when a debugger client is connected at
  construction or attaches before a later step (unilabsim/unisim#55). Serial
  mode also enables native interactive playback: `run_playback` in the
  `interactive` render mode drives the upstream Polyscope viewer on the single
  environment scene, failing closed unless the backend is serial with
  `num_envs=1`.

- Switch the SuperDex adapter's optional runtime to the temporary unilabsim
  `superdex-physics-uni` / `superdex-robotics-uni` 1.0.0 wheels, which carry
  the native batch executor ahead of the upstream project_superdex release,
  and extend the supported interpreter range to CPython 3.12 and 3.13. The
  adapter still rejects other Python versions with a targeted diagnostic.
  Switch the distribution names back to upstream once the upstream PR merges.
  Map unlimited actuator force ranges to the dtype's finite bounds so the
  native `step_control` validation accepts MJCF motors without a `forcerange`.

- **Fix:** multi-env grid rendering in `render_many.render_frame_job` now
  translates mocap bodies with the environment. Worker `MjData` is reused
  across frames, so `init_worker` caches cold-path `mocap_pos` defaults and
  `set_state` resets from them before adding the grid offset; mocap bodies
  are excluded from the legacy `geom_xpos`/`site_xpos` post-shift so the two
  mechanisms cannot double the offset. Previously, models whose first body
  has a free joint (e.g. the Wuji in-hand scene: free-joint cube plus mocap
  palm) rendered every env's mocap-driven geometry stacked at env 0,
  misaligned with both the free-joint objects and the debug overlay
  primitives, which already receive the offset exactly once
  (unilabsim/wuji_unilab#21).

- **Breaking:** replace `run_playback(..., extra_data_getter=...)` with
  `debug_overlay_getter`. The new callback returns per-frame, per-env
  sequences of typed `DebugPrimitive` values (`sphere`, `box`, `frame`,
  `arrow`, `ghost_geom`, `text`) with env-local poses instead of a single
  `(num_envs, 3)` marker-position array; grid offsets are applied by the
  renderer. `BackendPlayCapabilities` gains `supports_debug_overlay`; the
  MuJoCo-family offline snapshot pipeline (mujoco, mjwarp, drake, newton,
  superdex) advertises it, while other backends fail closed with
  `NotImplementedError` when a getter is supplied. `ghost_geom` primitives
  resolve `mesh_asset` against playback-model mesh names or mesh asset files
  injected into the render model; `text` primitives are a documented no-op on
  the MuJoCo off-screen path. Newton record playback with overlays routes to
  the offline MuJoCo snapshot renderer (the native ViewerGL path cannot
  inject user geoms).
- mjwarp interactive playback now consumes `debug_overlay_getter`: each frame
  injects the tracked world's primitives into the passive viewer's
  `user_scn` before `sync()`, resolving `ghost_geom` meshes against the
  playback model (fail-closed when unregistered). `BackendPlayCapabilities`
  gains `supports_interactive_debug_overlay` (default False; mjwarp reports
  True) so callers can tell whether the interactive path consumes the getter.
  The interactive `on_frame` hook remains fail-closed
  (unilabsim/wuji_unilab#21).
- **Breaking:** `camera_kwargs` is normalized into the typed frozen
  `CameraCfg` at the `run_playback`/`init_renderer` boundary. Unknown keys —
  including the historical `distance`/`elevation_deg`/`azimuth_deg` aliases —
  now raise an error naming them instead of being silently ignored; the
  optional `cam_fov` key is supported by the MuJoCo offline renderer and
  Genesis. `DebugPrimitive`, `DebugOverlayGetter`, `CameraCfg`, and
  `validate_debug_overlays` are exported from `unisim` and
  `unisim.contract` (unilabsim/wuji_unilab#21).

- Add `unisim.visualization.render_many.append_debug_primitives`, the public
  primitive-injection entry shared by the offline render workers and
  interactive viewers (single-env `viewer.user_scn` callers pass
  `overlays=[primitives]`, `offsets=None`); it returns the injected geom
  count. Interactive `ghost_geom` meshes must already be registered in the
  loaded model (resolved via `mesh_ids`), failing closed otherwise — the
  interactive path cannot recompile the model (unilabsim/wuji_unilab#21).

- Add `run_playback(..., on_frame=...)`: the offline MuJoCo pipeline calls
  `on_frame(frame_index, frame)` with each `(H, W, 3)` uint8 frame before
  video encoding; returning a replacement array (same shape/dtype, validated
  fail-closed) substitutes it and `None` keeps the original. Backends on
  native renderers (motrix, genesis, subprocess IPC; newton/mjwarp
  interactive paths) fail closed with `NotImplementedError`; newton record
  playback with `on_frame` routes to the offline snapshot renderer
  (unilabsim/wuji_unilab#21).


- Add a local-source SuperDex `SceneBatchExecutor` integration: a persistent
  C++ CPU barrier batches independent-scene generalized force writes, stepping,
  and articulated state reads. `superdex_num_workers=0` resolves an
  affinity-aware outer worker count while SDK-internal and outer workers remain
  mutually exclusive.

- Batch SuperDex body and sensor frame transforms over selected environments,
  removing repeated small-array work while preserving native stepping order,
  controls, sensor precision and reset isolation.

- Add the optional SuperDex 1.0.0 CPU adapter with native fixed-base bot and
  audited MJCF articulation materialization, NumPy state/control translation,
  independent scene resets, named state/contact sensors and process-owned
  cleanup. Python 3.12 is required by the upstream wheels. See
  `docs/superdex.md` for the experimental contact profile and explicit limits.

## 1.1.4 - 2026-09-08

- Add cold-bound selected-world mocap pose reads/writes and reset ordering to
  `SimBackend`, with an explicit unsupported default and a MJWarp implementation.
- Add MJWarp reset randomization for primitive geometry size (including derived
  bounds), contact solref/solimp, joint damping and joint friction loss. New
  payload tables validate before mutation, preserve unselected worlds, and
  expose cold-path defaults through the public backend contract. See
  [the owner contract](docs/mocap-reset-contract.md) and issue #40.
- Preserve position actuator gain signs in MJWarp domain randomization.

## 1.1.3 - 2026-09-06

- Ensure Newton ViewerGL playback shows authored static planes and supplies a
  default visual-only floor when a scene does not define one.

## 1.1.2 - 2026-09-06

- Merge Newton's native ViewerGL dependencies (`pyglet` and `imgui-bundle`)
  into the single `newton` extra. Newton playback now uses its native renderer
  by default after `uv sync --extra newton`; the separate `newton-render`
  extra is removed.

## 1.1.1 - 2026-09-06

- Align all MuJoCo-related extras on the 3.11 line (unilabsim/UniLab#1515,
  unilabsim/unisim#34): the `mujoco` extra now requires `mujoco~=3.11.0` with
  `mujoco-uni-runtime==0.5.0` (exact pin — one runtime release carries one
  prebuilt MuJoCo binding, so a lock bump is gated on wheel availability); the
  `mjwarp` extra moves from `mujoco-warp==3.10.0.3` to `mujoco-warp~=3.11.0`
  with `warp-lang==1.16.0`; and the `newton` extra keeps its exact pins.
  The extras are now jointly resolvable, so the `[tool.uv]` extra conflicts
  are removed and the MJWarp runtime check accepts the whole
  `mujoco-warp` 3.11 line instead of one exact version.
- Add snapshot playback support to the Newton adapter: `get_physics_state` /
  `set_physics_state` ([time, qpos, qvel] host-cache layout), record/none
  `resolve_play_render_plan` semantics, and `run_playback` through the shared
  offline MuJoCo snapshot renderer now in `unisim.backend.playback_common`
  (mjwarp behavior and messages unchanged). Add a fail-closed
  `SimBackend.set_physics_state` default.
- Add MJCF contact-sensor (`mjSENS_CONTACT`) support to the Newton adapter for
  the exact `data="found" num=1` named-geom-pair shape: per-env binary flags
  are resolved once against `SolverMuJoCo.mjc_geom_to_newton_shape` at
  materialization and refreshed through `SolverMuJoCo.update_contacts`; all
  other contact-sensor configurations remain fail-closed.
- Rework the README around the project overview, UniLab relationship,
  installation, and quick start, and add the Chinese `README_zh.md`.
- Add the Newton 1.5.1 runtime extra and a metadata/import probe on the
  MuJoCo-Warp 3.11 line (the 3.11 alignment above later lifted the initial
  mutual exclusion with the `mjwarp` extra).
- Add fail-closed Newton nconmax/njmax capacity sampling and overflow
  diagnostics for the forthcoming adapter.
- Add the Newton `SimBackend` adapter with explicit CUDA placement, cold-path
  MJCF materialization/audits, host NumPy state caches, and fail-closed sensor
  and geometry coverage.
- Fixed Genesis device selection on multi-GPU hosts: the engine only honors
  the first entry of `CUDA_VISIBLE_DEVICES`, so the adapter now pins
  `CUDA_VISIBLE_DEVICES` to the requested physical device before any CUDA
  query (including `torch.cuda.is_available()`, which itself latches the
  visible-device set) and remaps the process-local device index to `cuda:0`.
  Out-of-range requests fail closed with a clear error.

## 1.1.0 - 2026-09-05

- Added the declarative interval domain-randomization term contract in
  `unisim.dr.interval`: builtin term specs (`INTERVAL_TERM_SPECS`,
  `interval_term_spec`), the pickle-safe `IntervalTermOp` descriptor with
  builtin-contract validation, and the `ops` field on
  `IntervalRandomizationPlan` (`iter_ops()` translates the legacy fields).
- Added the `supported_interval_terms` capability set on
  `DomainRandomizationCapabilities` with `supports_interval_term()` /
  `get_unsupported_interval_terms()`, falling back to the legacy bools so old
  constructor call sites keep their meaning.
- Replaced the abstract per-backend `apply_interval_randomization`
  implementations with generic `SimBackend` dispatch over the backend-owned
  `_interval_term_handlers()` table; terms without a handler fail closed with
  `NotImplementedError` naming the backend class and the term.
- Deprecated the five legacy `IntervalRandomizationPlan` fields and the five
  `supports_interval_*` capability bools; they remain functional and will be
  removed in the next major release.
- Fixed the mjwarp and genesis backends silently dropping unsupported
  interval body-torque and body-angular-velocity randomization; both now fail
  closed through the base dispatch.

## 1.0.0 - 2026-09-04

- Promote the contract and seven-adapter manifest to the stable `1.0.x` line;
  the public import boundary (`SimBackend`, `create_backend`, `ADAPTER_SPECS`,
  adapter classes, and `unisim.backend.subprocess_ipc`) is now stable.
- Support `SimBackend.set_pre_step_control` on the `mjwarp` backend: a
  registered converter now runs on the host before every physics substep with
  the qpos/qvel cache refreshed to the substep-start state (matching the
  MuJoCo backend's substep boundary and `callback_sensordata=False` sensor
  semantics), and `None` unregisters it.  The callback path uses eager kernel
  launches instead of captured step graphs.
- Restore the missing 0.1.10 changelog entry and the 0.1.4/0.1.5 ordering, and
  correct the `unisim.backend.subprocess_ipc` path and Isaac extras spelling in
  the migration and support-matrix documentation.

## 0.1.14 - 2026-09-02

- Update the trusted-publishing action to support the source distribution's
  current Python Core Metadata version.
- Require successful cross-platform tests and pre-release sdist verification
  before a version tag can publish to PyPI.

## 0.1.13 - 2026-09-02

- Add GitHub Actions CI and tag-triggered PyPI trusted publishing.
- Publish only the source distribution so releases do not select a Python
  version, operating system, or wheel platform.
- Document repository development, compatibility, and release conventions.

## 0.1.12 - 2026-09-02

- Fix the root public export surface so wildcard imports resolve
  `MjcfSubprocessBackend` and its historical `SubprocessBackend` alias.
- Use package-owned `UNISIM_*` worker/cache environment variables and
  `~/.cache/unisim` defaults, with read-only fallback to legacy `UNILAB_*`
  overrides.
- Expand adapter/factory/import-boundary tests for the complete seven-backend
  manifest and package isolation.

## 0.1.11

- Corrected the Drake adapter to consume the external `drake-uni` distribution
  through its `drake_uni` import namespace.
- Added fail-closed import diagnostics and support-matrix documentation for the
  standalone Drake runtime boundary.

## 0.1.10

- Replaced the `drake` PyPI dependency with the external `drake-uni==0.1.0`
  distribution and aligned the Drake adapter with its batch runtime API.

## 0.1.9

- Added the MuJoCo batch runtime to the `mujoco` optional extra so the
  production adapter is installable from a clean UniSim environment.
- Made conformance checks exercise adapters through their cold-path
  `materialize()` lifecycle before stepping.
- Expanded standalone MuJoCo and Motrix adapter tests to cover full state
  shapes and identity-quaternion reset semantics.

## 0.1.8

- Kept playback video I/O monkeypatchable while preserving lazy optional
  ``imageio`` loading for import isolation.

## 0.1.7

- Corrected factory option translation for the extracted backend adapters.
- Preserved backend-specific validation and fail-closed diagnostics when
  callers pass options from the UniLab owner layer.

## 0.1.6

- Migrated the complete production backend implementations and shared
  subprocess IPC into `unisim-core`.
- Removed the test-only runtime bridge from the public factory so every named
  backend resolves to its concrete adapter and fails closed when unavailable.
- Added support-matrix, migration, and package-boundary documentation for all
  seven adapters.

## 0.1.5

- Exported adapter-specific dependency diagnostics and the shared subprocess
  backend types from the public `unisim` namespace.

## 0.1.4

- Added public Drake, MJWarp, Genesis, IsaacGym and IsaacSim adapter boundaries.
- Added shared subprocess IPC framing used by Isaac worker integrations.
- Promoted all seven UniLab backend identities to the adapter manifest; SDK
  availability remains lazy and fail-closed.

## 0.1.3

- Add the staged adapter identity manifest for all roadmap backends.

## 0.1.2

- Add the lazy Motrix adapter and shared contract smoke coverage.

## 0.1.1

- Add the lazy MuJoCo adapter and backend factory.
- Add MuJoCo contract smoke coverage and adapter documentation.

## 0.1.0

- Bootstrap the `unisim` namespace and `unisim-core` distribution.
- Add the backend-neutral `SimBackend` contract, fake backend, and conformance helper.
- Reserve benchmark case/result interfaces without implementing workloads or measurements.
