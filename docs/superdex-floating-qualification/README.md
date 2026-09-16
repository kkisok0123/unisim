# Stage 4 complete: native floating roots

Stage 4 is complete for the local asset baseline. The adapter supports authored
parent-joint and joint-link translations and rotations without changing asset
bytes or the public contract. Changes remain unstaged and uncommitted.

## Acceptance evidence

| Stage 4 requirement | Evidence |
| --- | --- |
| Root position, quaternion and velocity conventions | SDK transform composition and root-link Jacobians; independent SDK-free expected-value test |
| Joint offsets and state dimensions | Positions start at 7, velocities at 6; root layout, names and control ordering checked for every model |
| Initial state and geometry | Authored defaults, all link positions, masses and collision AABBs checked against direct SDK loading |
| Movement and state round trips | 1,040 steps per model in each of serial and batch modes; translated/rotated poses and nonzero root/joint velocities |
| Full and selective reset | Public reset restores initial root/joint state and zero velocity; unselected environment remains identical |
| Two-environment isolation | Each untouched environment matches an independently created single-environment backend |
| Controls | Small bounded efforts and positive/negative effort-limit saturation checks |
| Cleanup and recreation | Repeated backend creation, close/double-close, native viewer exit, and initialization/error-path tests |
| Viewer smoke checks | Native viewer completed movement and reset for all ten compatible floating models without saving images |
| Fixed-base compatibility | All eleven previously qualified fixed-base models pass all eight existing qualification checks |

## Current compatibility

All **37 bot models** are accounted for: **21 qualify and 16 have explicit
blockers**. Of the 22 FREE-root models, ten qualify and twelve require
later-stage capabilities. All eleven previously qualified HARD-root models
still pass; four other HARD-root models retain their existing blockers.

The passing floating models are Allegro V5 left/right, DG5F Short left/right,
DG5F Long left/right, unactuated Wuji Hand2 Beta1 left/right, and OpenArm V20
left/right grippers. Both Wuji root orientations now work; authored root
offsets are no longer a blocker.

- [All floating models and precise blockers](all-models/compatibility-table.md)
- [Machine-readable floating summary](all-models/summary.json)
- [Fixed-base regression reports](fixed-base-regression/compatibility-table.md)
- [Viewer measurements](viewers/summary.json)
- [Native interactive close check](interactive-close/close-check.json)

Remaining floating blockers are seven assets with sensor/actuator components,
four Oculus XR assets with spherical joints, and the 2f_85 gripper with
mechanical cycles. These belong to stages 6 and 8. Stage 5 owns independent
objects and object contact. None was bypassed by deleting features.

The four remaining HARD-root blockers are components in example_bot_2dof and
fr3_dg5f_short_seed_right, cycles in fr3_v2_2f_85, and the zero-DoF torso's
batch-executor restriction. The fixed-base runner also probes its historical
blocked list without effort overrides; its DG5F Short left diagnostic does
not supersede that hand's passing floating report with explicit limits.

## Repeatable checks

Run from the repository root with the optional SDK and verified local assets:

    uv run scripts/superdex_floating_qualify.py --all-floating --out docs/superdex-floating-qualification/all-models
    uv run scripts/superdex_bot_qualify.py --out docs/superdex-floating-qualification/fixed-base-regression
    SUPERDEX_ASSETS_PATH="$PWD/assets/superdex" uv run pytest -q tests/test_superdex_native_floating.py
    uv run scripts/superdex_floating_viewers.py
    make check
    make package

Select any discovered floating model with --bots, for example
--bots wuji_hand2_beta1_left. Explicit selection requires every selected model
to pass. The all-floating audit records expected unsupported features as
blocked and continues; numerical or unexpected failures cause a nonzero exit.
Missing assets, hash mismatches and missing runtimes fail the explicit run.

## State and reference method

Canonical root state is world xyz plus wxyz quaternion, with world linear
velocity at the root-link origin and angular velocity in the root-link frame.
Native free velocity uses the parent joint's axes. For authored frames A and B,
root pose is A * native_joint_pose * B. Translating B also introduces the
angular-velocity cross offset contribution to root-origin linear velocity.

Public set_state is first checked against independent SDK inverse transforms
and the SDK Jacobian. For trajectory comparison, both scenes are then seeded
with the same validated native arrays, avoiding tiny float32 quaternion
composition differences selecting different contact/friction branches.
Every subsequent adapter state is compared with direct SDK world poses and
Jacobian-derived velocities. The public round-trip and reset checks remain
separate. The SDK's reconstructed link velocities are also checked at the first
step; they lose precision during prolonged free fall.

The dedicated translated/rotated regression modifies an in-memory prefab:
both reference transforms have nonzero translations and nonidentity rotations.
The source asset remains unchanged. An SDK-free test independently checks a
known 90-degree frame example, including the origin-velocity cross term.

Runs use 0.002-second steps, gravity [0,0,-9.81] m/s², initial root position
[0.2,-0.3,1.5], rotation vector [0.4,-0.2,0.7], and generalized velocities
spanning -0.08 to 0.12. Joint effort is 0.001*sin(3*t + joint_index).
Authored positive limits are used; DG5F receives explicit 1-unit effort limits.
Per-model JSON records all limits, SDK/platform/precision, source provenance,
verified tree digest, base commit, dirty-tree status and implementation hashes.

State tolerances remain 2e-6 absolute plus 1e-6 relative. The maximum observed
component difference is 7.63e-6 during long float32 free fall, within the
combined tolerance. No tolerance was widened for Wuji.

## Viewer lifecycle

Run one interactive demonstration:

    uv run scripts/superdex_bot_viewer.py --bot wuji_hand2_beta1_left

The batch viewer command renders 570 native Polyscope frames per model in
separate processes without saving images. Profiles apply per-link gravity
compensation through UniSim's force API, a bounded finger sweep, and a slow
root translation/rotation. Numerical qualification uses ordinary gravity.
Measurement summaries record root displacement, joint movement and full-state
reset errors.

A real on-screen Wuji run completed movement and reset, received an X11
WM_DELETE_WINDOW event addressed only to its process-owned viewer window, and
exited successfully with its windows destroyed. Separate SDK-free tests cover
window-close handling and exceptions during scene binding, camera placement,
initialization, stepping and rendering; the viewer closes exactly once.

## Validation

- Floating qualification: **10 passed, 12 later-stage blockers, 0 failures**.
- Fixed-base qualification: **11 models passed all 8 checks each**.
- Explicit floating/frame/lifecycle pytest: **23 passed**.
- Full gate: Ruff passed; **315 passed, 18 skipped**. Eleven skipped
  asset-dependent tests are covered by the explicit run.
- Package build passed; the wheel includes root_state.py and excludes local
  bot/HDF5/GLB payloads.
- The original ignored plan is preserved. After qualification, the local asset
  README was updated at the user's request and the inventory checksum refreshed.
  Per-file hashes confirmed every other bundle file unchanged. Reports retain
  the original run's digest; see the inventory's documentation-update record.
- Current acceptance evidence is in all-models/, viewers/,
  fixed-base-regression/ and interactive-close/. Viewer evidence consists of
  numerical reports and lifecycle logs; no Stage 4 images are retained.
