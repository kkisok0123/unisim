# Stage 8 native rigid-body qualification

Completed on 2026-09-17 for the declared rigid-body profile. All **55 native target files** are covered: 54 model/prefab/scene files in the table below and the original FR3 pose-controller configuration. The 21 previously qualified robots remain regression coverage; 28 bots now pass in serial mode. Seven custom-component bots and two deformable fixtures remain deferred.

**65 runtime profiles passed:** 54 native files and 11 generated fixtures. Every profile ran 1,000 steps at 0.002 s in one serial environment against an independently loaded SDK scene. The original controller profile separately passed 1,040 steps in the existing controller runner.

## Evidence and limits

- Native state, body transforms and measured contact forces had **zero maximum adapter/SDK deviation**. Maximum state round-trip error was `3.993511199951172e-06`; reset error was `3.990708430379186e-08`. Initial authored states were checked before using identical representable native seeds on both sides; float32 quaternion/frame conversion can round by one ULP.
- Contact queries cover actors with native contact samples. Static/collision-disabled actors can lack samples; query availability is compared on both sides. The generated falling-cube fixture checks nonzero contact response. Articulated skin contact is included through its native actor queries.
- Synthetic tests verify per-actor effort clipping and body-force isolation, spherical coordinate grouping, root transforms and velocities, nonzero coupled motion, complete reset and cleanup/recreation. Separate spherical JSC and pose-controller tests each compare 1,000 steps with the SDK.
- Five 120-frame native renderer smokes cover closed loops, spherical hands, multiple articulations, link tracking and articulated skin. Reports are under `viewer/`. No images were saved. **Manual visual inspection remains outstanding**, including the prior Stage 5 and Stage 7 inspections. The viewer uses the qualified small per-coordinate torque waveform; an earlier unqualified simultaneous-torque waveform caused an Oculus solver divergence and was replaced in the viewer tool, without changing the adapter or solver.
- New capability coverage is one serial environment. Earlier batch regressions remain separate; this report does not promote batch support for new profiles.
- Runtime URDF is not promoted: the installed public loader documents silently dropped primitive visual/collision geometry. Floating OSC retains its tested SDK indexing blocker. Deformables, custom components, image rendering, conversion tooling and solver changes remain deferred. See [SDK limitations](sdk-limitations.json).

## Reproduce

Final implementation validation: `make check` passed with **399 passed, 21 skipped**
and Ruff clean. The full asset-enabled SuperDex suite passed **197 tests**, including
the existing batch regressions. `make package` built the source distribution and wheel; package
inspection confirmed both native asset trees are excluded and the new adapter
modules are included. The refreshed original-bundle inventory verifies 888 files
with no dependency errors; the 84 physics fixture files match their source bytes.

Keep the original bundle at `assets/superdex/` and an unchanged independent copy of the SDK physics asset tree at `assets/superdex-physics/`. Acquiring fixtures is a separate maintainer operation; the qualification command never reads the SDK source checkout or downloads assets. The physics copy contains 84 unchanged files. See [acquisition provenance](physics-fixture-provenance.json). Both asset roots remain local and ignored.

```bash
uv run --no-sync scripts/superdex_rigid_qualify.py
SUPERDEX_ASSETS_PATH="$PWD/assets/superdex" uv run --no-sync scripts/superdex_component_qualify.py \
  --bots bots/arms/fr3_v2/fr3_v2.superdex_bot --effort-limit 1 \
  --controller-config docs/superdex-rigid-qualification/configs/fr3-authored-pose.json \
  --out docs/superdex-rigid-qualification/controller-profile
SUPERDEX_ASSETS_PATH="$PWD/assets/superdex" uv run --no-sync scripts/superdex_component_qualify.py \
  --bots bots/torsos/openarm_v20/openarm_v20_torso.superdex_bot \
  --out docs/superdex-rigid-qualification/passive-torso
uv run --no-sync pytest -q tests/test_superdex_rigid.py
uv run --no-sync scripts/superdex_rigid_viewer.py \
  assets/superdex/bots/grippers/2f_85/2f_85.superdex_bot --frames 120
```

The standalone qualification records every failure and returns nonzero if any profile fails. `--match` selects a diagnostic subset; use a separate `--out` directory for subset runs. Generated fixtures use closed geometry, native serialized coupling definitions, stable archives, cameras and cross-root references. They require no source checkout.

## Provenance

The [summary](summary.json) records SuperDex Physics 1.0.0, platform, installed physics/robotics binary hashes, code commit and hashes of the uncommitted implementation. The [fixture manifest](fixture-manifest.json) records the local files used by that run. Reports retain their original digests; later documentation-only inventory refreshes do not rewrite numerical evidence. Earlier Stage 1–7 reports are unchanged.

## Native controller profile

- [`fr3_v2_pose.superdex_controller`](../../assets/superdex/bots/arms/fr3_v2/control/fr3_v2_pose.superdex_controller): qualified on FR3v2 with the explicit `MOCHI_ARTICULATED_POSE` API; [report](controller-profile/fr3_v2.json).
- OpenArm torso: zero-action serial qualification through the component runner; [report](passive-torso/openarm_v20_torso.json).

## Runtime target matrix

Each passed native entry links to its own report. Disposition follows authored components and native schemas; names are fixture identifiers, not adapter dispatch rules.

| Asset or generated fixture | Status | Controls | Native articulations / rigid actors |
| --- | --- | ---: | ---: |
| [bundle/benchmarks/ant/ant.mochi_prefab](bundle__benchmarks__ant__ant.mochi_prefab.json) | passed | 8 | 1 / 0 |
| [bundle/benchmarks/cart_pole/cart_pole.mochi_scene](bundle__benchmarks__cart_pole__cart_pole.mochi_scene.json) | passed | 2 | 1 / 0 |
| [bundle/benchmarks/half_cheetah/half_cheetah.mochi_scene](bundle__benchmarks__half_cheetah__half_cheetah.mochi_scene.json) | passed | 9 | 1 / 0 |
| [bundle/bots/arm_hand_combos/fr3_dg5f_short/left/fr3_dg5f_short_left.superdex_bot](bundle__bots__arm_hand_combos__fr3_dg5f_short__left__fr3_dg5f_short_left.superdex_bot.json) | passed | 27 | 1 / 0 |
| [bundle/bots/arm_hand_combos/fr3_dg5f_short/right/fr3_dg5f_short_right.superdex_bot](bundle__bots__arm_hand_combos__fr3_dg5f_short__right__fr3_dg5f_short_right.superdex_bot.json) | passed | 27 | 1 / 0 |
| [bundle/bots/arm_hand_combos/fr3_dg5f_short_seed/right/fr3_dg5f_short_seed_right.superdex_bot](bundle__bots__arm_hand_combos__fr3_dg5f_short_seed__right__fr3_dg5f_short_seed_right.superdex_bot.json) | deferred | — | — / — |
| [bundle/bots/arm_hand_combos/fr3_v2_2f_85/fr3_v2_2f_85.superdex_bot](bundle__bots__arm_hand_combos__fr3_v2_2f_85__fr3_v2_2f_85.superdex_bot.json) | passed | 13 | 1 / 0 |
| [bundle/bots/arm_hand_combos/fr3_v2_allegro_v5/right/fr3_v2_allegro_v5_right.superdex_bot](bundle__bots__arm_hand_combos__fr3_v2_allegro_v5__right__fr3_v2_allegro_v5_right.superdex_bot.json) | passed | 23 | 1 / 0 |
| [bundle/bots/arm_hand_combos/openarm_v20/openarm_v20.superdex_bot](bundle__bots__arm_hand_combos__openarm_v20__openarm_v20.superdex_bot.json) | passed | 18 | 1 / 0 |
| [bundle/bots/arm_hand_combos/openarm_v20/openarm_v20_wuji.superdex_bot](bundle__bots__arm_hand_combos__openarm_v20__openarm_v20_wuji.superdex_bot.json) | passed | 54 | 1 / 0 |
| [bundle/bots/arms/fr3/fr3.superdex_bot](bundle__bots__arms__fr3__fr3.superdex_bot.json) | passed | 7 | 1 / 0 |
| [bundle/bots/arms/fr3_v2/fr3_v2.superdex_bot](bundle__bots__arms__fr3_v2__fr3_v2.superdex_bot.json) | passed | 7 | 1 / 0 |
| [bundle/bots/arms/openarm_v20/left/openarm_v20_left_arm.superdex_bot](bundle__bots__arms__openarm_v20__left__openarm_v20_left_arm.superdex_bot.json) | passed | 7 | 1 / 0 |
| [bundle/bots/arms/openarm_v20/right/openarm_v20_right_arm.superdex_bot](bundle__bots__arms__openarm_v20__right__openarm_v20_right_arm.superdex_bot.json) | passed | 7 | 1 / 0 |
| [bundle/bots/fun/arm_eyes_combos/fr3_v2_with_eyes.superdex_bot](bundle__bots__fun__arm_eyes_combos__fr3_v2_with_eyes.superdex_bot.json) | passed | 9 | 1 / 0 |
| [bundle/bots/fun/example_bot_2dof/example_bot_2dof.superdex_bot](bundle__bots__fun__example_bot_2dof__example_bot_2dof.superdex_bot.json) | deferred | — | — / — |
| [bundle/bots/fun/googly_eyes/googly_eyes.superdex_bot](bundle__bots__fun__googly_eyes__googly_eyes.superdex_bot.json) | passed | 1 | 1 / 0 |
| [bundle/bots/grippers/2f_85/2f_85.superdex_bot](bundle__bots__grippers__2f_85__2f_85.superdex_bot.json) | passed | 6 | 1 / 0 |
| [bundle/bots/grippers/openarm_v20/left/openarm_v20_left_gripper.superdex_bot](bundle__bots__grippers__openarm_v20__left__openarm_v20_left_gripper.superdex_bot.json) | passed | 2 | 1 / 0 |
| [bundle/bots/grippers/openarm_v20/right/openarm_v20_right_gripper.superdex_bot](bundle__bots__grippers__openarm_v20__right__openarm_v20_right_gripper.superdex_bot.json) | passed | 2 | 1 / 0 |
| [bundle/bots/hands/allegro_v5/left/allegro_v5_left.superdex_bot](bundle__bots__hands__allegro_v5__left__allegro_v5_left.superdex_bot.json) | passed | 16 | 1 / 0 |
| [bundle/bots/hands/allegro_v5/right/allegro_v5_right.superdex_bot](bundle__bots__hands__allegro_v5__right__allegro_v5_right.superdex_bot.json) | passed | 16 | 1 / 0 |
| [bundle/bots/hands/dg5f_long/left/dg5f_long_left.superdex_bot](bundle__bots__hands__dg5f_long__left__dg5f_long_left.superdex_bot.json) | passed | 20 | 1 / 0 |
| [bundle/bots/hands/dg5f_long/right/dg5f_long_right.superdex_bot](bundle__bots__hands__dg5f_long__right__dg5f_long_right.superdex_bot.json) | passed | 20 | 1 / 0 |
| [bundle/bots/hands/dg5f_long_seed/left/dg5f_long_seed_left.superdex_bot](bundle__bots__hands__dg5f_long_seed__left__dg5f_long_seed_left.superdex_bot.json) | deferred | — | — / — |
| [bundle/bots/hands/dg5f_long_seed/right/dg5f_long_seed_right.superdex_bot](bundle__bots__hands__dg5f_long_seed__right__dg5f_long_seed_right.superdex_bot.json) | deferred | — | — / — |
| [bundle/bots/hands/dg5f_short/left/dg5f_short_left.superdex_bot](bundle__bots__hands__dg5f_short__left__dg5f_short_left.superdex_bot.json) | passed | 20 | 1 / 0 |
| [bundle/bots/hands/dg5f_short/right/dg5f_short_right.superdex_bot](bundle__bots__hands__dg5f_short__right__dg5f_short_right.superdex_bot.json) | passed | 20 | 1 / 0 |
| [bundle/bots/hands/dg5f_short_seed/left/dg5f_short_seed_left.superdex_bot](bundle__bots__hands__dg5f_short_seed__left__dg5f_short_seed_left.superdex_bot.json) | deferred | — | — / — |
| [bundle/bots/hands/dg5f_short_seed/right/dg5f_short_seed_right.superdex_bot](bundle__bots__hands__dg5f_short_seed__right__dg5f_short_seed_right.superdex_bot.json) | deferred | — | — / — |
| [bundle/bots/hands/oculus_xr/left/oculus_xr_hand_highpoly_left.superdex_bot](bundle__bots__hands__oculus_xr__left__oculus_xr_hand_highpoly_left.superdex_bot.json) | passed | 27 | 1 / 0 |
| [bundle/bots/hands/oculus_xr/left/oculus_xr_hand_lowpoly_left.superdex_bot](bundle__bots__hands__oculus_xr__left__oculus_xr_hand_lowpoly_left.superdex_bot.json) | passed | 27 | 1 / 0 |
| [bundle/bots/hands/oculus_xr/right/oculus_xr_hand_highpoly_right.superdex_bot](bundle__bots__hands__oculus_xr__right__oculus_xr_hand_highpoly_right.superdex_bot.json) | passed | 27 | 1 / 0 |
| [bundle/bots/hands/oculus_xr/right/oculus_xr_hand_lowpoly_right.superdex_bot](bundle__bots__hands__oculus_xr__right__oculus_xr_hand_lowpoly_right.superdex_bot.json) | passed | 27 | 1 / 0 |
| [bundle/bots/hands/wuji_hand2_beta1/left/wuji_hand2_beta1_left.superdex_bot](bundle__bots__hands__wuji_hand2_beta1__left__wuji_hand2_beta1_left.superdex_bot.json) | passed | 20 | 1 / 0 |
| [bundle/bots/hands/wuji_hand2_beta1/right/wuji_hand2_beta1_right.superdex_bot](bundle__bots__hands__wuji_hand2_beta1__right__wuji_hand2_beta1_right.superdex_bot.json) | passed | 20 | 1 / 0 |
| [bundle/bots/sensors/dg5f_seed/dg5f_seed.superdex_bot](bundle__bots__sensors__dg5f_seed__dg5f_seed.superdex_bot.json) | deferred | — | — / — |
| [bundle/bots/torsos/openarm_v20/openarm_v20_torso.superdex_bot](bundle__bots__torsos__openarm_v20__openarm_v20_torso.superdex_bot.json) | passed | 0 | 1 / 0 |
| [bundle/prefabs/box_and_blocks/block_blue.mochi_prefab](bundle__prefabs__box_and_blocks__block_blue.mochi_prefab.json) | passed | 0 | 0 / 1 |
| [bundle/prefabs/box_and_blocks/block_green.mochi_prefab](bundle__prefabs__box_and_blocks__block_green.mochi_prefab.json) | passed | 0 | 0 / 1 |
| [bundle/prefabs/box_and_blocks/block_red.mochi_prefab](bundle__prefabs__box_and_blocks__block_red.mochi_prefab.json) | passed | 0 | 0 / 1 |
| [bundle/prefabs/box_and_blocks/block_yellow.mochi_prefab](bundle__prefabs__box_and_blocks__block_yellow.mochi_prefab.json) | passed | 0 | 0 / 1 |
| [bundle/prefabs/box_and_blocks/box_and_blocks.mochi_prefab](bundle__prefabs__box_and_blocks__box_and_blocks.mochi_prefab.json) | passed | 0 | 0 / 33 |
| [bundle/prefabs/chain/chain.mochi_prefab](bundle__prefabs__chain__chain.mochi_prefab.json) | passed | 0 | 0 / 10 |
| [bundle/prefabs/duck_lamp/duck_lamp_recumbent.mochi_prefab](bundle__prefabs__duck_lamp__duck_lamp_recumbent.mochi_prefab.json) | deferred | — | — / — |
| [bundle/prefabs/functional_dexterity_test/fdt_peg.mochi_prefab](bundle__prefabs__functional_dexterity_test__fdt_peg.mochi_prefab.json) | passed | 0 | 0 / 1 |
| [bundle/prefabs/functional_dexterity_test/functional_dexterity_test.mochi_prefab](bundle__prefabs__functional_dexterity_test__functional_dexterity_test.mochi_prefab.json) | passed | 0 | 0 / 17 |
| [bundle/prefabs/nine_hole_peg_test/nine_hole_peg_test.mochi_prefab](bundle__prefabs__nine_hole_peg_test__nine_hole_peg_test.mochi_prefab.json) | passed | 0 | 0 / 10 |
| [bundle/prefabs/paper_cups/paper_cup.mochi_prefab](bundle__prefabs__paper_cups__paper_cup.mochi_prefab.json) | passed | 0 | 0 / 1 |
| [bundle/prefabs/paper_cups/paper_cup_pyramid.mochi_prefab](bundle__prefabs__paper_cups__paper_cup_pyramid.mochi_prefab.json) | passed | 0 | 0 / 11 |
| [bundle/prefabs/shape_box/shape_box.mochi_prefab](bundle__prefabs__shape_box__shape_box.mochi_prefab.json) | passed | 0 | 0 / 14 |
| [bundle/prefabs/sphere/sphere.mochi_prefab](bundle__prefabs__sphere__sphere.mochi_prefab.json) | passed | 0 | 0 / 1 |
| [physics/allegro/allegro.mochi_prefab](physics__allegro__allegro.mochi_prefab.json) | passed | 16 | 1 / 0 |
| [physics/articulated/mixed/mixed_articulation.mochi_prefab](physics__articulated__mixed__mixed_articulation.mochi_prefab.json) | passed | 6 | 1 / 0 |
| [physics/franka_arm/fr3/fr3.mochi_prefab](physics__franka_arm__fr3__fr3.mochi_prefab.json) | passed | 9 | 1 / 0 |
| [physics/samples/articulations_double_pendulum_on_rail.mochi_scene](physics__samples__articulations_double_pendulum_on_rail.mochi_scene.json) | passed | 5 | 1 / 2 |
| [physics/samples/articulations_pose_controller.mochi_scene](physics__samples__articulations_pose_controller.mochi_scene.json) | passed | 5 | 1 / 2 |
| [physics/samples/articulations_skinned_double_pendulum.mochi_scene](physics__samples__articulations_skinned_double_pendulum.mochi_scene.json) | passed | 2 | 1 / 2 |
| [physics/samples/articulations_soft_skinned_double_pendulum.mochi_scene](physics__samples__articulations_soft_skinned_double_pendulum.mochi_scene.json) | deferred | — | — / — |
| [physics/samples/constraints_double_pendulum.mochi_scene](physics__samples__constraints_double_pendulum.mochi_scene.json) | passed | 0 | 0 / 2 |
| [physics/samples/static_environments/ground_plane.mochi_prefab](physics__samples__static_environments__ground_plane.mochi_prefab.json) | passed | 0 | 0 / 1 |
| [physics/samples/tendon_comparison_articulation.mochi_scene](physics__samples__tendon_comparison_articulation.mochi_scene.json) | passed | 4 | 1 / 0 |
| [physics/table/table.mochi_scene](physics__table__table.mochi_scene.json) | passed | 0 | 0 / 1 |
| [synthetic/plain](synthetic__plain.json) | passed | 2 | 1 / 0 |
| [synthetic/transmission](synthetic__transmission.json) | passed | 2 | 1 / 0 |
| [synthetic/tendon](synthetic__tendon.json) | passed | 2 | 1 / 0 |
| [synthetic/ball_bot](synthetic__ball_bot.json) | passed | 4 | 1 / 0 |
| [synthetic/archive](synthetic__archive.json) | passed | 2 | 1 / 0 |
| [synthetic/camera_archive](synthetic__camera_archive.json) | passed | 4 | 1 / 0 |
| [synthetic/spherical](synthetic__spherical.json) | passed | 4 | 1 / 0 |
| [synthetic/multiple](synthetic__multiple.json) | passed | 8 | 2 / 0 |
| [synthetic/nested](synthetic__nested.json) | passed | 8 | 2 / 0 |
| [synthetic/contact](synthetic__contact.json) | passed | 0 | 0 / 2 |
| [synthetic/external](synthetic__external.json) | passed | 2 | 1 / 0 |
