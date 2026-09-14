# Wuji Hand 2 (Beta 1) Actuated Description

## Overview

This package is a derived version of the [Wuji Hand 2 (Beta 1)](https://github.com/wuji-technology/wuji-description/tree/1407beed7f478f6ba472d1c42bbdee6f4eec8f7f/hand2/hand2_beta1) SuperDex asset. It adds link-level actuator and sensor blocks to the hand description, using per-joint position-servo gains carried over from the upstream MuJoCo MJCF files (`hand2/hand2_beta1/body/mjcf/{left,right}.xml`).

All geometry (CAD, collision, render) is **referenced in place** from the base package `assets/bots/hands/wuji_hand2_beta1/` via `//` root-relative paths; no geometry files are duplicated or modified.

## What was added

- **20 actuators per hand** (type `WUJI_POSITION_SERVO`), one per actuated revolute joint, attached to the joint's child link. Params carry the upstream MJCF calibration verbatim: `k_p` (MJCF kp), `k_d` (MJCF kv), `effort_limit` (MJCF forcerange magnitude). Upstream notes these gains are carried over from the Wuji Hand platform calibration, pending Wuji Hand 2 system identification.
- **5 sensors per hand** (type `WUJI_CONTACT_FORCE_SENSOR`), one on each fingertip pad link (`{l,r}_index_finger_pad`, `_middle_finger_pad`, `_ring_finger_pad`, `_pinky_pad`, `_thumb_pad`), exposing world-frame contact force.
- **Joint armature** (rotor inertia) on every revolute joint, from the upstream MJCF: `0.0005` kg·m² on MCP-flex and thumb-CMC joints, `0.0002` elsewhere. Without it the tiny distal-link inertias make the explicit dynamics unstable at 200 Hz — the hand flails to its joint limits under a PD hold even at zero gravity; with it, PD-hold drift stays at ~0.05 rad (gravity sag) and a 2 s fist-close tracks within ~0.07 rad.
- **Left-hand contact overrides**: the base left-hand asset ships without `contactOverrides`; this package mirrors the right hand's 140 adjacent-link collision-filter pairs with `r_` → `l_` name substitution so both hands filter self-collision identically (142 pairs total per hand including the fixes below).
- **Two appended contact overrides per hand** fixing a base-asset rest-pose defect: `{l,r}_pinky_middle ↔ {l,r}_pinky_pad` and `{l,r}_pinky_middle ↔ {l,r}_pinky_nail`. The pinky reuses index-finger collision geometry and interpenetrates its own pad and nail at the default pose, producing ~1 N of contact force at rest that escalates to tens of newtons once the hand moves (empirically isolated by pairwise contact toggling).

## Runtime registration required

Actuator and sensor types are not built into SuperDex. Before spawning this asset, register the types from Python, e.g. with `robotics.register_python_actuator(ctx, "WUJI_POSITION_SERVO", ...)` and `robotics.register_python_sensor(ctx, "WUJI_CONTACT_FORCE_SENSOR", ...)`. Unregistered types are skipped with a warning at load time — the asset still loads.

## License

The upstream Wuji Hand 2 assets are provided under the [MIT License](https://github.com/wuji-technology/wuji-description/blob/main/LICENSE) (Copyright (c) 2025 Wuji Technology). The actuator gains carried over from the upstream MJCF files are subject to the same license.

**You are responsible for ensuring your use is compatible with all third-party licenses.**
