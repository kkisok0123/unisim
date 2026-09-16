"""Shared stage-3 bot registry: candidate assets and their control profiles.

Both ``superdex_bot_qualify.py`` and ``superdex_bot_viewer.py`` resolve
per-bot control data from this module, so the qualification profile and the
viewer demonstration cannot drift apart. Each profile defines a research
control profile (not hardware ratings): joint-position targets in radians,
a pre-step PD converter clipped to explicit effort limits in N·m.

Selection follows the stage-1 inventory dispositions: ``profile-candidate``
bots (fixed HARD root, no components/cycles) and ``recipe-candidate`` bots
whose SDK-compiled result stays inside the native loader's feature profile.
Bots needing later capabilities are not registered here; the qualification
runner probes them separately and records their precise blocker.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BotProfile:
    """One candidate asset and the profile used to exercise it."""

    key: str
    relpath: str
    # Explicit effort limits (N·m). None means every authored limit is finite
    # and positive, so the adapter falls back to the authored values.
    effort_limits: tuple[float, ...] | None
    kp: float = 60.0  # uniform P gain; kd is critically damped per joint
    # Armature floor for kd computation on authored armature-0 joints so the
    # PD converter keeps some damping there (research profile choice).
    armature_floor: float = 1e-3
    # Requested sweep amplitude per joint (rad); the runner/viewer bounds it
    # further by a fraction of the reachable range around the default pose.
    sweep_amplitude: float = 0.3
    sweep_period: float = 2.0
    # Joint index used for the penetrating-pose contact probe. None skips the
    # contact check (too few links for a representative self-contact).
    contact_joint: int | None = 1
    # Optional ceiling on reference |qvel| during the sweep (rad/s); None
    # records the value as evidence without a bound. Only the pure fr3 arms
    # keep the stage-2B 10 rad/s guard; near-massless distal joints
    # (googly_eyes, hand/finger combos) legitimately oscillate far faster
    # while staying adapter-vs-SDK exact, so their profiles carry no bound.
    max_qvel_bound: float | None = None
    camera_direction: tuple[float, float, float] | None = None


PROFILES: dict[str, BotProfile] = {}


def _register(profile: BotProfile) -> None:
    if profile.key in PROFILES:
        raise ValueError(f"duplicate bot profile key {profile.key!r}")
    PROFILES[profile.key] = profile


# --- native fixed-base bots (inventory disposition: profile-candidate) --- #

_register(
    BotProfile(
        key="fr3",
        relpath="bots/arms/fr3/fr3.superdex_bot",
        # Authored effort limits are -1 (unlimited); explicit research profile
        # with the SRMS rating shape of the fr3_v2 demonstration.
        effort_limits=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
        kp=400.0,
        sweep_amplitude=0.35,
        max_qvel_bound=10.0,
    )
)
_register(
    BotProfile(
        key="openarm_v20_left_arm",
        relpath="bots/arms/openarm_v20/left/openarm_v20_left_arm.superdex_bot",
        effort_limits=None,  # authored 7..40 N·m, all finite positive
        kp=60.0,
        # Authored armature is 0 on every joint; the 1e-3 default floor gives
        # kd=0.49 which diverges the solver during the viewer sweep. 0.05
        # keeps the demo and qualification sweeps stable with margin.
        armature_floor=0.05,
    )
)
_register(
    BotProfile(
        key="openarm_v20_right_arm",
        relpath="bots/arms/openarm_v20/right/openarm_v20_right_arm.superdex_bot",
        effort_limits=None,
        kp=60.0,
        armature_floor=0.05,
    )
)
_register(
    BotProfile(
        key="googly_eyes",
        relpath="bots/fun/googly_eyes/googly_eyes.superdex_bot",
        effort_limits=(2.0,),  # authored limit is 0
        kp=5.0,
        sweep_amplitude=0.5,
        contact_joint=None,  # two links; no representative self-contact
    )
)
_register(
    BotProfile(
        key="fr3_v2_with_eyes",
        relpath="bots/fun/arm_eyes_combos/fr3_v2_with_eyes.superdex_bot",
        # 7 arm joints (authored -1) + 2 eye joints (authored 0).
        effort_limits=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 2.0, 2.0),
        kp=400.0,
        sweep_amplitude=0.35,
    )
)

# --- recipe compositions (SDK resolves base + AttachBot/ReplaceLinkWithBot) --- #

_register(
    BotProfile(
        key="openarm_v20",
        relpath="bots/arm_hand_combos/openarm_v20/openarm_v20.superdex_bot",
        effort_limits=None,  # authored 7..40 N·m across arm, torso, fingers
        kp=60.0,
        armature_floor=0.05,
    )
)
_register(
    BotProfile(
        key="openarm_v20_wuji",
        relpath="bots/arm_hand_combos/openarm_v20/openarm_v20_wuji.superdex_bot",
        effort_limits=None,  # authored 0.2..40 N·m across 54 joints
        kp=40.0,
        sweep_amplitude=0.25,
        armature_floor=0.05,
    )
)
_register(
    BotProfile(
        key="fr3_dg5f_short_left",
        relpath="bots/arm_hand_combos/fr3_dg5f_short/left/fr3_dg5f_short_left.superdex_bot",
        # 7 arm joints (authored -1) + 20 dg5f finger joints.
        effort_limits=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, *(5.0,) * 20),
        kp=400.0,
    )
)
_register(
    BotProfile(
        key="fr3_dg5f_short_right",
        relpath="bots/arm_hand_combos/fr3_dg5f_short/right/fr3_dg5f_short_right.superdex_bot",
        effort_limits=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, *(5.0,) * 20),
        kp=400.0,
    )
)
_register(
    BotProfile(
        key="fr3_v2_allegro_v5_right",
        relpath="bots/arm_hand_combos/fr3_v2_allegro_v5/right/fr3_v2_allegro_v5_right.superdex_bot",
        # 7 arm joints (authored -1) + 16 allegro joints.
        effort_limits=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, *(7.0,) * 16),
        kp=400.0,
    )
)

# The stage-2B-qualified FR3 v2 stays the reference asset of the previous
# stage; include it so the compatibility table covers it with one code path.
_register(
    BotProfile(
        key="fr3_v2",
        relpath="bots/arms/fr3_v2/fr3_v2.superdex_bot",
        effort_limits=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
        kp=400.0,
        sweep_amplitude=0.35,
        max_qvel_bound=10.0,
    )
)

DEFAULT_KEY = "fr3_v2"

# Stage-4 viewer profiles are kept separate from the fixed-base qualification
# registry. The floating runner records its independent small-effort profile.
FLOATING_PROFILES: dict[str, BotProfile] = {}
for _side in ("left", "right"):
    for _kind in ("allegro_v5", "dg5f_short", "dg5f_long", "wuji_hand2_beta1"):
        _key = f"{_kind}_{_side}"
        FLOATING_PROFILES[_key] = BotProfile(
            key=_key, relpath=f"bots/hands/{_kind}/{_side}/{_key}.superdex_bot",
            effort_limits=(1.0,) * 20 if _kind.startswith("dg5f") else None,
            kp=0.03, armature_floor=1e-6, sweep_amplitude=0.1,
        )
    _key = f"openarm_v20_{_side}_gripper"
    FLOATING_PROFILES[_key] = BotProfile(
        key=_key,
        relpath=f"bots/grippers/openarm_v20/{_side}/{_key}.superdex_bot",
        effort_limits=None, kp=0.03, armature_floor=1e-6, sweep_amplitude=0.1,
        camera_direction=(1.0, 1.0, 1.0),
    )

# Bots with known later-capability blockers, probed for precise evidence
# instead of qualification (stage-4/5/6 work).
BLOCKED_KEYS: dict[str, str] = {
    "fr3_v2_2f_85": "bots/arm_hand_combos/fr3_v2_2f_85/fr3_v2_2f_85.superdex_bot",
    "fr3_dg5f_short_seed_right": (
        "bots/arm_hand_combos/fr3_dg5f_short_seed/right/fr3_dg5f_short_seed_right.superdex_bot"
    ),
    "dg5f_short_seed_left": "bots/hands/dg5f_short_seed/left/dg5f_short_seed_left.superdex_bot",
    "dg5f_short_seed_right": "bots/hands/dg5f_short_seed/right/dg5f_short_seed_right.superdex_bot",
    "dg5f_long_seed_left": "bots/hands/dg5f_long_seed/left/dg5f_long_seed_left.superdex_bot",
    "dg5f_long_seed_right": "bots/hands/dg5f_long_seed/right/dg5f_long_seed_right.superdex_bot",
    "openarm_v20_torso": "bots/torsos/openarm_v20/openarm_v20_torso.superdex_bot",
    "2f_85": "bots/grippers/2f_85/2f_85.superdex_bot",
    "example_bot_2dof": "bots/fun/example_bot_2dof/example_bot_2dof.superdex_bot",
    "dg5f_short_left": "bots/hands/dg5f_short/left/dg5f_short_left.superdex_bot",
    "wuji_hand2_beta1_actuated_left": (
        "bots/hands/wuji_hand2_beta1_actuated/left/wuji_hand2_beta1_actuated_left.superdex_bot"
    ),
    "oculus_xr_hand_highpoly_left": (
        "bots/hands/oculus_xr/left/oculus_xr_hand_highpoly_left.superdex_bot"
    ),
}


@dataclass(frozen=True)
class ResolvedProfile:
    """A bot profile completed with data derived from the authored prefab."""

    profile: BotProfile
    joint_names: tuple[str, ...]
    dof_armature: tuple[float, ...]
    effort_limits: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]

    @property
    def n(self) -> int:
        return len(self.joint_names)


def resolve_gains(
    profile: BotProfile,
    joint_names: Sequence[str],
    dof_armature: Sequence[float],
    authored_efforts: Sequence[float],
) -> ResolvedProfile:
    """Complete a profile with authored joint data.

    ``kd`` is critically damped against each joint's authored inertia with a
    small floor so authored armature-0 joints keep damping. Effort limits come
    from the profile when it defines them, otherwise from the authored values;
    authored -1/0 (unlimited/none) must be overridden explicitly.
    """
    joints = tuple(str(name) for name in joint_names)
    armature = tuple(float(v) for v in dof_armature)
    authored = tuple(float(v) for v in authored_efforts)
    if not len(joints) == len(armature) == len(authored):
        raise ValueError("joint metadata length mismatch")
    if profile.effort_limits is not None:
        if len(profile.effort_limits) != len(joints):
            raise ValueError(
                f"{profile.key}: profile defines {len(profile.effort_limits)} effort "
                f"limits for {len(joints)} joints"
            )
        efforts = tuple(profile.effort_limits)
    else:
        invalid = [i for i, v in enumerate(authored) if not (v > 0 and math.isfinite(v))]
        if invalid:
            raise ValueError(
                f"{profile.key}: authored effort limits are not finite positive at "
                f"joints {invalid}; the profile must supply explicit limits"
            )
        efforts = authored
    kp = (profile.kp,) * len(joints)
    kd = tuple(
        2.0 * math.sqrt(k * (a if a > 0 else profile.armature_floor))
        for k, a in zip(kp, armature)
    )
    return ResolvedProfile(
        profile=profile,
        joint_names=joints,
        dof_armature=armature,
        effort_limits=efforts,
        kp=kp,
        kd=kd,
    )


def sweep_amplitudes(
    profile: BotProfile,
    default_qpos: Sequence[float],
    joint_ranges: Sequence[Sequence[float]],
    reachable_fraction: float = 0.6,
) -> tuple[float, ...]:
    """Bound the profile's sweep request by the reachable range per joint.

    Joints with infinite authored ranges keep the requested amplitude.
    """
    amplitudes = []
    for q0, (lo, hi) in zip(default_qpos, joint_ranges):
        reach = min(
            q0 - lo if math.isfinite(lo) else profile.sweep_amplitude,
            hi - q0 if math.isfinite(hi) else profile.sweep_amplitude,
        )
        amplitudes.append(min(profile.sweep_amplitude, reachable_fraction * max(reach, 0.0)))
    return tuple(amplitudes)
