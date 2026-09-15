# SuperDex assets

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
