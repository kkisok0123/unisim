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
driven by the same inputs. It checks:

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
