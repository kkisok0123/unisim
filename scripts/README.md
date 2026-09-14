# Repository scripts

This directory contains maintainer entry points that are deliberately outside `src/unisim` and are therefore not part of the installed package or public API.

- `benchmarks/superdex_scene_step.py` measures the raw SuperDex scene-step and native batch-executor barrier; it is not an RL throughput benchmark.
- `diagnostics/check_newton_runtime.py` checks the pinned Newton distribution metadata and can optionally import the native stack.

Put reusable runtime code in `src/unisim`, regression coverage in the matching `tests/` subtree, and add a script here only when it needs to be a standalone maintainer command. Do not use this directory as an unversioned scratch area.
