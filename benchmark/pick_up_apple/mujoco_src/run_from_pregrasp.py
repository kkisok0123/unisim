#!/usr/bin/env python3
"""Run MuJoCo from 3 s, with the arm at the apple and fingers ready to close."""

from __future__ import annotations

from run import main

if __name__ == "__main__":
    raise SystemExit(main(start_time=3.0, default_seconds=11.0))
