# -*- coding: utf-8 -*-
"""Entrypoint: run the existing 6-phase research via multi-agent orchestration."""

from __future__ import annotations

import os
import sys


def resolve_scripts_dir(root: str) -> str:
    direct = os.path.join(root, "pipeline", "scripts")
    if os.path.isdir(direct):
        return direct

    # Fallback: find any folder containing phase scripts.
    for name in os.listdir(root):
        candidate = os.path.join(root, name, "scripts")
        if os.path.isdir(candidate) and os.path.exists(
            os.path.join(candidate, "phase1_preprocess.py")
        ):
            return candidate
    raise FileNotFoundError("Could not find scripts directory containing phase1_preprocess.py")


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = resolve_scripts_dir(root)
    output_dir = os.path.join(os.path.dirname(scripts_dir), "outputs")

    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from multi_agent_orchestrator import run_multi_agent_team

    return run_multi_agent_team(scripts_dir=scripts_dir, output_dir=output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
