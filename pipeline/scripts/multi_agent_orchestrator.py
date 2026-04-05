# -*- coding: utf-8 -*-
"""Multi-agent orchestration for the fixed 6-phase research pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: str
    script: str
    deliverable: str


AGENT_TEAM: list[AgentSpec] = [
    AgentSpec(
        name="Agent-DataEngineer",
        role="Preprocess and augment raw datasets",
        script="phase1_preprocess.py",
        deliverable="combined_* and *_processed.csv",
    ),
    AgentSpec(
        name="Agent-CustomerProfiler",
        role="Build customer RFM and persona profiles",
        script="phase2_customer_profiling.py",
        deliverable="customer_profiles.csv and rfm_data.csv",
    ),
    AgentSpec(
        name="Agent-ProductProfiler",
        role="Build product embedding/clustering profiles",
        script="phase3_product_profiling.py",
        deliverable="product_profiles.csv and cluster artifacts",
    ),
    AgentSpec(
        name="Agent-JourneyProfiler",
        role="Classify AIDA journey stages",
        script="phase4_journey_profiling.py",
        deliverable="journey_profiles.csv",
    ),
    AgentSpec(
        name="Agent-RecommendationEngine",
        role="Generate hybrid recommendations and reranking",
        script="phase5_recommendation.py",
        deliverable="recommendations.csv",
    ),
    AgentSpec(
        name="Agent-Evaluator",
        role="Compare baselines and produce final report",
        script="phase6_evaluation.py",
        deliverable="evaluation_results.csv and final_report.md",
    ),
]


def run_multi_agent_team(scripts_dir: str, output_dir: str) -> int:
    """Run all agents in strict dependency order."""
    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(output_dir, "agent_logs")
    os.makedirs(logs_dir, exist_ok=True)

    run_started = datetime.now().isoformat(timespec="seconds")
    manifest = {
        "run_started_at": run_started,
        "execution_mode": "multi_agent_orchestration",
        "agent_count": len(AGENT_TEAM),
        "agents": [asdict(a) for a in AGENT_TEAM],
    }
    _write_json(os.path.join(output_dir, "multi_agent_team_manifest.json"), manifest)

    summary: dict[str, object] = {
        "run_started_at": run_started,
        "status": "running",
        "agents": [],
    }

    print("\n" + "=" * 72)
    print("MULTI-AGENT TEAM ORCHESTRATION STARTED")
    print("=" * 72)

    for index, agent in enumerate(AGENT_TEAM, start=1):
        script_path = os.path.join(scripts_dir, agent.script)
        if not os.path.exists(script_path):
            msg = f"Missing agent script: {script_path}"
            print(msg)
            summary["status"] = "failed"
            summary["failed_reason"] = msg
            _write_json(os.path.join(output_dir, "multi_agent_run_summary.json"), summary)
            return 1

        print("\n" + "-" * 72)
        print(f"[{index}/{len(AGENT_TEAM)}] {agent.name}")
        print(f"  Role       : {agent.role}")
        print(f"  Script     : {agent.script}")
        print(f"  Deliverable: {agent.deliverable}")
        print("-" * 72)

        started_at = time.time()
        return_code = _run_script(agent, script_path, logs_dir)
        elapsed = round(time.time() - started_at, 2)

        agent_result = {
            "name": agent.name,
            "script": agent.script,
            "return_code": return_code,
            "elapsed_sec": elapsed,
            "log_file": os.path.join("agent_logs", f"{agent.name}.log"),
        }
        summary["agents"].append(agent_result)

        if return_code != 0:
            summary["status"] = "failed"
            summary["failed_agent"] = agent.name
            summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
            _write_json(os.path.join(output_dir, "multi_agent_run_summary.json"), summary)
            print(f"\n{agent.name} failed with exit code {return_code}.")
            return return_code

        print(f"{agent.name} completed in {elapsed:.2f}s")

    summary["status"] = "success"
    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(os.path.join(output_dir, "multi_agent_run_summary.json"), summary)

    print("\n" + "=" * 72)
    print("MULTI-AGENT TEAM ORCHESTRATION COMPLETED SUCCESSFULLY")
    print("=" * 72)
    return 0


def _run_script(agent: AgentSpec, script_path: str, logs_dir: str) -> int:
    log_path = os.path.join(logs_dir, f"{agent.name}.log")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["ACTIVE_AGENT_NAME"] = agent.name

    with open(log_path, "w", encoding="utf-8") as log_fp:
        completed = subprocess.run(
            [sys.executable, script_path],
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    print(f"  Log saved: {log_path}")
    return int(completed.returncode)


def _write_json(path: str, payload: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
