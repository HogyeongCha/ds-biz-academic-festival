# -*- coding: utf-8 -*-
"""Phase 4: Journey profiling (AIDA state transition with 30-minute session logic)."""

from __future__ import annotations

import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common_utils import normalize_interaction_ids, parse_timestamp


def classify_journey_stage(
    *,
    last_action: str,
    days_since_last: int,
    session_view_count: int,
    session_like_count: int,
    session_purchase_count: int,
    dormant_days: int = 60,
) -> str:
    """
    Apply the paper's AIDA-style rules.

    Inference note:
    The PDF explicitly defines Dormant, Awareness, Interest, Desire, and Post-Purchase,
    and elsewhere lists Action as part of the six-stage output set.
    We operationalize Action as a recent session containing purchase behavior where the
    final event is not itself the purchase event.
    """

    if days_since_last >= dormant_days:
        return "Dormant"
    if last_action == "purchase":
        return "Post-Purchase"
    if session_purchase_count > 0:
        return "Action"
    if last_action == "like" and session_view_count >= 3:
        return "Desire"
    if last_action == "like" and session_purchase_count == 0:
        return "Interest"
    if last_action == "view" and session_like_count == 0 and session_purchase_count == 0:
        return "Awareness"
    return "Unknown"


def main() -> None:
    warnings.filterwarnings("ignore")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "outputs")

    print("=" * 60)
    print("Phase 4: Journey Profiling")
    print("=" * 60)

    print("\n[4.1] Loading interactions...")
    interactions = pd.read_csv(os.path.join(output_dir, "combined_interactions.csv"), low_memory=False)
    interactions = normalize_interaction_ids(interactions)
    interactions["interaction_type"] = interactions["interaction_type"].astype(str).str.lower().str.strip()
    interactions["timestamp_parsed"] = parse_timestamp(interactions["timestamp"])
    interactions = interactions.dropna(subset=["timestamp_parsed"])
    interactions = interactions.sort_values(["user_id", "timestamp_parsed"])
    reference_date = interactions["timestamp_parsed"].max()

    print(f"  Rows: {len(interactions):,}")
    print(f"  Unique users: {interactions['user_id'].nunique():,}")
    print(f"  Reference date: {reference_date}")

    print("\n[4.2] Building session-based journey profiles...")
    rows = []
    user_ids = interactions["user_id"].dropna().unique()
    for i, uid in enumerate(user_ids):
        udata = interactions[interactions["user_id"] == uid].copy()
        last_row = udata.iloc[-1]
        last_ts = last_row["timestamp_parsed"]
        last_action = str(last_row["interaction_type"]).lower().strip()

        session_start = last_ts - pd.Timedelta(minutes=30)
        session = udata[udata["timestamp_parsed"] >= session_start]

        session_view_count = int((session["interaction_type"] == "view").sum())
        session_like_count = int((session["interaction_type"] == "like").sum())
        session_purchase_count = int((session["interaction_type"] == "purchase").sum())
        days_since_last = int((reference_date - last_ts).days)

        stage = classify_journey_stage(
            last_action=last_action,
            days_since_last=days_since_last,
            session_view_count=session_view_count,
            session_like_count=session_like_count,
            session_purchase_count=session_purchase_count,
        )

        rows.append(
            {
                "user_id": uid,
                "last_interacted_product_id": str(last_row["product_id"]),
                "last_interaction_type": last_action,
                "last_interaction_timestamp": last_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "session_view_count": session_view_count,
                "session_like_count": session_like_count,
                "session_purchase_count": session_purchase_count,
                "session_total_count": int(len(session)),
                "days_since_last": days_since_last,
                "current_journey_stage": stage,
                "journey_stage": stage,
            }
        )

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1:,}/{len(user_ids):,}")

    journey = pd.DataFrame(rows)
    stage_counts = journey["current_journey_stage"].value_counts()
    stage_pcts = (stage_counts / len(journey) * 100).round(1)

    print("\n[4.3] Stage distribution:")
    for stage, count in stage_counts.items():
        pct = float(stage_pcts[stage])
        bar = "#" * int(pct / 2)
        print(f"  {stage:15s}: {count:>6,} ({pct:>5.1f}%) {bar}")

    print("\n[4.4] Saving chart and output...")
    stage_order = ["Dormant", "Awareness", "Interest", "Desire", "Action", "Post-Purchase", "Unknown"]
    stage_colors = {
        "Dormant": "#95a5a6",
        "Awareness": "#3498db",
        "Interest": "#2ecc71",
        "Desire": "#f39c12",
        "Action": "#9b59b6",
        "Post-Purchase": "#e74c3c",
        "Unknown": "#bdc3c7",
    }
    ordered_counts = [int(stage_counts.get(stage, 0)) for stage in stage_order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(
        stage_order,
        ordered_counts,
        color=[stage_colors[stage] for stage in stage_order],
        edgecolor="white",
    )
    axes[0].set_xlabel("users")
    axes[0].set_title("AIDA Journey Stage Distribution")
    if max(ordered_counts) > 0:
        for idx, (count, stage) in enumerate(zip(ordered_counts, stage_order)):
            if count > 0:
                pct = count / len(journey) * 100
                axes[0].text(count + max(ordered_counts) * 0.02, idx, f"{count:,} ({pct:.1f}%)", va="center")

    non_zero = {stage: count for stage, count in zip(stage_order, ordered_counts) if count > 0}
    axes[1].pie(
        non_zero.values(),
        labels=non_zero.keys(),
        colors=[stage_colors[stage] for stage in non_zero.keys()],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[1].set_title("Journey Stage Proportions")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "journey_distribution.png"), dpi=150)
    plt.close()

    journey.to_csv(os.path.join(output_dir, "journey_profiles.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("Phase 4 COMPLETE")
    print("=" * 60)
    print(stage_counts.to_string())


if __name__ == "__main__":
    main()
