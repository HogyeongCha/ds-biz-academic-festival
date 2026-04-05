# -*- coding: utf-8 -*-
"""Phase 2: Customer profiling (demographics + RFM clustering + persona generation)."""

from __future__ import annotations

import os
import warnings
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from common_utils import normalize_customer_id, normalize_interaction_ids, normalize_product_id, parse_timestamp
from llm_utils import summarize_customer_personas_batch


def label_cluster(row: pd.Series, medians: pd.Series) -> str:
    """Map a cluster centroid to a stable behavior label."""

    recency = float(row["recency"])
    frequency = float(row["frequency"])
    monetary = float(row["monetary"])
    diversity = float(row["involvement_diversity"])

    is_recent = recency <= medians["recency"]
    is_frequent = frequency >= medians["frequency"]
    is_high_value = monetary >= medians["monetary"]
    is_high_diversity = diversity >= medians["involvement_diversity"]

    if is_recent and is_frequent and is_high_value and is_high_diversity:
        return "VIP Explorer"
    if is_recent and is_frequent and not is_high_diversity:
        return "Low-Value Loyal"
    if not is_recent and is_high_value and is_frequent:
        return "At-Risk High-Value"
    if is_recent and not is_frequent and is_high_diversity:
        return "New Explorer"
    if not is_recent and not is_frequent:
        return "Dormant"
    return "Regular"


def _determine_optimal_k(rfm_scaled: pd.DataFrame) -> tuple[int, list[int], list[float]]:
    if len(rfm_scaled) < 2:
        return 1, [1], [0.0]

    candidate_ks = [k for k in range(5, 8) if k <= len(rfm_scaled)]
    if not candidate_ks:
        candidate_ks = [k for k in range(2, min(4, len(rfm_scaled)) + 1) if k <= len(rfm_scaled)]
    if not candidate_ks:
        return 1, [1], [0.0]

    inertias: list[float] = []
    for k in candidate_ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        model.fit(rfm_scaled)
        inertias.append(float(model.inertia_))

    if len(candidate_ks) == 1:
        return candidate_ks[0], candidate_ks, inertias
    if len(candidate_ks) == 2:
        return candidate_ks[1], candidate_ks, inertias

    diffs2 = np.diff(np.diff(inertias))
    elbow_idx = int(np.argmax(np.abs(diffs2))) + 1
    return candidate_ks[elbow_idx], candidate_ks, inertias


def _build_purchase_lookup(
    interactions: pd.DataFrame,
    products: pd.DataFrame,
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, str], dict[str, str]]:
    product_meta = (
        products[["Uniqe Id", "Product Name", "Category"]]
        .drop_duplicates(subset=["Uniqe Id"])
        .assign(Uniqe_Id=lambda df: df["Uniqe Id"].map(normalize_product_id))
    )
    product_map = (
        product_meta.rename(columns={"Uniqe_Id": "normalized_product_id"})
        .set_index("normalized_product_id")[["Product Name", "Category"]]
        .to_dict("index")
    )

    purchase_lookup: dict[str, list[tuple[str, str]]] = defaultdict(list)
    top_categories: dict[str, str] = {}
    preview_text: dict[str, str] = {}

    priority_orders = [
        ["purchase"],
        ["purchase", "like"],
        ["purchase", "like", "view"],
    ]
    for priority in priority_orders:
        filtered = interactions[interactions["interaction_type"].isin(priority)].copy()
        filtered = filtered.sort_values(["user_id", "timestamp_parsed"])
        for user_id, group in filtered.groupby("user_id"):
            if user_id in purchase_lookup and purchase_lookup[user_id]:
                continue
            seen: set[str] = set()
            items: list[tuple[str, str]] = []
            for _, row in group.iterrows():
                pid = normalize_product_id(row["product_id"])
                if not pid or pid in seen or pid not in product_map:
                    continue
                seen.add(pid)
                meta = product_map[pid]
                items.append((str(meta["Product Name"]), str(meta["Category"])))
                if len(items) >= 8:
                    break
            if items:
                purchase_lookup[user_id] = items

    for user_id, items in purchase_lookup.items():
        category_counts = Counter(category for _, category in items if category)
        product_names = [name for name, _ in items]
        top_categories[user_id] = ", ".join(category for category, _ in category_counts.most_common(3))
        preview_text[user_id] = " | ".join(product_names[:5])

    return purchase_lookup, top_categories, preview_text


def main() -> None:
    warnings.filterwarnings("ignore")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "outputs")

    print("=" * 60)
    print("Phase 2: Customer Profiling")
    print("=" * 60)

    print("\n[2.1] Loading processed data...")
    customers = pd.read_csv(os.path.join(output_dir, "combined_customers.csv"), low_memory=False)
    interactions = pd.read_csv(os.path.join(output_dir, "combined_interactions.csv"), low_memory=False)
    products = pd.read_csv(os.path.join(output_dir, "combined_products.csv"), low_memory=False)

    if "Customer ID" in customers.columns:
        customers["Customer ID"] = customers["Customer ID"].map(normalize_customer_id)
    customers["user_id"] = customers["Customer ID"].map(normalize_customer_id)

    products["Uniqe Id"] = products["Uniqe Id"].map(normalize_product_id)
    interactions = normalize_interaction_ids(interactions)
    interactions["interaction_type"] = interactions["interaction_type"].astype(str).str.lower().str.strip()
    interactions["timestamp_parsed"] = parse_timestamp(interactions["timestamp"])
    interactions = interactions.dropna(subset=["timestamp_parsed"])

    print(f"  Customers: {len(customers):,}")
    print(f"  Interactions: {len(interactions):,}")
    print(f"  Products: {len(products):,}")

    print("\n[2.2] Demographic profiling...")
    age_bins = [0, 19, 29, 39, 49, 59, 200]
    age_labels = ["10s", "20s", "30s", "40s", "50s", "60+"]
    if "Age" in customers.columns:
        customers["age_group"] = pd.cut(
            pd.to_numeric(customers["Age"], errors="coerce"),
            bins=age_bins,
            labels=age_labels,
            include_lowest=True,
        )
    else:
        customers["age_group"] = "unknown"
    customers["age_group"] = customers["age_group"].astype(str).replace("nan", "unknown")
    customers["preferred_size"] = customers.get("Size", pd.Series(["unknown"] * len(customers))).astype(str)

    print("\n[2.3] Computing RFM + involvement metrics...")
    reference_date = interactions["timestamp_parsed"].max()
    recency = interactions.groupby("user_id")["timestamp_parsed"].max()
    recency = (reference_date - recency).dt.days.rename("recency")

    purchase_mask = interactions["interaction_type"] == "purchase"
    frequency = interactions[purchase_mask].groupby("user_id").size().rename("frequency")

    involvement_mask = interactions["interaction_type"].isin(["view", "like"])
    involvement = (
        interactions[involvement_mask]
        .groupby("user_id")["product_id"]
        .nunique()
        .rename("involvement_diversity")
    )

    monetary_map = (
        customers.groupby("user_id")["Purchase Amount (USD)"].mean()
        if "Purchase Amount (USD)" in customers.columns
        else pd.Series(dtype=float)
    )

    rfm = pd.DataFrame(index=sorted(interactions["user_id"].dropna().unique()))
    rfm = rfm.join(recency).join(frequency).join(involvement)
    rfm["monetary"] = rfm.index.map(lambda x: float(monetary_map.get(x, 0.0)))
    rfm = rfm.fillna(0.0)

    print(f"  RFM users: {len(rfm):,}")
    if len(rfm):
        print(rfm.describe().round(3).to_string())

    print("\n[2.4] Clustering users with K-Means (K=5~7 search)...")
    rfm_features = ["recency", "frequency", "monetary", "involvement_diversity"]
    if len(rfm) < 2:
        rfm["behavior_cluster"] = 0
        optimal_k = 1
        k_values, inertias = [1], [0.0]
        cluster_summary = rfm.groupby("behavior_cluster")[rfm_features].mean().round(3)
    else:
        scaler = MinMaxScaler()
        rfm_scaled = pd.DataFrame(
            scaler.fit_transform(rfm[rfm_features]),
            columns=rfm_features,
            index=rfm.index,
        )
        optimal_k, k_values, inertias = _determine_optimal_k(rfm_scaled)

        final_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        rfm["behavior_cluster"] = final_model.fit_predict(rfm_scaled)
        cluster_summary = rfm.groupby("behavior_cluster")[rfm_features].mean().round(3)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, "bo-")
    plt.axvline(optimal_k, color="r", linestyle="--", label=f"K={optimal_k}")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.title("Elbow Method (Customer Profiling)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "elbow_plot.png"), dpi=150)
    plt.close()

    medians = rfm[rfm_features].median() if len(rfm) else pd.Series({f: 0 for f in rfm_features})
    cluster_labels = cluster_summary.apply(lambda row: label_cluster(row, medians), axis=1)
    label_map = cluster_labels.to_dict()
    rfm["behavior_cluster_label"] = rfm["behavior_cluster"].map(label_map)
    cluster_summary["label"] = cluster_labels
    cluster_summary.to_csv(os.path.join(output_dir, "cluster_summary.csv"))

    print(f"  Optimal K: {optimal_k}")
    print(f"  Cluster count: {rfm['behavior_cluster'].nunique()}")

    print("\n[2.5] Building customer profiles...")
    purchase_lookup, top_categories, purchase_preview = _build_purchase_lookup(interactions, products)

    customer_profiles = customers.copy()
    customer_profiles = customer_profiles.merge(
        rfm[
            [
                "behavior_cluster",
                "behavior_cluster_label",
                "recency",
                "frequency",
                "monetary",
                "involvement_diversity",
            ]
        ],
        left_on="user_id",
        right_index=True,
        how="left",
    )
    customer_profiles["major_purchase_categories"] = customer_profiles["user_id"].map(top_categories).fillna("")
    customer_profiles["purchase_history_preview"] = customer_profiles["user_id"].map(purchase_preview).fillna("")

    print("\n[2.6] Generating persona summaries...")
    unique_profiles = customer_profiles.drop_duplicates(subset=["user_id"])
    persona_batch_size = int(os.getenv("PERSONA_BATCH_SIZE", "20"))
    persona_records = [
        {
            "user_id": str(row["user_id"]),
            "age_group": str(row.get("age_group", "unknown")),
            "gender": str(row.get("Gender", "Unknown")),
            "cluster_label": str(row.get("behavior_cluster_label", "Unclassified")),
            "purchase_items": purchase_lookup.get(str(row["user_id"]), []),
        }
        for _, row in unique_profiles.iterrows()
    ]
    persona_lookup: dict[str, dict[str, str]] = {}
    for start in range(0, len(persona_records), persona_batch_size):
        chunk = persona_records[start : start + persona_batch_size]
        persona_lookup.update(summarize_customer_personas_batch(chunk, batch_size=persona_batch_size))
        print(f"  Persona prompts: {min(start + len(chunk), len(persona_records)):,}/{len(persona_records):,}")

    persona_df = customer_profiles["user_id"].map(lambda uid: persona_lookup.get(str(uid), {})).apply(pd.Series)
    customer_profiles = pd.concat([customer_profiles.reset_index(drop=True), persona_df.reset_index(drop=True)], axis=1)

    customer_profiles["behavior_cluster_label"] = customer_profiles["behavior_cluster_label"].fillna("Unclassified")
    customer_profiles["persona"] = customer_profiles["persona"].fillna("Insufficient interaction data.")
    customer_profiles["interest_keywords"] = customer_profiles["interest_keywords"].fillna("#general")
    customer_profiles["purchase_motive"] = customer_profiles["purchase_motive"].fillna("unknown")

    print("\n[2.7] Saving outputs...")
    customer_profiles.to_csv(
        os.path.join(output_dir, "customer_profiles.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    rfm.reset_index(names="user_id").to_csv(
        os.path.join(output_dir, "rfm_data.csv"),
        index=False,
    )

    print("\n" + "=" * 60)
    print("Phase 2 COMPLETE")
    print("=" * 60)
    print(f"  Total profiles: {len(customer_profiles):,}")
    print(f"  Classified users: {(customer_profiles['behavior_cluster_label'] != 'Unclassified').sum():,}")


if __name__ == "__main__":
    main()
