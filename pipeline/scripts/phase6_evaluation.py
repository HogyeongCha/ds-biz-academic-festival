# -*- coding: utf-8 -*-
"""Phase 6: Temporal-split evaluation and final analysis."""

from __future__ import annotations

import os
import warnings
from collections import Counter, defaultdict
from math import log2

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

from common_utils import normalize_interaction_ids, normalize_product_id, normalize_user_id, parse_timestamp
from phase5_recommendation import (
    _coerce_price,
    _dynamic_weights,
    _factorize_map,
    _safe_qcut,
    build_cf_score_matrix,
    build_semantic_score_matrix,
    dedupe_preserve_order,
)


def _paper_repro() -> bool:
    return os.getenv("PAPER_REPRO", "0") == "1"


def precision_at_k(recommended: list[str], relevant: set[str], k: int = 10) -> float:
    rec_k = recommended[:k]
    if not relevant:
        return 0.0
    return len(set(rec_k) & relevant) / max(k, 1)


def recall_at_k(recommended: list[str], relevant: set[str], k: int = 10) -> float:
    rec_k = recommended[:k]
    if not relevant:
        return 0.0
    return len(set(rec_k) & relevant) / len(relevant)


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int = 10) -> float:
    rec_k = recommended[:k]
    dcg = sum(1.0 / log2(i + 2) for i, item in enumerate(rec_k) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def coverage(all_recommendations: list[list[str]], total_items: int) -> float:
    if total_items <= 0:
        return 0.0
    seen = set()
    for rec in all_recommendations:
        seen.update(rec)
    return len(seen) / total_items * 100


def novelty_score(recommended: list[str], item_popularity: dict[str, float]) -> float:
    vals = []
    for item in recommended:
        pop = float(item_popularity.get(item, 1e-10))
        if pop > 0:
            vals.append(-log2(pop))
    return float(np.mean(vals)) if vals else 0.0


def intra_list_similarity(
    recommended: list[str],
    embeddings: np.ndarray | None,
    id_to_idx: dict[str, int],
) -> float:
    if embeddings is None:
        return 0.0
    idx = [id_to_idx[x] for x in recommended if x in id_to_idx and id_to_idx[x] < len(embeddings)]
    if len(idx) < 2:
        return 0.0
    embs = embeddings[idx]
    sim = cos_sim(embs)
    n = len(idx)
    return float((sim.sum() - n) / (n * (n - 1)))


def temporal_split(interactions: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    interactions = interactions.sort_values("timestamp_parsed").reset_index(drop=True)
    split_idx = max(1, min(len(interactions) - 1, int(len(interactions) * train_ratio)))
    return interactions.iloc[:split_idx].copy(), interactions.iloc[split_idx:].copy()


def per_user_temporal_split(
    interactions: pd.DataFrame,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_rows = []
    test_rows = []
    for _, group in interactions.sort_values(["user_id", "timestamp_parsed"]).groupby("user_id"):
        if len(group) < 2:
            train_rows.append(group)
            continue
        split_idx = max(1, int(len(group) * train_ratio))
        split_idx = min(split_idx, len(group) - 1)
        train_rows.append(group.iloc[:split_idx])
        test_rows.append(group.iloc[split_idx:])
    train_data = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame(columns=interactions.columns)
    test_data = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=interactions.columns)
    return train_data, test_data


def _build_candidate_rows(
    *,
    customers: pd.DataFrame,
    product_frame: pd.DataFrame,
    user_ids: list[str],
    product_ids: list[str],
    cf_scores: np.ndarray,
    semantic_scores: np.ndarray,
    journey_map: dict[str, str],
    all_history: dict[str, list[str]],
    purchase_history: dict[str, list[str]],
    product_popularity: dict[str, float],
    exclude_seen_mode: str,
    candidate_topk: int,
) -> pd.DataFrame:
    customer_frame = customers.drop_duplicates(subset=["user_id"]).set_index("user_id")
    product_frame = product_frame.set_index("Uniqe Id")
    product_index = {pid: idx for idx, pid in enumerate(product_ids)}

    involvement_map = customer_frame.get("involvement_diversity", pd.Series(dtype=float)).fillna(0.0).to_dict()
    median_involvement = float(np.median(list(involvement_map.values()))) if involvement_map else 0.0
    gender_map = customer_frame.get("Gender", pd.Series(dtype=object)).fillna("Unknown").astype(str).to_dict()
    age_group_map = customer_frame.get("age_group", pd.Series(dtype=object)).fillna("unknown").astype(str).to_dict()
    cluster_label_map = (
        customer_frame.get("behavior_cluster_label", pd.Series(dtype=object)).fillna("Unclassified").astype(str).to_dict()
    )
    behavior_cluster_map = (
        customer_frame.get("behavior_cluster", pd.Series(dtype=float)).fillna(-1).astype(int).to_dict()
    )
    user_mean_spend = (
        pd.to_numeric(customer_frame.get("Purchase Amount (USD)", pd.Series(dtype=float)), errors="coerce")
        .fillna(0.0)
        .to_dict()
    )
    user_spend_bucket = _safe_qcut(pd.Series(user_mean_spend, dtype=float), labels=["low", "mid", "high"]).to_dict()
    product_price_bucket = _safe_qcut(product_frame["Selling Price"], labels=["low", "mid", "high"]).to_dict()

    purchase_cluster_pref = (
        pd.DataFrame(
            [
                {"user_id": uid, "product_id": pid}
                for uid, items in purchase_history.items()
                for pid in items
            ]
        )
        .merge(product_frame[["major_cluster"]], left_on="product_id", right_index=True, how="left")
        .dropna(subset=["major_cluster"])
        .groupby("user_id")["major_cluster"]
        .agg(lambda values: Counter(values.astype(int)).most_common(1)[0][0] if len(values) else -1)
        .to_dict()
    )

    gender_encoder = _factorize_map(pd.Series(list(gender_map.values()), dtype=object))
    age_group_encoder = _factorize_map(pd.Series(list(age_group_map.values()), dtype=object))
    cluster_label_encoder = _factorize_map(pd.Series(list(cluster_label_map.values()), dtype=object))
    journey_encoder = _factorize_map(pd.Series(list(journey_map.values()), dtype=object))

    rows = []
    for user_idx, user_id in enumerate(user_ids):
        all_seen = set(all_history.get(user_id, []))
        purchase_seen = set(purchase_history.get(user_id, []))
        if exclude_seen_mode == "all":
            exclude_items = all_seen
        elif exclude_seen_mode == "purchase":
            exclude_items = purchase_seen
        else:
            exclude_items = set()

        involvement = float(involvement_map.get(user_id, 0.0))
        cf_weight, llm_weight = _dynamic_weights(
            involvement,
            median_involvement,
            cluster_label_map.get(user_id, "Unclassified"),
        )

        cf_row = cf_scores[user_idx].copy() if len(cf_scores) else np.zeros(len(product_ids), dtype=float)
        llm_row = semantic_scores[user_idx].copy() if len(semantic_scores) else np.zeros(len(product_ids), dtype=float)
        for pid in exclude_items:
            if pid in product_index:
                cf_row[product_index[pid]] = -1e9
                llm_row[product_index[pid]] = -1e9

        cf_top_idx = np.argsort(cf_row)[-candidate_topk:][::-1]
        llm_top_idx = np.argsort(llm_row)[-candidate_topk:][::-1]
        candidate_ids = dedupe_preserve_order(
            [product_ids[idx] for idx in cf_top_idx] + [product_ids[idx] for idx in llm_top_idx]
        )

        preferred_cluster = int(purchase_cluster_pref.get(user_id, -1))
        interest_tokens = (
            set(str(customer_frame.loc[user_id].get("interest_keywords", "")).lower().replace("#", " ").replace(",", " ").split())
            if user_id in customer_frame.index
            else set()
        )

        for pid in candidate_ids:
            pid_idx = product_index[pid]
            product_row = product_frame.loc[pid]
            product_cluster = int(product_row.get("major_cluster", -1)) if pd.notna(product_row.get("major_cluster")) else -1
            product_tokens = set(
                (
                    str(product_row.get("attribute_tags", ""))
                    + " "
                    + str(product_row.get("tfidf_keywords", ""))
                ).lower().replace(",", " ").split()
            )
            rows.append(
                {
                    "user_id": user_id,
                    "product_id": pid,
                    "cf_score": float(cf_row[pid_idx]),
                    "llm_score": float(llm_row[pid_idx]),
                    "hybrid_score": float(cf_weight * cf_row[pid_idx] + llm_weight * llm_row[pid_idx]),
                    "cf_weight": cf_weight,
                    "llm_weight": llm_weight,
                    "journey_stage": journey_map.get(user_id, "Unknown"),
                    "behavior_cluster_label": cluster_label_map.get(user_id, "Unclassified"),
                    "behavior_cluster": int(behavior_cluster_map.get(user_id, -1)),
                    "involvement_diversity": involvement,
                    "age_group": age_group_map.get(user_id, "unknown"),
                    "gender": gender_map.get(user_id, "Unknown"),
                    "product_price": float(product_row.get("Selling Price", 0.0) or 0.0),
                    "product_popularity": float(product_popularity.get(pid, 0.0)),
                    "major_cluster": product_cluster,
                    "sub_cluster": int(product_row.get("sub_cluster", -1)) if pd.notna(product_row.get("sub_cluster")) else -1,
                    "preferred_cluster_match": int(preferred_cluster == product_cluster and preferred_cluster >= 0),
                    "price_sensitivity_match": int(
                        str(user_spend_bucket.get(user_id, "mid")) == str(product_price_bucket.get(pid, "mid"))
                    ),
                    "keyword_overlap": len(interest_tokens & product_tokens),
                }
            )

    candidate_df = pd.DataFrame(rows)
    candidate_df["journey_encoded"] = candidate_df["journey_stage"].map(journey_encoder).fillna(0).astype(int)
    candidate_df["cluster_label_encoded"] = candidate_df["behavior_cluster_label"].map(cluster_label_encoder).fillna(0).astype(int)
    candidate_df["age_group_encoded"] = candidate_df["age_group"].map(age_group_encoder).fillna(0).astype(int)
    candidate_df["gender_encoded"] = candidate_df["gender"].map(gender_encoder).fillna(0).astype(int)
    return candidate_df


def _fit_ranker(train_candidate_df: pd.DataFrame, interaction_pairs: set[tuple[str, str]], feature_cols: list[str]):
    train_candidate_df = train_candidate_df.copy()
    train_candidate_df["pair_key"] = train_candidate_df["user_id"].astype(str) + "||" + train_candidate_df["product_id"].astype(str)
    pair_keys = {f"{uid}||{pid}" for uid, pid in interaction_pairs}
    train_candidate_df["label"] = train_candidate_df["pair_key"].isin(pair_keys).astype(int)
    positives = int(train_candidate_df["label"].sum())
    min_positive_labels = 1 if _paper_repro() else 100
    if train_candidate_df["label"].nunique() <= 1 or positives < min_positive_labels:
        return None
    try:
        import lightgbm as lgb

        group_sizes = train_candidate_df.groupby("user_id").size().to_numpy()
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        )
        ranker.fit(train_candidate_df[feature_cols], train_candidate_df["label"], group=group_sizes)
        return ranker
    except Exception:
        if _paper_repro():
            raise
        return None


def main() -> None:
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    fast_mode = os.getenv("FAST_MODE", "0") == "1"
    paper_repro = _paper_repro()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "outputs")

    print("=" * 60)
    print("Phase 6: Evaluation and Analysis")
    print("=" * 60)
    print(f"  Fast mode: {fast_mode}")

    print("\n[6.1] Loading data...")
    interactions = pd.read_csv(os.path.join(output_dir, "combined_interactions.csv"), low_memory=False)
    customers = pd.read_csv(os.path.join(output_dir, "customer_profiles.csv"), low_memory=False)
    products = pd.read_csv(os.path.join(output_dir, "product_profiles.csv"), low_memory=False)
    journey = pd.read_csv(os.path.join(output_dir, "journey_profiles.csv"), low_memory=False)

    interactions = normalize_interaction_ids(interactions)
    interactions["interaction_type"] = interactions["interaction_type"].astype(str).str.lower().str.strip()
    interactions["timestamp_parsed"] = parse_timestamp(interactions["timestamp"])
    interactions = interactions.dropna(subset=["timestamp_parsed"])
    interactions = interactions[interactions["product_id"] != "unknown_product"].copy()
    interactions["rating"] = interactions["interaction_type"].map({"view": 1.0, "like": 3.0, "purchase": 5.0}).fillna(1.0)

    customers["user_id"] = customers["user_id"].map(normalize_user_id)
    products["Uniqe Id"] = products["Uniqe Id"].map(normalize_product_id)
    products["Selling Price"] = _coerce_price(products["Selling Price"])
    journey["user_id"] = journey["user_id"].map(normalize_user_id)

    print(f"  Interactions: {len(interactions):,}")
    print(f"  Users: {interactions['user_id'].nunique():,}")
    print(f"  Products: {interactions['product_id'].nunique():,}")

    print("\n[6.2] Creating temporal split...")
    split_method = "global temporal split (80/20)"
    train_data, test_data = temporal_split(interactions, train_ratio=0.8)

    train_history_all = train_data.groupby("user_id")["product_id"].apply(list).to_dict()
    train_purchase_history = (
        train_data[train_data["interaction_type"] == "purchase"]
        .groupby("user_id")["product_id"]
        .apply(list)
        .to_dict()
    )
    test_ground_truth = test_data.groupby("user_id")["product_id"].apply(set).to_dict()
    eval_users = [uid for uid in test_ground_truth.keys() if uid in train_history_all]
    min_overlap = max(100, int(interactions["user_id"].nunique() * 0.1))
    if len(eval_users) < min_overlap and not paper_repro:
        split_method = "per-user temporal split fallback (80/20)"
        train_data, test_data = per_user_temporal_split(interactions, train_ratio=0.8)
        train_history_all = train_data.groupby("user_id")["product_id"].apply(list).to_dict()
        train_purchase_history = (
            train_data[train_data["interaction_type"] == "purchase"]
            .groupby("user_id")["product_id"]
            .apply(list)
            .to_dict()
        )
        test_ground_truth = test_data.groupby("user_id")["product_id"].apply(set).to_dict()
        eval_users = [uid for uid in test_ground_truth.keys() if uid in train_history_all]
        print("  Global split overlap was too small for stable offline evaluation.")
        print(f"  Falling back to {split_method}.")

    print(f"  Split method: {split_method}")
    print(f"  Train interactions: {len(train_data):,}")
    print(f"  Test interactions:  {len(test_data):,}")
    eval_limit = int(os.getenv("EVAL_USERS", "0"))
    if eval_limit > 0:
        eval_users = eval_users[:eval_limit]
    print(f"  Eval users with overlap: {len(eval_users):,}")

    full_product_order = products.drop_duplicates(subset=["Uniqe Id"])["Uniqe Id"].astype(str).tolist()
    embeddings_path = os.path.join(output_dir, "product_embeddings.npy")
    product_embeddings = np.load(embeddings_path) if os.path.exists(embeddings_path) else None
    if product_embeddings is not None and len(product_embeddings) != len(full_product_order):
        n = min(len(product_embeddings), len(full_product_order))
        product_embeddings = product_embeddings[:n]
        full_product_order = full_product_order[:n]
    embedding_product_idx = {pid: idx for idx, pid in enumerate(full_product_order)}

    product_ids = sorted(set(products["Uniqe Id"].astype(str)) & set(train_data["product_id"].astype(str)))
    product_frame = products[products["Uniqe Id"].isin(product_ids)].drop_duplicates(subset=["Uniqe Id"]).copy()
    journey_stage_col = "current_journey_stage" if "current_journey_stage" in journey.columns else "journey_stage"
    journey_map = dict(zip(journey["user_id"], journey[journey_stage_col].fillna("Unknown").astype(str)))
    item_popularity = (train_data["product_id"].value_counts() / max(len(train_data), 1)).to_dict()

    print("\n[6.3] Training baseline models...")
    user_product_train = train_data.groupby(["user_id", "product_id"])["rating"].max().reset_index()
    cf_scores, cf_model_name = build_cf_score_matrix(user_product_train, eval_users, product_ids)
    semantic_scores, semantic_model_name = build_semantic_score_matrix(
        customers,
        product_frame,
        eval_users,
        product_ids,
        journey_map,
        output_dir=output_dir,
    )
    print(f"  CF baseline: {cf_model_name}")
    print(f"  Content baseline: {semantic_model_name}")

    user_index = {uid: idx for idx, uid in enumerate(eval_users)}
    product_index = {pid: idx for idx, pid in enumerate(product_ids)}
    popular_rank = train_data["product_id"].value_counts().index.tolist()

    from scipy.sparse import csr_matrix

    train_matrix_source = user_product_train.copy()
    train_matrix_source["user_idx"] = train_matrix_source["user_id"].map(user_index)
    train_matrix_source["product_idx"] = train_matrix_source["product_id"].map(product_index)
    train_matrix_source = train_matrix_source.dropna(subset=["user_idx", "product_idx"])
    train_matrix = csr_matrix(
        (
            train_matrix_source["rating"].astype(float),
            (
                train_matrix_source["user_idx"].astype(int),
                train_matrix_source["product_idx"].astype(int),
            ),
        ),
        shape=(len(eval_users), len(product_ids)),
    )
    item_limit = min(train_matrix.shape[1], int(os.getenv("IBCF_ITEM_LIMIT", "1500" if not fast_mode else "1000")))
    item_sim = cos_sim(train_matrix[:, :item_limit].T.toarray()) if item_limit > 1 else np.zeros((item_limit, item_limit))

    def recommend_popular(user_id: str, k: int = 10, exclude: set[str] | None = None) -> list[str]:
        exclude = exclude or set()
        return [pid for pid in popular_rank if pid not in exclude][:k]

    def recommend_ibcf(user_id: str, k: int = 10, exclude: set[str] | None = None) -> list[str]:
        exclude = exclude or set()
        if user_id not in user_index or item_limit < 2:
            return recommend_popular(user_id, k, exclude)
        uidx = user_index[user_id]
        user_vec = train_matrix[uidx].toarray().ravel()
        rated = np.where(user_vec[:item_limit] > 0)[0]
        if len(rated) == 0:
            return recommend_popular(user_id, k, exclude)
        scores = np.zeros(item_limit)
        for idx in rated:
            scores += item_sim[idx] * user_vec[idx]
        for idx in rated:
            scores[idx] = -1e9
        top_idx = np.argsort(scores)[-k * 4 :][::-1]
        recs = []
        for idx in top_idx:
            pid = product_ids[idx]
            if pid not in exclude:
                recs.append(pid)
            if len(recs) >= k:
                break
        return recs

    def recommend_mf(user_id: str, k: int = 10, exclude: set[str] | None = None) -> list[str]:
        exclude = exclude or set()
        if user_id not in user_index or len(cf_scores) == 0:
            return recommend_popular(user_id, k, exclude)
        scores = cf_scores[user_index[user_id]].copy()
        for pid in exclude:
            if pid in product_index:
                scores[product_index[pid]] = -1e9
        top_idx = np.argsort(scores)[-k:][::-1]
        return [product_ids[idx] for idx in top_idx]

    def recommend_content(user_id: str, k: int = 10, exclude: set[str] | None = None) -> list[str]:
        exclude = exclude or set()
        if user_id not in user_index or len(semantic_scores) == 0:
            return recommend_popular(user_id, k, exclude)
        scores = semantic_scores[user_index[user_id]].copy()
        for pid in exclude:
            if pid in product_index:
                scores[product_index[pid]] = -1e9
        top_idx = np.argsort(scores)[-k:][::-1]
        return [product_ids[idx] for idx in top_idx]

    def recommend_simple_hybrid(user_id: str, k: int = 10, exclude: set[str] | None = None) -> list[str]:
        exclude = exclude or set()
        if user_id not in user_index or len(cf_scores) == 0 or len(semantic_scores) == 0:
            return recommend_popular(user_id, k, exclude)
        scores = 0.5 * cf_scores[user_index[user_id]] + 0.5 * semantic_scores[user_index[user_id]]
        for pid in exclude:
            if pid in product_index:
                scores[product_index[pid]] = -1e9
        top_idx = np.argsort(scores)[-k:][::-1]
        return [product_ids[idx] for idx in top_idx]

    feature_cols = [
        "cf_score",
        "llm_score",
        "hybrid_score",
        "cf_weight",
        "llm_weight",
        "behavior_cluster",
        "involvement_diversity",
        "journey_encoded",
        "cluster_label_encoded",
        "age_group_encoded",
        "gender_encoded",
        "product_price",
        "product_popularity",
        "major_cluster",
        "sub_cluster",
        "preferred_cluster_match",
        "price_sensitivity_match",
        "keyword_overlap",
    ]
    candidate_topk = int(os.getenv("CANDIDATE_TOPK", "100" if fast_mode else "120"))
    train_candidate_df = _build_candidate_rows(
        customers=customers,
        product_frame=product_frame,
        user_ids=eval_users,
        product_ids=product_ids,
        cf_scores=cf_scores,
        semantic_scores=semantic_scores,
        journey_map=journey_map,
        all_history=train_history_all,
        purchase_history=train_purchase_history,
        product_popularity=item_popularity,
        exclude_seen_mode="purchase",
        candidate_topk=candidate_topk,
    )
    ranker = _fit_ranker(
        train_candidate_df,
        interaction_pairs=set(zip(train_data["user_id"].astype(str), train_data["product_id"].astype(str))),
        feature_cols=feature_cols,
    )
    eval_candidate_df = _build_candidate_rows(
        customers=customers,
        product_frame=product_frame,
        user_ids=eval_users,
        product_ids=product_ids,
        cf_scores=cf_scores,
        semantic_scores=semantic_scores,
        journey_map=journey_map,
        all_history=train_history_all,
        purchase_history=train_purchase_history,
        product_popularity=item_popularity,
        exclude_seen_mode="purchase",
        candidate_topk=candidate_topk,
    )
    if ranker is not None:
        eval_candidate_df["reranked_score"] = ranker.predict(eval_candidate_df[feature_cols])
    else:
        eval_candidate_df["reranked_score"] = eval_candidate_df["hybrid_score"]

    hyper_persona_recs = (
        eval_candidate_df.sort_values(["user_id", "reranked_score"], ascending=[True, False])
        .groupby("user_id")["product_id"]
        .apply(list)
        .to_dict()
    )

    def recommend_hyper_persona(user_id: str, k: int = 10, exclude: set[str] | None = None) -> list[str]:
        exclude = exclude or set()
        items = [pid for pid in hyper_persona_recs.get(user_id, []) if pid not in exclude]
        return items[:k]

    models = {
        "Most Popular": recommend_popular,
        "IBCF": recommend_ibcf,
        "Matrix Factorization (SVD-style)": recommend_mf,
        "Content-Based": recommend_content,
        "Simple Hybrid": recommend_simple_hybrid,
        "Hyper-Persona Engine": recommend_hyper_persona,
    }

    print("\n[6.4] Evaluating models...")
    K = int(os.getenv("EVAL_K", "10"))
    eval_results: dict[str, dict[str, float]] = {}
    for model_name, recommend_fn in models.items():
        precisions: list[float] = []
        recalls: list[float] = []
        ndcgs: list[float] = []
        ils_scores: list[float] = []
        all_recs: list[list[str]] = []

        for uid in eval_users:
            relevant = test_ground_truth.get(uid, set())
            if not relevant:
                continue
            seen = set(train_purchase_history.get(uid, []))
            rec = recommend_fn(uid, k=K, exclude=seen)
            rec = dedupe_preserve_order(rec)[:K]
            if not rec:
                continue
            precisions.append(precision_at_k(rec, relevant, K))
            recalls.append(recall_at_k(rec, relevant, K))
            ndcgs.append(ndcg_at_k(rec, relevant, K))
            all_recs.append(rec)
            ils_scores.append(intra_list_similarity(rec, product_embeddings, embedding_product_idx))

        cov = coverage(all_recs, len(product_ids)) if all_recs else 0.0
        nov = float(np.mean([novelty_score(rec, item_popularity) for rec in all_recs])) if all_recs else 0.0
        eval_results[model_name] = {
            "Precision@10": round(float(np.mean(precisions)), 3) if precisions else 0.0,
            "Recall@10": round(float(np.mean(recalls)), 3) if recalls else 0.0,
            "NDCG@10": round(float(np.mean(ndcgs)), 3) if ndcgs else 0.0,
            "Coverage (%)": round(cov, 1),
            "ILS": round(float(np.mean(ils_scores)), 3) if ils_scores else 0.0,
            "Novelty": round(nov, 2),
        }
        metrics = eval_results[model_name]
        print(
            f"  {model_name:30s} "
            f"P@10={metrics['Precision@10']:.3f} "
            f"R@10={metrics['Recall@10']:.3f} "
            f"NDCG@10={metrics['NDCG@10']:.3f} "
            f"Cov={metrics['Coverage (%)']:.1f}% "
            f"Nov={metrics['Novelty']:.2f}"
        )

    results_df = pd.DataFrame(eval_results).T
    results_df.index.name = "Model"
    results_df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), encoding="utf-8-sig")

    print("\n[6.5] Saving evaluation chart...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = ["Precision@10", "Recall@10", "NDCG@10", "Coverage (%)", "ILS", "Novelty"]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#95a5a6", "#1abc9c"]
    names = list(eval_results.keys())
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3][idx % 3]
        values = [eval_results[name][metric] for name in names]
        bars = ax.barh(names, values, color=colors[idx], alpha=0.85, edgecolor="white")
        ax.set_title(metric)
        ax.set_xlabel(metric)
        if values:
            best_idx = int(np.argmax(values)) if metric != "ILS" else int(np.argmin(values))
            bars[best_idx].set_edgecolor("black")
            bars[best_idx].set_linewidth(2)
            max_v = max(values) if max(values) > 0 else 1.0
            for i, value in enumerate(values):
                ax.text(value + max_v * 0.02, i, f"{value:.3f}", va="center", fontsize=9)
    plt.suptitle("Recommendation Model Comparison (Temporal 80/20 Split)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_chart.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("\n[6.6] Writing final report...")
    best_model = results_df["Precision@10"].idxmax() if not results_df.empty else "N/A"
    report = f"""# Hyper-Persona Evaluation Report

## Data split
- Method: {split_method}
- Train interactions: {len(train_data):,}
- Test interactions: {len(test_data):,}
- Eval users: {len(eval_users):,}

## Model summary
- CF baseline: {cf_model_name}
- Content baseline: {semantic_model_name}
- Ranker: {"LightGBM ranker" if ranker is not None else "hybrid fallback"}
- Best Precision@10 model: {best_model}

## Metrics

{results_df.to_markdown()}
"""
    with open(os.path.join(output_dir, "final_report.md"), "w", encoding="utf-8") as fp:
        fp.write(report)

    print("\n" + "=" * 60)
    print("Phase 6 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
