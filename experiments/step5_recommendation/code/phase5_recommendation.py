# -*- coding: utf-8 -*-
"""Phase 5: Hyper-Persona recommendation engine with dynamic weighting and reranking."""

from __future__ import annotations

import os
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common_utils import (
    normalize_customer_id,
    normalize_interaction_ids,
    normalize_product_id,
    normalize_user_id,
)


def _paper_repro() -> bool:
    return os.getenv("PAPER_REPRO", "0") == "1"


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _coerce_price(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _safe_qcut(series: pd.Series, labels: list[str]) -> pd.Series:
    filled = pd.to_numeric(series, errors="coerce").fillna(series.median() if series.notna().any() else 0)
    try:
        return pd.qcut(filled, q=len(labels), labels=labels, duplicates="drop").astype(str)
    except Exception:
        return pd.Series([labels[min(len(labels) - 1, 1)]] * len(series), index=series.index)


def _factorize_map(values: pd.Series) -> dict[str, int]:
    uniques = pd.Index(values.fillna("unknown").astype(str).unique())
    return {value: idx for idx, value in enumerate(uniques)}


def _build_user_texts(
    customers: pd.DataFrame,
    user_ids: list[str],
    journey_map: dict[str, str],
) -> list[str]:
    customer_frame = customers.drop_duplicates(subset=["user_id"]).set_index("user_id")
    user_texts = []
    for uid in user_ids:
        row = customer_frame.loc[uid] if uid in customer_frame.index else pd.Series(dtype=object)
        user_texts.append(
            " [SEP] ".join(
                [
                    str(row.get("persona", "")),
                    str(row.get("interest_keywords", "")),
                    str(row.get("purchase_motive", "")),
                    str(row.get("major_purchase_categories", "")),
                    str(journey_map.get(uid, "Unknown")),
                ]
            )
        )
    return user_texts


def _dynamic_weights(
    involvement: float,
    median_involvement: float,
    behavior_label: str,
) -> tuple[float, float]:
    high_engagement_segments = {"VIP Explorer", "New Explorer"}
    is_high_engagement = involvement >= median_involvement or behavior_label in high_engagement_segments
    return (0.4, 0.6) if is_high_engagement else (0.7, 0.3)


def build_cf_score_matrix(
    user_product_ratings: pd.DataFrame,
    user_ids: list[str],
    product_ids: list[str],
) -> tuple[np.ndarray, str]:
    if _paper_repro():
        # Skip Surprise SVD if numpy version is incompatible (numpy 2.x vs compiled with 1.x)
        _surprise_ok = False
        try:
            import importlib.util
            _surprise_ok = importlib.util.find_spec("surprise") is not None
            if _surprise_ok:
                import numpy as _np
                if int(_np.__version__.split(".")[0]) >= 2:
                    _surprise_ok = False  # surprise compiled with numpy 1.x crashes on numpy 2.x
        except Exception:
            _surprise_ok = False

        if _surprise_ok:
            try:
                from surprise import Dataset, Reader, SVD

                ratings_df = user_product_ratings[["user_id", "product_id", "rating"]].copy()
                reader = Reader(rating_scale=(1, 5))
                dataset = Dataset.load_from_df(ratings_df, reader)
                trainset = dataset.build_full_trainset()
                model = SVD(n_factors=20, lr_all=0.05, reg_all=0.02, random_state=42)
                model.fit(trainset)

                scores = np.zeros((len(user_ids), len(product_ids)), dtype=float)
                global_mean = float(trainset.global_mean)
                for user_idx, uid in enumerate(user_ids):
                    try:
                        inner_uid = trainset.to_inner_uid(str(uid))
                        user_bias = float(model.bu[inner_uid])
                        user_factors = model.pu[inner_uid]
                    except ValueError:
                        inner_uid = None
                        user_bias = 0.0
                        user_factors = None

                    for item_idx, pid in enumerate(product_ids):
                        try:
                            inner_iid = trainset.to_inner_iid(str(pid))
                            item_bias = float(model.bi[inner_iid])
                            if user_factors is None:
                                pred = global_mean + item_bias
                            else:
                                pred = global_mean + user_bias + item_bias + float(np.dot(user_factors, model.qi[inner_iid]))
                        except ValueError:
                            pred = global_mean + user_bias
                        scores[user_idx, item_idx] = pred
                return scores, "Surprise SVD (n_factors=20, lr=0.05, reg=0.02)"
            except Exception as exc:
                import warnings
                warnings.warn(f"Surprise SVD failed ({exc}); falling back to NMF.")
        else:
            import warnings
            warnings.warn("Surprise SVD skipped (numpy 2.x incompatibility); falling back to NMF.")

    user_index = {uid: idx for idx, uid in enumerate(user_ids)}
    product_index = {pid: idx for idx, pid in enumerate(product_ids)}
    matrix = np.zeros((len(user_ids), len(product_ids)), dtype=float)

    for _, row in user_product_ratings.iterrows():
        uid = str(row["user_id"])
        pid = str(row["product_id"])
        if uid in user_index and pid in product_index:
            matrix[user_index[uid], product_index[pid]] = float(row["rating"])

    if matrix.size == 0 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return np.full_like(matrix, 2.5, dtype=float), "Constant fallback"

    n_components = min(20, max(2, min(matrix.shape[0] - 1, matrix.shape[1] - 1)))
    model = NMF(n_components=n_components, init="nndsvda", random_state=42, max_iter=400)
    user_factors = model.fit_transform(matrix)
    item_factors = model.components_
    return user_factors @ item_factors, "Matrix Factorization (NMF fallback)"


def build_semantic_score_matrix(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    user_ids: list[str],
    product_ids: list[str],
    journey_map: dict[str, str],
    output_dir: str | None = None,
) -> tuple[np.ndarray, str]:
    customer_frame = customers.drop_duplicates(subset=["user_id"]).set_index("user_id")
    product_frame = products.drop_duplicates(subset=["Uniqe Id"]).set_index("Uniqe Id")

    product_texts = (
        product_frame.loc[product_ids, "combined_text"].fillna("").astype(str)
        + " [SEP] "
        + product_frame.loc[product_ids, "attribute_tags"].fillna("").astype(str)
        + " [SEP] "
        + product_frame.loc[product_ids, "cluster_summary"].fillna("").astype(str)
    )
    user_texts = _build_user_texts(customers, user_ids, journey_map)

    requested_backend = os.getenv("EMBEDDING_BACKEND", "").strip().lower() or ("sbert" if _paper_repro() else "tfidf")
    if requested_backend == "sbert":
        try:
            from sentence_transformers import SentenceTransformer

            model_name = os.getenv("SBERT_MODEL", "paraphrase-mpnet-base-v2")
            batch_size = int(os.getenv("SBERT_BATCH_SIZE", "16" if _paper_repro() else "64"))
            model = SentenceTransformer(model_name)

            product_embeddings = None
            if output_dir:
                emb_path = os.path.join(output_dir, "product_embeddings.npy")
                id_path = os.path.join(output_dir, "product_embedding_ids.csv")
                if os.path.exists(emb_path) and os.path.exists(id_path):
                    emb_ids = pd.read_csv(id_path, low_memory=False)["Uniqe Id"].astype(str).tolist()
                    emb_map = {pid: idx for idx, pid in enumerate(emb_ids)}
                    full_embeddings = np.load(emb_path)
                    product_embeddings = np.vstack([full_embeddings[emb_map[pid]] for pid in product_ids])

            if product_embeddings is None:
                product_embeddings = model.encode(
                    product_texts.tolist(),
                    show_progress_bar=False,
                    batch_size=min(batch_size, max(1, len(product_texts))),
                    normalize_embeddings=False,
                )
            user_embeddings = model.encode(
                user_texts,
                show_progress_bar=False,
                batch_size=min(batch_size, max(1, len(user_texts))),
                normalize_embeddings=False,
            )
            return cosine_similarity(np.asarray(user_embeddings), np.asarray(product_embeddings)), f"SBERT cosine similarity ({model_name})"
        except Exception:
            if _paper_repro():
                raise

    vectorizer = TfidfVectorizer(max_features=2048, stop_words="english", min_df=1, max_df=0.95)
    product_matrix = vectorizer.fit_transform(product_texts.tolist())
    user_matrix = vectorizer.transform(user_texts)
    return cosine_similarity(user_matrix, product_matrix), "Persona semantic TF-IDF"


def main() -> None:
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    fast_mode = os.getenv("FAST_MODE", "0") == "1"
    paper_repro = _paper_repro()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "outputs")

    print("=" * 60)
    print("Phase 5: Hyper-Persona Recommendation Engine")
    print("=" * 60)
    print(f"  Fast mode: {fast_mode}")

    print("\n[5.1] Loading profiles...")
    customers = pd.read_csv(os.path.join(output_dir, "customer_profiles.csv"), low_memory=False)
    products = pd.read_csv(os.path.join(output_dir, "product_profiles.csv"), low_memory=False)
    journey = pd.read_csv(os.path.join(output_dir, "journey_profiles.csv"), low_memory=False)
    interactions = pd.read_csv(os.path.join(output_dir, "combined_interactions.csv"), low_memory=False)

    if "Customer ID" in customers.columns:
        customers["Customer ID"] = customers["Customer ID"].map(normalize_customer_id)
    customers["user_id"] = customers["user_id"].map(normalize_user_id)
    products["Uniqe Id"] = products["Uniqe Id"].map(normalize_product_id)
    journey["user_id"] = journey["user_id"].map(normalize_user_id)
    interactions = normalize_interaction_ids(interactions)
    interactions["interaction_type"] = interactions["interaction_type"].astype(str).str.lower().str.strip()

    print(f"  Customers:    {len(customers):,}")
    print(f"  Products:     {len(products):,}")
    print(f"  Journey:      {len(journey):,}")
    print(f"  Interactions: {len(interactions):,}")

    print("\n[5.2] Preparing matrices and profile features...")
    rating_map = {"view": 1.0, "like": 3.0, "purchase": 5.0}
    interactions["rating"] = interactions["interaction_type"].map(rating_map).fillna(1.0)
    interactions = interactions[(interactions["user_id"] != "") & (interactions["product_id"] != "")]
    interactions = interactions[interactions["product_id"] != "unknown_product"]

    product_frame = products.drop_duplicates(subset=["Uniqe Id"]).copy()
    product_frame["Selling Price"] = _coerce_price(product_frame["Selling Price"])
    product_frame["review_proxy"] = product_frame.get("review_proxy", pd.Series([0.0] * len(product_frame))).fillna(0.0)

    profile_products = set(product_frame["Uniqe Id"].astype(str))
    interaction_products = set(interactions["product_id"].astype(str))
    product_ids = sorted(profile_products & interaction_products)
    user_ids = sorted(interactions["user_id"].dropna().unique())

    product_frame = product_frame[product_frame["Uniqe Id"].isin(product_ids)].copy()
    product_frame = product_frame.drop_duplicates(subset=["Uniqe Id"]).set_index("Uniqe Id").loc[product_ids].reset_index()

    user_product_ratings = interactions.groupby(["user_id", "product_id"])["rating"].max().reset_index()
    cf_scores, cf_model_name = build_cf_score_matrix(user_product_ratings, user_ids, product_ids)

    journey_stage_col = "current_journey_stage" if "current_journey_stage" in journey.columns else "journey_stage"
    journey_map = dict(zip(journey["user_id"], journey[journey_stage_col].fillna("Unknown").astype(str)))
    semantic_scores, semantic_model_name = build_semantic_score_matrix(
        customers,
        product_frame,
        user_ids,
        product_ids,
        journey_map,
        output_dir=output_dir,
    )

    print(f"  Candidate users: {len(user_ids):,}")
    print(f"  Candidate products: {len(product_ids):,}")
    print(f"  CF model: {cf_model_name}")
    print(f"  Semantic model: {semantic_model_name}")

    customer_frame = customers.drop_duplicates(subset=["user_id"]).set_index("user_id")
    product_frame = product_frame.set_index("Uniqe Id")

    all_history = interactions.groupby("user_id")["product_id"].apply(list).to_dict()
    purchase_history = (
        interactions[interactions["interaction_type"] == "purchase"]
        .groupby("user_id")["product_id"]
        .apply(list)
        .to_dict()
    )
    product_popularity = (
        interactions[interactions["interaction_type"] == "purchase"]["product_id"].value_counts(normalize=True).to_dict()
    )
    if not product_popularity:
        product_popularity = interactions["product_id"].value_counts(normalize=True).to_dict()

    interaction_pairs = set(zip(interactions["user_id"].astype(str), interactions["product_id"].astype(str)))
    candidate_topk = int(os.getenv("CANDIDATE_TOPK", "100" if fast_mode else "120"))
    topn_per_user = int(os.getenv("RECO_TOPN", "10"))

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

    user_spend_series = pd.Series(user_mean_spend, dtype=float)
    user_spend_bucket = _safe_qcut(user_spend_series, labels=["low", "mid", "high"]).to_dict()
    product_price_bucket = _safe_qcut(product_frame["Selling Price"], labels=["low", "mid", "high"]).to_dict()

    purchase_cluster_pref = (
        interactions[interactions["interaction_type"] == "purchase"][["user_id", "product_id"]]
        .merge(product_frame[["major_cluster"]], left_on="product_id", right_index=True, how="left")
        .dropna(subset=["major_cluster"])
        .groupby("user_id")["major_cluster"]
        .agg(lambda values: Counter(values.astype(int)).most_common(1)[0][0] if len(values) else -1)
        .to_dict()
    )
    if not purchase_cluster_pref:
        purchase_cluster_pref = defaultdict(lambda: -1)

    gender_encoder = _factorize_map(pd.Series(list(gender_map.values()), dtype=object))
    age_group_encoder = _factorize_map(pd.Series(list(age_group_map.values()), dtype=object))
    cluster_label_encoder = _factorize_map(pd.Series(list(cluster_label_map.values()), dtype=object))
    journey_encoder = _factorize_map(pd.Series(list(journey_map.values()), dtype=object))

    print("\n[5.3] Generating hybrid candidate pool...")
    rows = []
    product_index = {pid: idx for idx, pid in enumerate(product_ids)}
    for user_idx, user_id in enumerate(user_ids):
        seen_items = set(purchase_history.get(user_id, []))
        involvement = float(involvement_map.get(user_id, 0.0))
        cf_weight, llm_weight = _dynamic_weights(
            involvement,
            median_involvement,
            cluster_label_map.get(user_id, "Unclassified"),
        )

        cf_row = cf_scores[user_idx].copy() if len(cf_scores) else np.zeros(len(product_ids), dtype=float)
        llm_row = semantic_scores[user_idx].copy() if len(semantic_scores) else np.zeros(len(product_ids), dtype=float)
        for seen_pid in seen_items:
            if seen_pid in product_index:
                cf_row[product_index[seen_pid]] = -1e9
                llm_row[product_index[seen_pid]] = -1e9

        cf_top_idx = np.argsort(cf_row)[-candidate_topk:][::-1]
        llm_top_idx = np.argsort(llm_row)[-candidate_topk:][::-1]
        candidate_ids = dedupe_preserve_order(
            [product_ids[idx] for idx in cf_top_idx] + [product_ids[idx] for idx in llm_top_idx]
        )

        preferred_cluster = int(purchase_cluster_pref.get(user_id, -1))
        for pid in candidate_ids:
            product_row = product_frame.loc[pid]
            pid_idx = product_index[pid]
            cf_score = float(cf_row[pid_idx])
            llm_score = float(llm_row[pid_idx])
            hybrid_score = cf_weight * cf_score + llm_weight * llm_score
            product_cluster = int(product_row.get("major_cluster", -1)) if pd.notna(product_row.get("major_cluster")) else -1
            overlap_tokens = set(str(customer_frame.loc[user_id].get("interest_keywords", "")).lower().replace("#", " ").replace(",", " ").split()) if user_id in customer_frame.index else set()
            product_tokens = set(
                (
                    str(product_row.get("attribute_tags", ""))
                    + " "
                    + str(product_row.get("tfidf_keywords", ""))
                ).lower().replace(",", " ").split()
            )
            keyword_overlap = len(overlap_tokens & product_tokens)

            rows.append(
                {
                    "user_id": user_id,
                    "product_id": pid,
                    "product_name": str(product_row.get("Product Name", "")),
                    "cf_score": cf_score,
                    "llm_score": llm_score,
                    "hybrid_score": hybrid_score,
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
                    "keyword_overlap": keyword_overlap,
                    "attribute_tags": str(product_row.get("attribute_tags", "")),
                    "cluster_summary": str(product_row.get("cluster_summary", "")),
                }
            )

        if (user_idx + 1) % 500 == 0:
            print(f"  Processed users: {user_idx + 1:,}/{len(user_ids):,}")

    rec_df = pd.DataFrame(rows)
    rec_df["pair_key"] = rec_df["user_id"].astype(str) + "||" + rec_df["product_id"].astype(str)
    rec_df["is_relevant"] = rec_df["pair_key"].isin(
        {f"{uid}||{pid}" for uid, pid in interaction_pairs}
    ).astype(int)

    rec_df["journey_encoded"] = rec_df["journey_stage"].map(journey_encoder).fillna(0).astype(int)
    rec_df["cluster_label_encoded"] = rec_df["behavior_cluster_label"].map(cluster_label_encoder).fillna(0).astype(int)
    rec_df["age_group_encoded"] = rec_df["age_group"].map(age_group_encoder).fillna(0).astype(int)
    rec_df["gender_encoded"] = rec_df["gender"].map(gender_encoder).fillna(0).astype(int)

    print(f"  Candidate rows: {len(rec_df):,}")

    print("\n[5.4] Strategic reranking with LightGBM features...")
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

    total_labels = int(len(rec_df))
    positive_labels = int(rec_df["is_relevant"].sum())
    positive_rate = (positive_labels / total_labels) if total_labels else 0.0
    min_positive_labels = int(os.getenv("LGBM_MIN_POSITIVES", "1" if paper_repro else "100"))
    min_positive_rate = float(os.getenv("LGBM_MIN_POSITIVE_RATE", "0.0" if paper_repro else "0.01"))
    print(f"  Re-ranker labels: positives={positive_labels:,}/{total_labels:,} ({positive_rate:.2%})")

    rec_df["reranked_score"] = rec_df["hybrid_score"]
    can_train_ranker = (
        rec_df["is_relevant"].nunique() > 1
        and positive_labels >= min_positive_labels
        and positive_rate >= min_positive_rate
    )
    ranker_name = "Hybrid score fallback"
    if can_train_ranker:
        try:
            import lightgbm as lgb

            group_sizes = rec_df.groupby("user_id").size().to_numpy()
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
            ranker.fit(rec_df[feature_cols], rec_df["is_relevant"], group=group_sizes)
            rec_df["reranked_score"] = ranker.predict(rec_df[feature_cols])
            ranker.booster_.save_model(os.path.join(output_dir, "lgbm_ranker.txt"))
            ranker_name = "LightGBM Ranker"
        except Exception:
            if paper_repro:
                raise
            try:
                from sklearn.ensemble import HistGradientBoostingClassifier

                fallback_model = HistGradientBoostingClassifier(random_state=42, max_depth=6)
                fallback_model.fit(rec_df[feature_cols], rec_df["is_relevant"])
                rec_df["reranked_score"] = fallback_model.predict_proba(rec_df[feature_cols])[:, 1]
                ranker_name = "HistGradientBoosting fallback"
            except Exception:
                ranker_name = "Hybrid score fallback"

    print(f"  Ranker: {ranker_name}")

    print("\n[5.5] Saving final Top-10 recommendations...")
    final_recs = (
        rec_df.sort_values(["user_id", "reranked_score"], ascending=[True, False])
        .groupby("user_id")
        .head(topn_per_user)
        .copy()
    )
    final_recs = final_recs.drop(columns=["pair_key"])
    final_recs.to_csv(
        os.path.join(output_dir, "recommendations.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 60)
    print("Phase 5 COMPLETE")
    print("=" * 60)
    print(f"  Final recommendation rows: {len(final_recs):,}")
    print(f"  Users served: {final_recs['user_id'].nunique():,}")


if __name__ == "__main__":
    main()
