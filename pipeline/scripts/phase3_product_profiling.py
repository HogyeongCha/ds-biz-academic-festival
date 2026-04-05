# -*- coding: utf-8 -*-
"""Phase 3: Product profiling (semantic DNA + HDBSCAN clustering + attribute tags)."""

from __future__ import annotations

import json
import os
import warnings
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from common_utils import normalize_product_id
from llm_utils import summarize_product_attribute_tags


def _is_fast_mode() -> bool:
    return os.getenv("FAST_MODE", "0") == "1"


def _paper_repro() -> bool:
    return os.getenv("PAPER_REPRO", "0") == "1"


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _build_combined_text(products: pd.DataFrame) -> pd.Series:
    fields = [
        "Product Name",
        "Category",
        "About Product",
        "Product Details",
        "Product Description",
        "Technical Details",
    ]
    parts = []
    for field in fields:
        parts.append(products.get(field, pd.Series([""] * len(products))).fillna("").astype(str))
    combined = parts[0]
    for block in parts[1:]:
        combined = combined + " [SEP] " + block
    return combined.str.replace(r"\s+", " ", regex=True).str.strip()


def _coerce_price(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _get_embeddings(texts: list[str], fast_mode: bool) -> tuple[np.ndarray, str]:
    requested_backend = os.getenv("EMBEDDING_BACKEND", "").strip().lower()
    if not requested_backend:
        requested_backend = "sbert"

    if requested_backend == "sbert":
        try:
            from sentence_transformers import SentenceTransformer

            model_name = os.getenv("SBERT_MODEL", "paraphrase-mpnet-base-v2")
            batch_size = _get_int_env("SBERT_BATCH_SIZE", 16 if _paper_repro() else (128 if fast_mode else 96))
            model = SentenceTransformer(model_name)
            chunks = []
            for i in range(0, len(texts), batch_size):
                chunk = model.encode(
                    texts[i : i + batch_size],
                    show_progress_bar=False,
                    batch_size=min(batch_size, len(texts[i : i + batch_size])),
                    normalize_embeddings=False,
                )
                chunks.append(np.asarray(chunk))
            return np.vstack(chunks), f"SBERT:{model_name}"
        except Exception:
            if _paper_repro():
                raise

    tfidf_features = _get_int_env("TFIDF_MAX_FEATURES", 768 if not fast_mode else 384)
    tfidf = TfidfVectorizer(max_features=tfidf_features, stop_words="english", min_df=2, max_df=0.95)
    return tfidf.fit_transform(texts).toarray(), "TF-IDF"


def _reduce_dims(embeddings: np.ndarray, fast_mode: bool) -> tuple[np.ndarray, str]:
    n_samples, n_features = embeddings.shape
    n_pca = min(40, n_features - 1, n_samples - 1)
    n_pca = max(2, n_pca)
    reduced = PCA(n_components=n_pca, random_state=42).fit_transform(embeddings)

    try:
        import umap

        n_umap = min(10, reduced.shape[1] - 1)
        n_umap = max(2, n_umap)
        reducer = umap.UMAP(
            n_components=n_umap,
            random_state=42,
            n_neighbors=15 if not fast_mode else 10,
            min_dist=0.1,
            low_memory=True,
        )
        return reducer.fit_transform(reduced), "PCA(40)+UMAP(10)"
    except Exception:
        if _paper_repro():
            raise
        n_small = min(10, reduced.shape[1])
        n_small = max(2, n_small)
        return PCA(n_components=n_small, random_state=42).fit_transform(reduced), "PCA fallback"


def _score_clustering(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    mask = labels >= 0
    n_clusters = len(set(labels[mask])) if mask.any() else 0
    n_noise = int((labels == -1).sum())
    if n_clusters > 1 and mask.sum() > n_clusters:
        return {
            "n_clusters": float(n_clusters),
            "n_noise": float(n_noise),
            "silhouette": float(silhouette_score(features[mask], labels[mask])),
            "calinski_harabasz": float(calinski_harabasz_score(features[mask], labels[mask])),
            "davies_bouldin": float(davies_bouldin_score(features[mask], labels[mask])),
        }
    return {
        "n_clusters": float(n_clusters),
        "n_noise": float(n_noise),
        "silhouette": 0.0,
        "calinski_harabasz": 0.0,
        "davies_bouldin": 0.0,
    }


def _run_hdbscan(features: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
    import hdbscan

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        core_dist_n_jobs=1,
        prediction_data=False,
    )
    return model.fit_predict(features)


def _cluster_products(
    reduced_embeddings: np.ndarray,
    fast_mode: bool,
) -> tuple[np.ndarray, pd.DataFrame, str]:
    try:
        import hdbscan  # noqa: F401

        run_grid = os.getenv("RUN_CLUSTER_GRID", "1" if _paper_repro() else "0") == "1"
        if run_grid:
            trial_rows: list[dict[str, float | int]] = []
            for min_cluster_size in [50, 100, 150, 200]:
                for min_samples in [50, 100]:
                    labels = _run_hdbscan(reduced_embeddings, min_cluster_size, min_samples)
                    scores = _score_clustering(reduced_embeddings, labels)
                    trial_rows.append(
                        {
                            "min_cluster_size": min_cluster_size,
                            "min_samples": min_samples,
                            **scores,
                        }
                    )
            experiments = pd.DataFrame(trial_rows)
            best_row = experiments.sort_values(
                ["silhouette", "calinski_harabasz", "davies_bouldin"],
                ascending=[False, False, True],
            ).iloc[0]
            major_labels = _run_hdbscan(
                reduced_embeddings,
                int(best_row["min_cluster_size"]),
                int(best_row["min_samples"]),
            )
        else:
            major_labels = _run_hdbscan(reduced_embeddings, 50, 50)
            scores = _score_clustering(reduced_embeddings, major_labels)
            experiments = pd.DataFrame(
                [
                    {
                        "min_cluster_size": 50,
                        "min_samples": 50,
                        **scores,
                    }
                ]
            )
        return major_labels, experiments, "HDBSCAN"
    except Exception:
        if _paper_repro():
            raise
        from sklearn.cluster import KMeans

        n_clusters = min(166 if not fast_mode else 64, max(8, len(reduced_embeddings) // 80))
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        major_labels = model.fit_predict(reduced_embeddings)
        scores = _score_clustering(reduced_embeddings, major_labels)
        experiments = pd.DataFrame(
            [
                {
                    "min_cluster_size": np.nan,
                    "min_samples": np.nan,
                    **scores,
                }
            ]
        )
        return major_labels, experiments, "KMeans fallback"


def _subcluster_products(reduced_embeddings: np.ndarray, major_labels: np.ndarray) -> tuple[np.ndarray, int, str]:
    sub_labels = np.full(len(major_labels), -1, dtype=int)
    next_label = 0
    used_hdbscan = False

    for cluster_id in sorted(set(major_labels)):
        if cluster_id == -1:
            continue
        mask = major_labels == cluster_id
        cluster_size = int(mask.sum())
        if cluster_size < 10:
            sub_labels[mask] = next_label
            next_label += 1
            continue

        try:
            labels = _run_hdbscan(reduced_embeddings[mask], 10, 5)
            used_hdbscan = True
        except Exception:
            if _paper_repro():
                raise
            from sklearn.cluster import KMeans

            n_sub = min(5, max(2, cluster_size // 50))
            labels = KMeans(n_clusters=n_sub, random_state=42, n_init=10).fit_predict(reduced_embeddings[mask])

        cluster_local_labels = [label for label in sorted(set(labels)) if label >= 0]
        if not cluster_local_labels:
            sub_labels[mask] = next_label
            next_label += 1
            continue

        mapped = np.full(cluster_size, -1, dtype=int)
        for local_label in cluster_local_labels:
            mapped[labels == local_label] = next_label
            next_label += 1
        mapped[mapped < 0] = next_label
        next_label += 1
        sub_labels[mask] = mapped

    return sub_labels, next_label, "HDBSCAN" if used_hdbscan else "KMeans fallback"


def _extract_keywords(cluster_documents: dict[int, str], limit: int = 7) -> dict[int, list[str]]:
    if not cluster_documents:
        return {}
    try:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        cluster_ids = sorted(cluster_documents)
        docs = [cluster_documents[cluster_id] for cluster_id in cluster_ids]
        matrix = vectorizer.fit_transform(docs)
        tf = matrix.toarray()
        tf = tf / np.maximum(tf.sum(axis=1, keepdims=True), 1e-12)
        doc_freq = (matrix > 0).sum(axis=0).A1
        avg_nr_samples = np.mean(matrix.sum(axis=1).A1)
        idf = np.log((avg_nr_samples + 1.0) / (doc_freq + 1.0)) + 1.0
        scores = tf * idf
        vocab = vectorizer.get_feature_names_out()
        out: dict[int, list[str]] = {}
        for row_idx, cluster_id in enumerate(cluster_ids):
            top_idx = scores[row_idx].argsort()[-limit:][::-1]
            out[int(cluster_id)] = [str(vocab[i]) for i in top_idx if scores[row_idx][i] > 0]
        return out
    except Exception:
        out: dict[int, list[str]] = {}
        for cluster_id, text in cluster_documents.items():
            tokens: Counter[str] = Counter()
            for token in str(text).lower().split():
                if len(token) >= 4:
                    tokens[token] += 1
            out[int(cluster_id)] = [token for token, _ in tokens.most_common(limit)]
        return out


def main() -> None:
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    fast_mode = _is_fast_mode()
    paper_repro = _paper_repro()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "outputs")

    print("=" * 60)
    print("Phase 3: Product Profiling")
    print("=" * 60)

    print("\n[3.1] Loading products...")
    products = pd.read_csv(os.path.join(output_dir, "combined_products.csv"), low_memory=False)
    products["Uniqe Id"] = products["Uniqe Id"].map(normalize_product_id)
    products["Selling Price"] = _coerce_price(products.get("Selling Price", pd.Series([np.nan] * len(products))))
    max_products = _get_int_env("MAX_PRODUCTS", 0 if paper_repro else (12000 if fast_mode else 0))
    if max_products > 0 and len(products) > max_products:
        products = products.sample(n=max_products, random_state=42).reset_index(drop=True)

    products["combined_text"] = _build_combined_text(products)
    products["text_length"] = products["combined_text"].str.len()

    print(f"  Products: {len(products):,}")
    print(f"  Fast mode: {fast_mode}")
    print(f"  Avg text length: {products['text_length'].mean():.1f}")

    print("\n[3.2] Embedding products...")
    embeddings, embedding_method = _get_embeddings(products["combined_text"].tolist(), fast_mode)
    np.save(os.path.join(output_dir, "product_embeddings.npy"), embeddings)
    products[["Uniqe Id"]].to_csv(
        os.path.join(output_dir, "product_embedding_ids.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"  Embeddings: {embeddings.shape} ({embedding_method})")

    print("\n[3.3] Reducing dimensions...")
    reduced_embeddings, reduction_method = _reduce_dims(embeddings, fast_mode)
    print(f"  Reduced dims: {reduced_embeddings.shape[1]} ({reduction_method})")

    print("\n[3.4] Major clustering...")
    major_labels, experiments, clustering_method = _cluster_products(reduced_embeddings, fast_mode)
    experiments.to_csv(os.path.join(output_dir, "hdbscan_experiments.csv"), index=False)
    products["major_cluster"] = major_labels

    n_major = len({label for label in set(major_labels) if label >= 0})
    noise_count = int((major_labels == -1).sum())
    print(f"  Major clusters: {n_major}")
    print(f"  Noise points: {noise_count}")

    print("\n[3.5] Sub clustering...")
    sub_labels, total_subclusters, subcluster_method = _subcluster_products(reduced_embeddings, major_labels)
    products["sub_cluster"] = sub_labels
    print(f"  Sub clusters: {total_subclusters}")

    print("\n[3.6] Keyword extraction + attribute tagging...")
    cluster_documents = {
        int(cluster_id): " ".join(
            products.loc[products["major_cluster"] == cluster_id, "combined_text"].astype(str).tolist()
        )
        for cluster_id in sorted(set(major_labels))
        if cluster_id != -1
    }
    cluster_keywords = _extract_keywords(cluster_documents, limit=7)
    cluster_tags: dict[int, dict[str, str]] = {}
    for cluster_id in sorted(cluster_documents):
        if cluster_id == -1:
            continue
        keywords = cluster_keywords.get(int(cluster_id), [])

        sample_products = (
            products.loc[products["major_cluster"] == cluster_id, "Product Name"]
            .dropna()
            .astype(str)
            .head(8)
            .tolist()
        )
        cluster_tags[int(cluster_id)] = summarize_product_attribute_tags(
            cluster_id=int(cluster_id),
            keywords=keywords,
            sample_products=sample_products,
        )

    products["tfidf_keywords"] = products["major_cluster"].map(
        lambda x: ", ".join(cluster_keywords.get(int(x), [])) if pd.notna(x) and int(x) >= 0 else ""
    )
    products["attribute_tags"] = products["major_cluster"].map(
        lambda x: cluster_tags.get(int(x), {}).get("attribute_tags", "") if pd.notna(x) and int(x) >= 0 else ""
    )
    products["cluster_summary"] = products["major_cluster"].map(
        lambda x: cluster_tags.get(int(x), {}).get("cluster_summary", "") if pd.notna(x) and int(x) >= 0 else ""
    )

    with open(os.path.join(output_dir, "cluster_keywords.json"), "w", encoding="utf-8") as fp:
        json.dump({str(k): v for k, v in cluster_keywords.items()}, fp, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "cluster_attribute_tags.json"), "w", encoding="utf-8") as fp:
        json.dump({str(k): v for k, v in cluster_tags.items()}, fp, ensure_ascii=False, indent=2)

    print("\n[3.7] Saving visualization...")
    if reduced_embeddings.shape[1] > 2:
        coords = PCA(n_components=2, random_state=42).fit_transform(reduced_embeddings)
    else:
        coords = reduced_embeddings

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=major_labels, cmap="tab20", alpha=0.35, s=5)
    plt.title(f"Product Clusters (major={n_major})")
    plt.xlabel("component_1")
    plt.ylabel("component_2")
    plt.colorbar(scatter, label="cluster_id")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "product_clusters_viz.png"), dpi=150)
    plt.close()

    print("\n[3.8] Saving product profiles...")
    products[
        [
            "Uniqe Id",
            "Product Name",
            "Category",
            "Selling Price",
            "major_cluster",
            "sub_cluster",
            "tfidf_keywords",
            "attribute_tags",
            "cluster_summary",
            "combined_text",
        ]
    ].to_csv(
        os.path.join(output_dir, "product_profiles.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 60)
    print("Phase 3 COMPLETE")
    print("=" * 60)
    print(f"  Embedding method: {embedding_method}")
    print(f"  Reduction method: {reduction_method}")
    print(f"  Clustering method: {clustering_method}")
    print(f"  Sub clustering method: {subcluster_method}")


if __name__ == "__main__":
    main()
