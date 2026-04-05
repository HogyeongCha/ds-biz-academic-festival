# -*- coding: utf-8 -*-
"""Optional Gemini helpers with deterministic local fallbacks."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


_STOPWORDS = {
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "into",
    "your",
    "their",
    "have",
    "been",
    "more",
    "less",
    "very",
    "also",
    "about",
    "product",
    "products",
    "category",
    "categories",
    "item",
    "items",
    "shopper",
    "customer",
    "users",
    "user",
}

_CACHE: dict[str, dict[str, str]] | None = None


def _wants_gemini() -> bool:
    return os.getenv("USE_GEMINI", "0") == "1"


def _paper_repro() -> bool:
    return os.getenv("PAPER_REPRO", "0") == "1"


def _cache_path() -> Path:
    custom = os.getenv("LLM_CACHE_PATH", "").strip()
    if custom:
        return Path(custom)
    return Path(__file__).resolve().parents[1] / "outputs" / "llm_cache.json"


def _load_cache() -> dict[str, dict[str, str]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    path = _cache_path()
    if path.exists():
        try:
            _CACHE = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _CACHE = {}
    else:
        _CACHE = {}
    return _CACHE


def _save_cache() -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache()
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _cache_key(kind: str, payload: dict[str, object]) -> str:
    raw = json.dumps({"kind": kind, "payload": payload}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(kind: str, payload: dict[str, object]) -> dict[str, str] | None:
    cache = _load_cache()
    key = _cache_key(kind, payload)
    value = cache.get(key)
    if not isinstance(value, dict):
        return None
    return {str(k): str(v) for k, v in value.items()}


def _cache_set(kind: str, payload: dict[str, object], value: dict[str, str]) -> dict[str, str]:
    cache = _load_cache()
    cache[_cache_key(kind, payload)] = {str(k): str(v) for k, v in value.items()}
    _save_cache()
    return value


def _extract_json_block(text: str) -> dict[str, object] | None:
    text = text.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _extract_json_array(text: str) -> list[dict[str, object]] | None:
    text = text.strip()
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return [item for item in parsed if isinstance(item, dict)]


def _get_gemini_model():
    if not _wants_gemini():
        return None
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        if _paper_repro():
            raise RuntimeError("PAPER_REPRO requires a working GEMINI_API_KEY or GOOGLE_API_KEY.")
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        return genai.GenerativeModel(model_name)
    except Exception:
        if _paper_repro():
            raise
        return None


def _normalize_keyword_list(values: Iterable[str], limit: int = 5) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value).strip().lower().replace("#", "")
        token = re.sub(r"[^a-z0-9+\- ]", " ", token)
        token = re.sub(r"\s+", " ", token).strip()
        if len(token) < 3 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        cleaned.append(token)
        if len(cleaned) >= limit:
            break
    return cleaned


def _top_terms(texts: Iterable[str], limit: int = 5) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+\-]{2,}", str(text).lower()):
            if token in _STOPWORDS:
                continue
            counter[token] += 1
    return _normalize_keyword_list([token for token, _ in counter.most_common(limit * 3)], limit=limit)


def summarize_customer_persona(
    *,
    age_group: str,
    gender: str,
    cluster_label: str,
    purchase_items: list[tuple[str, str]],
    journey_stage: str | None = None,
) -> dict[str, str]:
    """Return persona sentence, interest keywords and purchase motive."""

    gender_text = (gender or "Unknown").strip() or "Unknown"
    age_text = (age_group or "unknown").strip() or "unknown"
    cluster_text = (cluster_label or "Regular").strip() or "Regular"
    journey_text = (journey_stage or "Unknown").strip() or "Unknown"
    purchase_preview = [
        {"product_name": name, "category": category}
        for name, category in purchase_items[:8]
    ]
    cache_payload = {
        "age_group": age_text,
        "gender": gender_text,
        "cluster_label": cluster_text,
        "journey_stage": journey_text,
        "purchase_items": purchase_preview,
    }
    cached = _cache_get("customer_persona", cache_payload)
    if cached:
        return cached

    model = _get_gemini_model()
    if model is not None:
        prompt = f"""
You are a professional marketing analyst.
Summarize the following customer into JSON with keys:
- persona
- interest_keywords
- purchase_motive

Input:
- age_group: {age_text}
- gender: {gender_text}
- behavior_cluster_label: {cluster_text}
- journey_stage: {journey_text}
- purchase_items: {json.dumps(purchase_preview, ensure_ascii=False)}

Rules:
- persona: exactly one concise sentence
- interest_keywords: 3 to 5 short hashtag-style keywords in one comma-separated string
- purchase_motive: one concise phrase
- return JSON only
""".strip()
        try:
            response = model.generate_content(prompt, generation_config={"temperature": 0})
            parsed = _extract_json_block(getattr(response, "text", "") or "")
            if parsed:
                return _cache_set(
                    "customer_persona",
                    cache_payload,
                    {
                    "persona": str(parsed.get("persona", "")).strip(),
                    "interest_keywords": str(parsed.get("interest_keywords", "")).strip(),
                    "purchase_motive": str(parsed.get("purchase_motive", "")).strip(),
                    },
                )
        except Exception:
            if _paper_repro():
                raise

    categories = [category for _, category in purchase_items if category]
    product_names = [name for name, _ in purchase_items if name]
    category_terms = _top_terms(categories, limit=3)
    product_terms = _top_terms(product_names, limit=3)
    keyword_terms = _normalize_keyword_list(category_terms + product_terms, limit=4)

    motive_map = {
        "VIP Explorer": "novelty, quality, and breadth of choice",
        "Low-Value Loyal": "repeat convenience and dependable value",
        "At-Risk High-Value": "premium preference with declining recent engagement",
        "New Explorer": "discovery and onboarding curiosity",
        "Dormant": "reactivation triggers and event-driven demand",
        "Regular": "stable, routine everyday needs",
        "Unclassified": "limited observed behavior",
    }
    motif = motive_map.get(cluster_text, "balanced everyday demand")
    focus = ", ".join(keyword_terms[:2]) if keyword_terms else "broad catalog exploration"
    persona = (
        f"{age_text} {gender_text.lower()} customer in the {cluster_text.lower()} segment "
        f"who gravitates toward {focus} and is currently in {journey_text.lower()} stage."
    )
    keywords = ", ".join(f"#{term.replace(' ', '')}" for term in keyword_terms) or "#general"
    return _cache_set(
        "customer_persona",
        cache_payload,
        {
        "persona": persona,
        "interest_keywords": keywords,
        "purchase_motive": motif,
        },
    )


def summarize_customer_personas_batch(
    records: list[dict[str, object]],
    batch_size: int = 20,
) -> dict[str, dict[str, str]]:
    resolved: dict[str, dict[str, str]] = {}
    pending: list[dict[str, object]] = []

    for record in records:
        uid = str(record["user_id"])
        age_text = str(record.get("age_group", "unknown")).strip() or "unknown"
        gender_text = str(record.get("gender", "Unknown")).strip() or "Unknown"
        cluster_text = str(record.get("cluster_label", "Regular")).strip() or "Regular"
        journey_text = str(record.get("journey_stage", "Unknown")).strip() or "Unknown"
        purchase_preview = [
            {"product_name": name, "category": category}
            for name, category in list(record.get("purchase_items", []))[:8]
        ]
        cache_payload = {
            "age_group": age_text,
            "gender": gender_text,
            "cluster_label": cluster_text,
            "journey_stage": journey_text,
            "purchase_items": purchase_preview,
        }
        cached = _cache_get("customer_persona", cache_payload)
        if cached:
            resolved[uid] = cached
            continue
        pending.append(
            {
                "user_id": uid,
                "age_group": age_text,
                "gender": gender_text,
                "cluster_label": cluster_text,
                "journey_stage": journey_text,
                "purchase_items": purchase_preview,
                "cache_payload": cache_payload,
            }
        )

    if not pending:
        return resolved

    model = _get_gemini_model()
    if model is None:
        for record in pending:
            resolved[record["user_id"]] = summarize_customer_persona(
                age_group=str(record["age_group"]),
                gender=str(record["gender"]),
                cluster_label=str(record["cluster_label"]),
                purchase_items=[
                    (str(item["product_name"]), str(item["category"]))
                    for item in record["purchase_items"]
                ],
                journey_stage=str(record["journey_stage"]),
            )
        return resolved

    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        prompt_records = [
            {
                "user_id": record["user_id"],
                "age_group": record["age_group"],
                "gender": record["gender"],
                "behavior_cluster_label": record["cluster_label"],
                "journey_stage": record["journey_stage"],
                "purchase_items": record["purchase_items"],
            }
            for record in chunk
        ]
        prompt = f"""
You are a professional marketing analyst.
For each input customer, return one JSON array element with keys:
- user_id
- persona
- interest_keywords
- purchase_motive

Rules:
- Preserve input user_id exactly.
- persona: exactly one concise sentence
- interest_keywords: 3 to 5 short hashtag-style keywords in one comma-separated string
- purchase_motive: one concise phrase
- return JSON array only

Input:
{json.dumps(prompt_records, ensure_ascii=False)}
""".strip()
        try:
            response = model.generate_content(prompt, generation_config={"temperature": 0})
            parsed = _extract_json_array(getattr(response, "text", "") or "")
            if not parsed:
                raise RuntimeError("Batch Gemini response did not contain a JSON array.")
            parsed_map = {str(item.get("user_id", "")).strip(): item for item in parsed}
            missing = [record["user_id"] for record in chunk if record["user_id"] not in parsed_map]
            if missing:
                raise RuntimeError(f"Batch Gemini response missed user_ids: {missing[:5]}")
            for record in chunk:
                item = parsed_map[record["user_id"]]
                resolved[record["user_id"]] = _cache_set(
                    "customer_persona",
                    record["cache_payload"],
                    {
                        "persona": str(item.get("persona", "")).strip(),
                        "interest_keywords": str(item.get("interest_keywords", "")).strip(),
                        "purchase_motive": str(item.get("purchase_motive", "")).strip(),
                    },
                )
        except Exception:
            if _paper_repro():
                raise
            for record in chunk:
                resolved[record["user_id"]] = summarize_customer_persona(
                    age_group=str(record["age_group"]),
                    gender=str(record["gender"]),
                    cluster_label=str(record["cluster_label"]),
                    purchase_items=[
                        (str(item["product_name"]), str(item["category"]))
                        for item in record["purchase_items"]
                    ],
                    journey_stage=str(record["journey_stage"]),
                )

    return resolved


def summarize_product_attribute_tags(
    *,
    cluster_id: int,
    keywords: list[str],
    sample_products: list[str],
) -> dict[str, str]:
    """Return cluster-level attribute tags and summary text."""

    clean_keywords = _normalize_keyword_list(keywords, limit=7)
    sample_products = [str(p).strip() for p in sample_products if str(p).strip()][:8]
    cache_payload = {
        "cluster_id": int(cluster_id),
        "keywords": clean_keywords,
        "sample_products": sample_products,
    }
    cached = _cache_get("product_attribute_tags", cache_payload)
    if cached:
        return cached

    model = _get_gemini_model()
    if model is not None:
        prompt = f"""
You are a product taxonomy analyst.
Generate JSON with keys:
- attribute_tags
- cluster_summary

Input:
- cluster_id: {cluster_id}
- keywords: {json.dumps(clean_keywords, ensure_ascii=False)}
- sample_products: {json.dumps(sample_products, ensure_ascii=False)}

Rules:
- attribute_tags: comma-separated 3 to 6 short tags
- cluster_summary: one concise sentence
- return JSON only
""".strip()
        try:
            response = model.generate_content(prompt, generation_config={"temperature": 0})
            parsed = _extract_json_block(getattr(response, "text", "") or "")
            if parsed:
                return _cache_set(
                    "product_attribute_tags",
                    cache_payload,
                    {
                    "attribute_tags": str(parsed.get("attribute_tags", "")).strip(),
                    "cluster_summary": str(parsed.get("cluster_summary", "")).strip(),
                    },
                )
        except Exception:
            if _paper_repro():
                raise

    tags = clean_keywords[:5]
    if not tags:
        tags = _top_terms(sample_products, limit=4)
    tag_text = ", ".join(tags) if tags else "misc"
    summary = (
        f"Cluster {cluster_id} groups products centered on {', '.join(tags[:3])}."
        if tags
        else f"Cluster {cluster_id} groups mixed long-tail catalog items."
    )
    return _cache_set(
        "product_attribute_tags",
        cache_payload,
        {
        "attribute_tags": tag_text,
        "cluster_summary": summary,
        },
    )
