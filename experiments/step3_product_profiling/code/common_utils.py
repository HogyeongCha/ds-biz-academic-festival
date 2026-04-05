# -*- coding: utf-8 -*-
"""Shared helpers for ID normalization and timestamp parsing."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


_SUFFIX_RE = re.compile(r"_[0-9]+$")


def normalize_user_id(value: Any) -> str:
    """Normalize user id like 2.0 -> '2' and preserve non-numeric IDs."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
        return str(num)
    except ValueError:
        return text


def normalize_customer_id(value: Any) -> str:
    """Customer ID uses the same normalization rules as user ID."""
    return normalize_user_id(value)


def normalize_product_id(value: Any, strip_legacy_suffix: bool = True) -> str:
    """
    Normalize product ids.

    Legacy outputs may contain suffixes like '<id>_1234'. Strip by default.
    """
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if strip_legacy_suffix:
        text = _SUFFIX_RE.sub("", text)
    return text


def parse_timestamp(series: pd.Series) -> pd.Series:
    """Parse mixed timestamp formats across pandas versions."""
    try:
        return pd.to_datetime(series, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce")


def normalize_interaction_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized user/product ids for interaction tables."""
    out = df.copy()
    if "user_id" in out.columns:
        out["user_id"] = out["user_id"].map(normalize_user_id)
    if "product_id" in out.columns:
        out["product_id"] = out["product_id"].map(normalize_product_id)
    return out
