# -*- coding: utf-8 -*-
"""Phase 1: Data preprocessing and augmentation."""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from common_utils import (
    normalize_customer_id,
    normalize_interaction_ids,
    normalize_product_id,
    parse_timestamp,
)


def _fill_missing_customers(customers: pd.DataFrame) -> pd.DataFrame:
    out = customers.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].fillna(out[col].median())
    for col in out.select_dtypes(include=["object"]).columns:
        mode_vals = out[col].mode()
        out[col] = out[col].fillna(mode_vals.iloc[0] if len(mode_vals) else "Unknown")
    return out


def _fill_missing_products(products: pd.DataFrame) -> pd.DataFrame:
    out = products.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].fillna(out[col].median())
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].fillna("")
    return out


def _to_numeric_money(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def augment_with_gaussian_noise(
    df: pd.DataFrame,
    numeric_cols: list[str],
    noise_factor: float = 0.05,
    n_copies: int = 2,
) -> pd.DataFrame:
    frames = [df.copy()]
    for _ in range(n_copies):
        copy_df = df.copy()
        for col in numeric_cols:
            base = pd.to_numeric(copy_df[col], errors="coerce")
            std = float(base.std()) if base.notna().any() else 0.0
            if std > 0:
                noise = np.random.normal(0, std * noise_factor, size=len(copy_df))
                copy_df[col] = base + noise
        frames.append(copy_df)
    return pd.concat(frames, ignore_index=True)


def build_interaction_augmentation(
    interactions: pd.DataFrame,
    interaction_mapping: dict[str, int],
    n_copies: int = 2,
) -> pd.DataFrame:
    frames = [interactions.copy()]
    inv_map = {v: k for k, v in interaction_mapping.items()}

    for i in range(n_copies):
        cp = interactions.copy()
        jitter = np.random.normal(0, 0.15, size=len(cp))
        cp["interaction_type_encoded"] = (
            cp["interaction_type_encoded"].astype(float) + jitter
        ).round().astype(int).clip(0, len(interaction_mapping) - 1)
        cp["interaction_type"] = cp["interaction_type_encoded"].map(inv_map)

        # Keep IDs unchanged. Only nudge timestamps to create sequence.
        cp["timestamp_parsed"] = cp["timestamp_parsed"] + pd.to_timedelta(i + 1, unit="m")
        cp["timestamp"] = cp["timestamp_parsed"].dt.strftime("%Y-%m-%d %H:%M:%S")
        frames.append(cp)

    out = pd.concat(frames, ignore_index=True)
    return normalize_interaction_ids(out)


def main() -> None:
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    fast_mode = os.getenv("FAST_MODE", "0") == "1"
    default_aug_copies = "1" if fast_mode else "2"
    aug_copies = int(os.getenv("AUG_COPIES", default_aug_copies))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Phase 1: Data Preprocessing and Augmentation")
    print("=" * 60)
    print(f"  Fast mode: {fast_mode}")

    print("\n[1.1] Loading data...")
    customers = pd.read_csv(os.path.join(data_dir, "customer_details.csv"), low_memory=False)
    products = pd.read_csv(os.path.join(data_dir, "product_details.csv"), low_memory=False)
    interactions = pd.read_csv(
        os.path.join(data_dir, "E-commerece sales data 2024.csv"),
        low_memory=False,
    )

    customers.columns = customers.columns.str.strip()
    products.columns = products.columns.str.strip()
    interactions.columns = interactions.columns.str.strip()
    for df in (customers, products, interactions):
        unnamed_cols = [c for c in df.columns if "Unnamed" in str(c)]
        if unnamed_cols:
            df.drop(columns=unnamed_cols, inplace=True)

    interactions = interactions.rename(
        columns={
            "user id": "user_id",
            "product id": "product_id",
            "Interaction type": "interaction_type",
            "Time stamp": "timestamp",
        }
    )

    if "Customer ID" in customers.columns:
        customers["Customer ID"] = customers["Customer ID"].map(normalize_customer_id)
    if "Uniqe Id" in products.columns:
        products["Uniqe Id"] = products["Uniqe Id"].map(normalize_product_id)
    interactions = normalize_interaction_ids(interactions)

    print(f"  Customers:    {customers.shape[0]:,} rows x {customers.shape[1]} cols")
    print(f"  Products:     {products.shape[0]:,} rows x {products.shape[1]} cols")
    print(f"  Interactions: {interactions.shape[0]:,} rows x {interactions.shape[1]} cols")

    print("\n[1.2] Handling missing values...")
    customers = _fill_missing_customers(customers)
    products = _fill_missing_products(products)

    interactions["user_id"] = interactions["user_id"].replace("", np.nan).ffill().bfill()
    interactions["timestamp"] = interactions["timestamp"].ffill().bfill()
    interactions["product_id"] = interactions["product_id"].replace("", np.nan).fillna("unknown_product")
    mode_inter = interactions["interaction_type"].mode()
    fill_inter = mode_inter.iloc[0] if len(mode_inter) else "view"
    interactions["interaction_type"] = (
        interactions["interaction_type"].fillna(fill_inter).astype(str).str.lower().str.strip()
    )

    print("\n[1.3] Removing duplicates and clipping outliers...")
    customers = customers.drop_duplicates()
    products = products.drop_duplicates()
    interactions = interactions.drop_duplicates()

    numeric_cols_cust = customers.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_cust = [c for c in numeric_cols_cust if "id" not in c.lower()]
    for col in numeric_cols_cust:
        q01, q99 = customers[col].quantile([0.01, 0.99])
        customers[col] = customers[col].clip(q01, q99)

    print("\n[1.4] Encoding categorical variables...")
    inter_encoder = LabelEncoder()
    interactions["interaction_type_encoded"] = inter_encoder.fit_transform(interactions["interaction_type"])
    interaction_mapping = dict(
        zip(inter_encoder.classes_, inter_encoder.transform(inter_encoder.classes_))
    )
    print(f"  Interaction type mapping: {interaction_mapping}")

    cat_cols_cust = [
        "Gender",
        "Category",
        "Size",
        "Color",
        "Season",
        "Subscription Status",
        "Shipping Type",
        "Discount Applied",
        "Promo Code Used",
        "Payment Method",
        "Frequency of Purchases",
    ]
    encoded_count = 0
    for col in cat_cols_cust:
        if col in customers.columns:
            le = LabelEncoder()
            customers[f"{col}_encoded"] = le.fit_transform(customers[col].astype(str))
            encoded_count += 1
    print(f"  Encoded customer categorical columns: {encoded_count}")

    print("\n[1.5] Standardizing numeric features...")
    numeric_for_scaling = [
        col
        for col in ["Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases"]
        if col in customers.columns
    ]
    if numeric_for_scaling:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(customers[numeric_for_scaling])
        scaled_mean = pd.DataFrame(scaled, columns=numeric_for_scaling).mean().round(6).to_dict()
        print(f"  Standardized columns: {numeric_for_scaling}")
        print(f"  Mean after scaling: {scaled_mean}")
    else:
        print("  No numeric columns found for scaling.")

    print("\n[1.6] Engineering time features...")
    interactions["timestamp_parsed"] = parse_timestamp(interactions["timestamp"])
    interactions["timestamp_parsed"] = interactions["timestamp_parsed"].ffill().bfill()
    interactions["timestamp_parsed"] = interactions["timestamp_parsed"].fillna(pd.Timestamp("2024-01-01"))
    interactions["year"] = interactions["timestamp_parsed"].dt.year
    interactions["month"] = interactions["timestamp_parsed"].dt.month
    interactions["day"] = interactions["timestamp_parsed"].dt.day
    interactions["day_of_week"] = interactions["timestamp_parsed"].dt.dayofweek
    interactions["hour"] = interactions["timestamp_parsed"].dt.hour
    interactions["timestamp"] = interactions["timestamp_parsed"].dt.strftime("%Y-%m-%d %H:%M:%S")

    print(
        "  Date range:",
        interactions["timestamp_parsed"].min(),
        "to",
        interactions["timestamp_parsed"].max(),
    )

    print("\n[1.7] Saving processed data...")
    customers.to_csv(os.path.join(output_dir, "customers_processed.csv"), index=False)
    products.to_csv(os.path.join(output_dir, "products_processed.csv"), index=False)
    interactions.to_csv(os.path.join(output_dir, "interactions_processed.csv"), index=False)

    print("\n[1.8] Augmenting data...")
    print(f"  Augmentation copies: {aug_copies}")
    aug_numeric_cust = [
        col
        for col in ["Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases"]
        if col in customers.columns
    ]
    customers_aug = augment_with_gaussian_noise(
        customers,
        aug_numeric_cust,
        noise_factor=0.05,
        n_copies=aug_copies,
    )

    interactions_aug = build_interaction_augmentation(
        interactions,
        interaction_mapping=interaction_mapping,
        n_copies=aug_copies,
    )

    products_aug = products.copy()
    for col in ["Selling Price", "Quantity"]:
        if col in products_aug.columns:
            products_aug[col] = _to_numeric_money(products_aug[col])
    aug_numeric_prod = [c for c in ["Selling Price", "Quantity"] if c in products_aug.columns]
    products_aug = augment_with_gaussian_noise(
        products_aug,
        aug_numeric_prod,
        noise_factor=0.05,
        n_copies=aug_copies,
    )
    if "Uniqe Id" in products_aug.columns:
        products_aug["Uniqe Id"] = products_aug["Uniqe Id"].map(normalize_product_id)

    print(f"  Customers:    {len(customers):,} -> {len(customers_aug):,}")
    print(f"  Interactions: {len(interactions):,} -> {len(interactions_aug):,}")
    print(f"  Products:     {len(products):,} -> {len(products_aug):,}")

    print("\n[1.9] Saving augmented data...")
    customers_aug.to_csv(os.path.join(output_dir, "combined_customers.csv"), index=False)
    interactions_aug.to_csv(os.path.join(output_dir, "combined_interactions.csv"), index=False)
    products_aug.to_csv(os.path.join(output_dir, "combined_products.csv"), index=False)

    print("\n" + "=" * 60)
    print("Phase 1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
