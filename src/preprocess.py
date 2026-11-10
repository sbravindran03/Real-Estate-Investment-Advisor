"""Step 1 - Data preprocessing.

* cleans the raw CSV (types, whitespace, duplicates, missing values)
* flags outliers with the IQR rule
* engineers features (Price per SqFt, Amenity Density Score, Infrastructure Score, ...)
* builds both targets: Future_Price_5Y (regression) and Good_Investment (classification)
* writes data/processed/cleaned_housing.csv and data/processed/lookups.json

Run:  python src/preprocess.py
"""
import json

import numpy as np
import pandas as pd

from config import CLF_TARGET, LOOKUPS_JSON, PROCESSED_CSV, RANDOM_STATE, RAW_CSV, REG_TARGET
from features import (GOOD_INVESTMENT_THRESHOLD, build_lookups, engineer_features,
                      expected_growth_rate, future_price, investment_score)

NUMERIC_RAW = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built", "Floor_No",
               "Total_Floors", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    report = {"rows_in": len(df)}
    for c in df.select_dtypes("object"):
        df[c] = df[c].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
    for c in NUMERIC_RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # duplicates (ignoring the surrogate ID)
    dup_mask = df.drop(columns="ID").duplicated()
    report["duplicates_removed"] = int(dup_mask.sum())
    df = df[~dup_mask].copy()

    # missing values: median for numeric, mode for categorical (Amenities -> empty)
    report["missing_before"] = {k: int(v) for k, v in df.isna().sum().items() if v}
    for c in NUMERIC_RAW:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    df["Amenities"] = df["Amenities"].fillna("")
    for c in df.select_dtypes("object"):
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])

    # invalid values
    invalid = (df["Size_in_SqFt"] <= 0) | (df["Price_in_Lakhs"] <= 0) | (df["BHK"] <= 0)
    report["invalid_removed"] = int(invalid.sum())
    df = df[~invalid].copy()
    report["rows_out"] = len(df)
    print("Cleaning report:", json.dumps(report, indent=2))
    return df


def flag_outliers(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flag = (df[c] < lo) | (df[c] > hi)
        df[f"Outlier_{c}"] = flag.astype(int)
        print(f"Outliers in {c}: {int(flag.sum())} ({flag.mean():.2%})  bounds=[{lo:.2f}, {hi:.2f}]")
    return df


def main():
    df = pd.read_csv(RAW_CSV)
    print("Raw shape:", df.shape)
    df = clean(df)

    lookups = build_lookups(df)
    df = engineer_features(df, lookups)
    df = flag_outliers(df, ["Price_per_SqFt", "Size_in_SqFt", "Price_in_Lakhs"])

    # ---- regression target
    rng = np.random.default_rng(RANDOM_STATE)
    df["Expected_Growth_Rate"] = expected_growth_rate(df, rng).round(5)
    df[REG_TARGET] = future_price(df["Price_in_Lakhs"], df["Expected_Growth_Rate"]).round(2)

    # ---- classification target
    crit = investment_score(df, lookups)
    df["Investment_Score"] = crit["Score"]
    df[CLF_TARGET] = (crit["Score"] >= GOOD_INVESTMENT_THRESHOLD).astype(int)
    print(f"Good_Investment rate: {df[CLF_TARGET].mean():.2%}")
    print(f"Mean expected growth: {df['Expected_Growth_Rate'].mean():.2%}")

    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)
    with open(LOOKUPS_JSON, "w") as f:
        json.dump(lookups, f, indent=1)
    print("Saved", PROCESSED_CSV, df.shape)


if __name__ == "__main__":
    main()
