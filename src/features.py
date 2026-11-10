"""Feature engineering + target construction shared by the offline pipeline and the Streamlit app.

Keeping this in one module guarantees that a property entered in the app is transformed
exactly like the training data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (AMENITY_LIST, BASE_GROWTH_RATE, CURRENT_YEAR, HORIZON_YEARS,
                    TRANSPORT_MAP)

TYPE_GROWTH_ADJ = {"Apartment": 0.000, "Independent House": 0.0025, "Villa": 0.005}


# --------------------------------------------------------------------------- helpers
def count_amenities(value) -> int:
    if not isinstance(value, str) or not value.strip():
        return 0
    items = {a.strip().lower() for a in value.split(",") if a.strip()}
    return len(items)


def yes_no(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["yes", "y", "true", "1"]).astype(int)


# --------------------------------------------------------------------------- lookups
def build_lookups(df: pd.DataFrame) -> dict:
    """City-level reference statistics learned from the full cleaned dataset."""
    ppsf = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]
    city_ppsf = ppsf.groupby(df["City"]).median()
    city_price = df.groupby("City")["Price_in_Lakhs"].median()

    # "Developing" (cheaper) cities are assumed to appreciate faster:
    # cheapest city -> 10% p.a., most expensive -> 6% p.a. (centred on the 8% baseline).
    rank = city_ppsf.rank(pct=True)
    city_growth = (BASE_GROWTH_RATE + 0.02) - 0.04 * rank

    transport = df["Public_Transport_Accessibility"].map(TRANSPORT_MAP).fillna(2)
    infra_raw = (df["Nearby_Schools"] / df["Nearby_Schools"].max()
                 + df["Nearby_Hospitals"] / df["Nearby_Hospitals"].max()
                 + transport / 3) / 3

    return {
        "city_median_ppsf": city_ppsf.round(2).to_dict(),
        "city_median_price": city_price.round(2).to_dict(),
        "city_growth_rate": city_growth.round(5).to_dict(),
        "schools_max": float(df["Nearby_Schools"].max()),
        "hospitals_max": float(df["Nearby_Hospitals"].max()),
        "infra_median": float(infra_raw.median()),
        "global_median_ppsf": float(ppsf.median()),
        "global_median_price": float(df["Price_in_Lakhs"].median()),
        "city_state": df.groupby("City")["State"].first().to_dict(),
        "city_localities": {c: sorted(g.unique().tolist())
                            for c, g in df.groupby("City")["Locality"]},
        "categories": {c: sorted(df[c].dropna().unique().tolist()) for c in
                       ["State", "City", "Property_Type", "Furnished_Status", "Facing",
                        "Owner_Type", "Availability_Status", "Public_Transport_Accessibility"]},
        "amenities": AMENITY_LIST,
        "current_year": CURRENT_YEAR,
    }


# --------------------------------------------------------------------------- features
def engineer_features(df: pd.DataFrame, lk: dict) -> pd.DataFrame:
    out = df.copy()
    # Dataset's Price_per_SqFt is lakhs/sqft rounded to 2 dp (mostly 0.0x) -> recompute in Rupees.
    out["Price_per_SqFt"] = (out["Price_in_Lakhs"] * 1e5 / out["Size_in_SqFt"]).round(2)
    out["Age_of_Property"] = lk["current_year"] - out["Year_Built"]

    out["Transport_Score"] = out["Public_Transport_Accessibility"].map(TRANSPORT_MAP).fillna(2).astype(int)
    out["Parking"] = yes_no(out["Parking_Space"])
    out["Has_Security"] = yes_no(out["Security"])
    out["Amenity_Count"] = out["Amenities"].apply(count_amenities)
    out["Amenity_Density_Score"] = (out["Amenity_Count"] / out["Size_in_SqFt"] * 1000).round(4)
    out["Infrastructure_Score"] = ((out["Nearby_Schools"] / lk["schools_max"]
                                    + out["Nearby_Hospitals"] / lk["hospitals_max"]
                                    + out["Transport_Score"] / 3) / 3).round(4)
    out["Floor_Ratio"] = (out["Floor_No"] / out["Total_Floors"].clip(lower=1)).clip(upper=1).round(4)

    city_ppsf = out["City"].map(lk["city_median_ppsf"]).fillna(lk["global_median_ppsf"])
    city_price = out["City"].map(lk["city_median_price"]).fillna(lk["global_median_price"])
    out["PPSF_vs_City_Median"] = (out["Price_per_SqFt"] / city_ppsf).round(4)
    out["Price_vs_City_Median"] = (out["Price_in_Lakhs"] / city_price).round(4)
    out["City_Growth_Rate"] = out["City"].map(lk["city_growth_rate"]).fillna(BASE_GROWTH_RATE)
    return out


# --------------------------------------------------------------------------- targets
def expected_growth_rate(df: pd.DataFrame, rng: np.random.Generator | None = None) -> pd.Series:
    """Feature-based annual appreciation rate (brief: 'By Location/Type' + 'Feature-based').

    r = city rate + property-type premium + infrastructure premium + amenity premium
        + under-construction premium - ageing penalty (+ small market noise when training).
    """
    r = (df["City_Growth_Rate"]
         + df["Property_Type"].map(TYPE_GROWTH_ADJ).fillna(0)
         + 0.02 * (df["Infrastructure_Score"] - 0.5)
         + 0.001 * df["Amenity_Count"]
         + 0.002 * df["Has_Security"]
         + np.where(df["Availability_Status"] == "Under_Construction", 0.01, 0.0)
         - 0.0004 * df["Age_of_Property"].clip(lower=0))
    if rng is not None:
        r = r + rng.normal(0, 0.003, len(df))
    return r.clip(0.02, 0.18)


def future_price(price_lakhs, rate, years: int = HORIZON_YEARS):
    return price_lakhs * (1 + rate) ** years


def investment_score(df: pd.DataFrame, lk: dict) -> pd.DataFrame:
    """Multi-factor rule (brief: Price vs median, PPSF vs median, BHK>=3, ready-to-move, ...).

    Returns one 0/1 column per criterion plus the total score.
    """
    crit = pd.DataFrame(index=df.index)
    crit["Price_per_SqFt <= city median"] = (df["PPSF_vs_City_Median"] <= 1).astype(int)
    crit["Price <= city median"] = (df["Price_vs_City_Median"] <= 1).astype(int)
    crit["BHK >= 3"] = (df["BHK"] >= 3).astype(int)
    crit["Ready to move"] = (df["Availability_Status"] == "Ready_to_Move").astype(int)
    crit["Infrastructure above median"] = (df["Infrastructure_Score"] >= lk["infra_median"]).astype(int)
    crit["4+ amenities"] = (df["Amenity_Count"] >= 4).astype(int)
    crit["Score"] = crit.sum(axis=1)
    return crit


GOOD_INVESTMENT_THRESHOLD = 4   # score >= 4 of 6 criteria -> Good Investment
