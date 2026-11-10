"""Step 5 - Streamlit app: Real Estate Investment Advisor.

Run:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import LOOKUPS_JSON, MODELS_DIR, PROCESSED_CSV  # noqa: E402
from features import (GOOD_INVESTMENT_THRESHOLD, engineer_features,  # noqa: E402
                      investment_score)

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")


# --------------------------------------------------------------------------- loaders
@st.cache_resource
def load_models():
    clf = joblib.load(MODELS_DIR / "classifier.joblib")
    reg = joblib.load(MODELS_DIR / "regressor.joblib")
    meta = json.loads((MODELS_DIR / "model_metadata.json").read_text())
    return clf, reg, meta


@st.cache_data
def load_data():
    cols = ["ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
            "Price_per_SqFt", "Year_Built", "Age_of_Property", "Furnished_Status", "Amenity_Count",
            "Infrastructure_Score", "Public_Transport_Accessibility", "Availability_Status",
            "Expected_Growth_Rate", "Future_Price_5Y", "Good_Investment", "Investment_Score"]
    return pd.read_csv(PROCESSED_CSV, usecols=cols)


@st.cache_data
def load_lookups():
    return json.loads(Path(LOOKUPS_JSON).read_text())


missing = [p for p in [PROCESSED_CSV, LOOKUPS_JSON, MODELS_DIR / "classifier.joblib"] if not Path(p).exists()]
if missing:
    st.error("Artifacts not found. Run `python src/preprocess.py` and `python src/train.py` first.\n\n"
             + "\n".join(f"- {m}" for m in missing))
    st.stop()

clf, reg, meta = load_models()
data = load_data()
lk = load_lookups()
cats = lk["categories"]

st.title("🏠 Real Estate Investment Advisor")
st.caption("Classifies whether a property is a **Good Investment** and forecasts its **price after 5 years**. "
           f"Models: {meta['classification']['best_model']} (classifier) · "
           f"{meta['regression']['best_model']} (regressor), tracked in MLflow.")

tab_pred, tab_explore, tab_insight, tab_model = st.tabs(
    ["🔮 Predict", "🔎 Explore listings", "📊 Market insights", "🧠 Model performance"])

# =========================================================================== PREDICT
with tab_pred:
    with st.form("property"):
        st.subheader("Property details")
        c1, c2, c3, c4 = st.columns(4)
        city = c1.selectbox("City", cats["City"], index=cats["City"].index("Mumbai") if "Mumbai" in cats["City"] else 0)
        locality = c2.selectbox("Locality", lk["city_localities"].get(city, ["—"]))
        ptype = c3.selectbox("Property type", cats["Property_Type"])
        avail = c4.selectbox("Availability", cats["Availability_Status"])

        c1, c2, c3, c4 = st.columns(4)
        bhk = c1.number_input("BHK", 1, 10, 3)
        size = c2.number_input("Size (sq ft)", 200, 10000, 1800, step=50)
        price = c3.number_input("Current price (₹ Lakhs)", 5.0, 1000.0, 150.0, step=5.0)
        year = c4.number_input("Year built", 1950, lk["current_year"] + 3, 2015)

        c1, c2, c3, c4 = st.columns(4)
        furnished = c1.selectbox("Furnished status", cats["Furnished_Status"])
        floor = c2.number_input("Floor no.", 0, 100, 5)
        total_floors = c3.number_input("Total floors", 1, 100, 12)
        facing = c4.selectbox("Facing", cats["Facing"])

        c1, c2, c3, c4 = st.columns(4)
        schools = c1.slider("Nearby schools", 1, int(lk["schools_max"]), 5)
        hospitals = c2.slider("Nearby hospitals", 1, int(lk["hospitals_max"]), 5)
        transport = c3.select_slider("Public transport", ["Low", "Medium", "High"], "Medium")
        owner = c4.selectbox("Owner type", cats["Owner_Type"])

        c1, c2, c3 = st.columns([1, 1, 2])
        parking = c1.radio("Parking", ["Yes", "No"], horizontal=True)
        security = c2.radio("Security", ["Yes", "No"], horizontal=True)
        amenities = c3.multiselect("Amenities", lk["amenities"], default=["Gym", "Garden"])
        submitted = st.form_submit_button("Analyse property", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "State": lk["city_state"][city], "City": city, "Locality": locality, "Property_Type": ptype,
            "BHK": bhk, "Size_in_SqFt": size, "Price_in_Lakhs": price, "Year_Built": year,
            "Furnished_Status": furnished, "Floor_No": floor, "Total_Floors": total_floors,
            "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
            "Public_Transport_Accessibility": transport, "Parking_Space": parking, "Security": security,
            "Amenities": ", ".join(amenities), "Facing": facing, "Owner_Type": owner,
            "Availability_Status": avail,
        }])
        feat = engineer_features(row, lk)
        X = feat[meta["features"]]
        p_good = float(clf.predict_proba(X)[0, 1])
        fut = float(reg.predict(X)[0])
        cagr = (fut / price) ** (1 / 5) - 1
        rmse = meta["regression"]["test_rmse"]

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        if p_good >= 0.5:
            m1.success(f"### ✅ Good Investment\nConfidence **{p_good:.2%}**")
        else:
            m1.error(f"### ❌ Not a Good Investment\nConfidence **{1 - p_good:.2%}**")
        m2.metric("Estimated price after 5 years", f"₹{fut:,.1f} L", f"+₹{fut - price:,.1f} L")
        m3.metric("Implied annual growth (CAGR)", f"{cagr:.2%}")
        m4.metric("Price per sq ft", f"₹{feat['Price_per_SqFt'].iloc[0]:,.0f}",
                  f"{feat['PPSF_vs_City_Median'].iloc[0] - 1:+.1%} vs {city} median", delta_color="inverse")
        st.caption(f"Forecast range (±1 test RMSE): ₹{fut - rmse:,.1f} L – ₹{fut + rmse:,.1f} L")

        g1, g2 = st.columns([3, 2])
        years = list(range(0, 6))
        proj = [price * (1 + cagr) ** y for y in years]
        fig = go.Figure(go.Scatter(x=[lk["current_year"] + y for y in years], y=proj, mode="lines+markers",
                                   line=dict(width=3), fill="tozeroy"))
        fig.update_layout(title="Projected value (₹ Lakhs)", height=320, margin=dict(t=40, b=10))
        g1.plotly_chart(fig, use_container_width=True)

        crit = investment_score(feat, lk).iloc[0]
        g2.markdown(f"**Investment checklist** — {int(crit['Score'])}/6 "
                    f"(rule: ≥{GOOD_INVESTMENT_THRESHOLD} → good)")
        for k, v in crit.drop("Score").items():
            g2.markdown(f"{'🟢' if v else '⚪'} {k}")

        imp = pd.Series(meta["classification"]["feature_importance"]).head(10)[::-1]
        fig = px.bar(imp, orientation="h", title="What drives the investment decision (top 10 features)",
                     labels={"value": "Relative importance", "index": ""})
        fig.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================================== EXPLORE
with tab_explore:
    f1, f2, f3, f4 = st.columns(4)
    sel_city = f1.multiselect("City", cats["City"])
    sel_type = f2.multiselect("Property type", cats["Property_Type"])
    sel_bhk = f3.multiselect("BHK", sorted(data["BHK"].unique()))
    only_good = f4.checkbox("Good investments only")
    f5, f6 = st.columns(2)
    pr = f5.slider("Price (₹ Lakhs)", float(data["Price_in_Lakhs"].min()), float(data["Price_in_Lakhs"].max()),
                   (float(data["Price_in_Lakhs"].min()), float(data["Price_in_Lakhs"].max())))
    sz = f6.slider("Size (sq ft)", int(data["Size_in_SqFt"].min()), int(data["Size_in_SqFt"].max()),
                   (int(data["Size_in_SqFt"].min()), int(data["Size_in_SqFt"].max())))

    q = data[data["Price_in_Lakhs"].between(*pr) & data["Size_in_SqFt"].between(*sz)]
    if sel_city: q = q[q["City"].isin(sel_city)]
    if sel_type: q = q[q["Property_Type"].isin(sel_type)]
    if sel_bhk: q = q[q["BHK"].isin(sel_bhk)]
    if only_good: q = q[q["Good_Investment"] == 1]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Matching listings", f"{len(q):,}")
    if len(q):
        k2.metric("Median price", f"₹{q['Price_in_Lakhs'].median():,.1f} L")
        k3.metric("Median ₹/sq ft", f"₹{q['Price_per_SqFt'].median():,.0f}")
        k4.metric("Good-investment share", f"{q['Good_Investment'].mean():.1%}")
        show = q.sort_values(["Investment_Score", "Expected_Growth_Rate"], ascending=False).head(500)
        st.dataframe(show, use_container_width=True, hide_index=True,
                     column_config={"Expected_Growth_Rate": st.column_config.NumberColumn(format="%.3f"),
                                    "Good_Investment": st.column_config.CheckboxColumn()})
        st.download_button("Download filtered listings (CSV)", q.to_csv(index=False), "filtered_properties.csv")

# =========================================================================== INSIGHTS
with tab_insight:
    hm = data.pivot_table(index="City", columns="Property_Type", values="Price_per_SqFt", aggfunc="mean")
    fig = px.imshow(hm.round(0), aspect="auto", color_continuous_scale="Viridis", text_auto=True,
                    title="Location heatmap — average ₹/sq ft by city × property type", height=900)
    st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    city_stats = data.groupby("City").agg(Growth=("Expected_Growth_Rate", "mean"),
                                          Good=("Good_Investment", "mean"),
                                          PPSF=("Price_per_SqFt", "median")).reset_index()
    fig = px.scatter(city_stats, x="PPSF", y="Growth", size="Good", text="City",
                     title="Cities: median ₹/sq ft vs expected annual growth (bubble = % good investments)",
                     labels={"PPSF": "Median ₹/sq ft", "Growth": "Expected growth rate"})
    fig.update_traces(textposition="top center", textfont_size=9)
    a.plotly_chart(fig, use_container_width=True)

    st_stats = data.groupby("State")["Price_per_SqFt"].mean().sort_values().reset_index()
    fig = px.bar(st_stats, x="Price_per_SqFt", y="State", orientation="h", title="Average ₹/sq ft by state")
    b.plotly_chart(fig, use_container_width=True)

    c, d = st.columns(2)
    trend = data.groupby(["Year_Built", "Property_Type"])["Price_in_Lakhs"].mean().reset_index()
    fig = px.line(trend, x="Year_Built", y="Price_in_Lakhs", color="Property_Type",
                  title="Trend — average price by year built")
    c.plotly_chart(fig, use_container_width=True)
    bhk = data.groupby("BHK").agg(Price=("Price_in_Lakhs", "mean"), Good=("Good_Investment", "mean")).reset_index()
    fig = px.bar(bhk, x="BHK", y="Good", title="Good-investment share by BHK", labels={"Good": "Share"})
    d.plotly_chart(fig, use_container_width=True)

    e, f = st.columns(2)
    fig = px.histogram(data, x="Expected_Growth_Rate", color="Good_Investment", nbins=60, barmode="overlay",
                       title="Expected growth rate distribution by class")
    e.plotly_chart(fig, use_container_width=True)
    tr = data.groupby("Public_Transport_Accessibility")["Good_Investment"].mean().reindex(["Low", "Medium", "High"])
    fig = px.bar(tr, title="Good-investment share by public transport access", labels={"value": "Share"})
    fig.update_layout(showlegend=False)
    f.plotly_chart(fig, use_container_width=True)

# =========================================================================== MODELS
with tab_model:
    a, b = st.columns(2)
    a.subheader("Classification — Good_Investment")
    a.caption(f"Registered: `{meta['classification']['registry']}` (alias *champion*)")
    a.dataframe(pd.DataFrame(meta["classification"]["leaderboard"]).set_index("model").round(4)
                .sort_values("f1", ascending=False), use_container_width=True)
    cm = meta["classification"]["confusion_matrix"]
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=["Pred: Not good", "Pred: Good"], y=["Actual: Not good", "Actual: Good"],
                    title=f"Confusion matrix — {meta['classification']['best_model']}")
    a.plotly_chart(fig, use_container_width=True)
    imp = pd.Series(meta["classification"]["feature_importance"]).head(15)[::-1]
    a.plotly_chart(px.bar(imp, orientation="h", title="Feature importance (classifier)").update_layout(
        showlegend=False), use_container_width=True)

    b.subheader("Regression — Future_Price_5Y")
    b.caption(f"Registered: `{meta['regression']['registry']}` (alias *champion*)")
    b.dataframe(pd.DataFrame(meta["regression"]["leaderboard"]).set_index("model").round(4)
                .sort_values("rmse"), use_container_width=True)
    imp = pd.Series(meta["regression"]["feature_importance"]).head(15)[::-1]
    b.plotly_chart(px.bar(imp, orientation="h", title="Feature importance (regressor)").update_layout(
        showlegend=False), use_container_width=True)
    st.info("Full experiment history: run `mlflow ui --backend-store-uri sqlite:///mlflow.db` "
            "and open http://127.0.0.1:5000")
