"""Step 2 - Exploratory Data Analysis.

Answers the 20 EDA questions from the project brief (+ the investment-specific ones),
saving one chart per question to reports/eda/ and a written summary to reports/eda_findings.md.

Run (after preprocess.py):  python src/eda.py
"""
import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252 (no ₹ glyph)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import EDA_DIR, PROCESSED_CSV, RANDOM_STATE, REPORTS_DIR

sns.set_theme(style="whitegrid", palette="viridis")
FINDINGS: list[str] = []


def save(fig, name):
    fig.tight_layout()
    fig.savefig(EDA_DIR / f"{name}.png", dpi=110)
    plt.close(fig)


def note(q, text, img):
    FINDINGS.append(f"### {q}\n\n{text}\n\n![{img}](eda/{img}.png)\n")
    print(q, "->", text.splitlines()[0])


def main():
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_CSV)
    sample = df.sample(min(20000, len(df)), random_state=RANDOM_STATE)

    # ---------------- 1-5 Price & size
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["Price_in_Lakhs"], bins=60, kde=True, ax=ax)
    ax.set_title("Q1. Distribution of property prices (Lakhs)")
    save(fig, "q01_price_distribution")
    p = df["Price_in_Lakhs"]
    note("Q1. What is the distribution of property prices?",
         f"Prices range from ₹{p.min():.0f}L to ₹{p.max():.0f}L, mean ₹{p.mean():.1f}L, median ₹{p.median():.1f}L, "
         f"skew {p.skew():.2f} — close to uniform, so there is no long luxury tail.", "q01_price_distribution")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["Size_in_SqFt"], bins=60, kde=True, ax=ax, color="teal")
    ax.set_title("Q2. Distribution of property sizes (sq ft)")
    save(fig, "q02_size_distribution")
    s = df["Size_in_SqFt"]
    note("Q2. What is the distribution of property sizes?",
         f"Sizes range from {s.min():.0f} to {s.max():.0f} sq ft (median {s.median():.0f}, skew {s.skew():.2f}).",
         "q02_size_distribution")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt", ax=ax, showfliers=False)
    ax.set_title("Q3. Price per sq ft by property type (₹)")
    save(fig, "q03_ppsf_by_type")
    t = df.groupby("Property_Type")["Price_per_SqFt"].median().sort_values()
    note("Q3. How does the price per sq ft vary by property type?",
         "Median ₹/sq ft: " + ", ".join(f"{k} ₹{v:,.0f}" for k, v in t.items())
         + ". The differences are small — property type alone does not drive unit price.", "q03_ppsf_by_type")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=sample, x="Size_in_SqFt", y="Price_in_Lakhs", alpha=0.15, s=8, ax=ax)
    ax.set_title("Q4. Size vs price (20k sample)")
    save(fig, "q04_size_vs_price")
    r = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"])
    note("Q4. Is there a relationship between property size and price?",
         f"Pearson r = {r:.3f}: size and total price are essentially independent in this dataset, "
         "which is why price per sq ft varies so widely.", "q04_size_vs_price")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sns.boxplot(y=df["Price_per_SqFt"], ax=axes[0]); axes[0].set_title("Price per sq ft")
    sns.boxplot(y=df["Size_in_SqFt"], ax=axes[1], color="teal"); axes[1].set_title("Size in sq ft")
    fig.suptitle("Q5. Outliers (IQR rule)")
    save(fig, "q05_outliers")
    note("Q5. Are there any outliers in price per sq ft or property size?",
         f"IQR rule flags {df['Outlier_Price_per_SqFt'].sum():,} rows ({df['Outlier_Price_per_SqFt'].mean():.2%}) on "
         f"price per sq ft (small homes priced high) and {df['Outlier_Size_in_SqFt'].sum():,} on size. "
         "They are kept (they are real listings) but flagged; tree models are robust to them.", "q05_outliers")

    # ---------------- 6-10 Location
    st = df.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=st.values, y=st.index, ax=ax)
    ax.set_title("Q6. Average price per sq ft by state (₹)")
    save(fig, "q06_ppsf_by_state")
    note("Q6. What is the average price per sq ft by state?",
         f"Highest: {st.index[0]} (₹{st.iloc[0]:,.0f}); lowest: {st.index[-1]} (₹{st.iloc[-1]:,.0f}).", "q06_ppsf_by_state")

    ct = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.barplot(x=ct.values, y=ct.index, ax=ax)
    ax.set_title("Q7. Average property price by city (Lakhs)")
    save(fig, "q07_price_by_city")
    note("Q7. What is the average property price by city?",
         f"Most expensive: {ct.index[0]} (₹{ct.iloc[0]:.1f}L); cheapest: {ct.index[-1]} (₹{ct.iloc[-1]:.1f}L). "
         f"Spread across {len(ct)} cities is only {ct.iloc[0] / ct.iloc[-1] - 1:.1%}.", "q07_price_by_city")

    la = df.groupby("Locality")["Age_of_Property"].median()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(la, bins=30, ax=ax)
    ax.set_title("Q8. Median property age per locality")
    save(fig, "q08_age_by_locality")
    note("Q8. What is the median age of properties by locality?",
         f"Across {len(la)} localities the median age ranges {la.min():.0f}–{la.max():.0f} years "
         f"(overall median {df['Age_of_Property'].median():.0f}).", "q08_age_by_locality")

    bc = pd.crosstab(df["City"], df["BHK"], normalize="index")
    fig, ax = plt.subplots(figsize=(9, 10))
    sns.heatmap(bc, cmap="viridis", annot=True, fmt=".0%", ax=ax, cbar=False, annot_kws={"size": 7})
    ax.set_title("Q9. BHK distribution across cities")
    save(fig, "q09_bhk_by_city")
    note("Q9. How is BHK distributed across cities?",
         "Each BHK level (1–5) makes up roughly 20% of listings in every city — the mix is uniform.", "q09_bhk_by_city")

    top5 = df.groupby("Locality")["Price_per_SqFt"].mean().nlargest(5).index
    tr = (df[df["Locality"].isin(top5)].assign(Decade=lambda d: d["Year_Built"] // 5 * 5)
          .groupby(["Decade", "Locality"])["Price_in_Lakhs"].mean().reset_index())
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=tr, x="Decade", y="Price_in_Lakhs", hue="Locality", marker="o", ax=ax)
    ax.set_title("Q10. Price trend by build year — top-5 most expensive localities")
    ax.set_xlabel("Year built (5-year buckets)")
    save(fig, "q10_top5_locality_trends")
    note("Q10. What are the price trends for the top 5 most expensive localities?",
         "Top 5 by ₹/sq ft: " + ", ".join(top5) + ". Prices fluctuate by build year without a monotonic trend, "
         "so build year is a weak price signal.", "q10_top5_locality_trends")

    # ---------------- 11-15 Relationships
    num = ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt", "BHK", "Age_of_Property", "Floor_No",
           "Total_Floors", "Nearby_Schools", "Nearby_Hospitals", "Transport_Score", "Amenity_Count",
           "Infrastructure_Score", "Expected_Growth_Rate", "Future_Price_5Y", "Good_Investment"]
    corr = df[num].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax, annot_kws={"size": 7})
    ax.set_title("Q11. Correlation between numeric features")
    save(fig, "q11_correlation")
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack().abs().sort_values(ascending=False)
    note("Q11. How are numeric features correlated with each other?",
         "Strongest pairs: " + "; ".join(f"{a}–{b} ({corr.loc[a, b]:.2f})" for a, b in pairs.index[:5])
         + ". Raw features are otherwise nearly uncorrelated.", "q11_correlation")

    for qn, col, title in [("Q12", "Nearby_Schools", "schools"), ("Q13", "Nearby_Hospitals", "hospitals")]:
        g = df.groupby(col)["Price_per_SqFt"].mean()
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(x=g.index, y=g.values, ax=ax)
        ax.set_title(f"{qn}. Avg price per sq ft by number of nearby {title}")
        name = f"{qn.lower().replace('q', 'q')}_{title}_vs_ppsf"
        save(fig, name)
        note(f"{qn}. How do nearby {title} relate to price per sq ft?",
             f"Correlation r = {df[col].corr(df['Price_per_SqFt']):.3f}; the average stays within "
             f"₹{g.min():,.0f}–₹{g.max():,.0f}, so {title} counts barely move unit prices.", name)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs", ax=ax)
    ax.set_title("Q14. Price by furnished status")
    save(fig, "q14_price_by_furnishing")
    f = df.groupby("Furnished_Status")["Price_in_Lakhs"].mean()
    note("Q14. How does price vary by furnished status?",
         "Mean price: " + ", ".join(f"{k} ₹{v:.1f}L" for k, v in f.items()) + ".", "q14_price_by_furnishing")

    fc = df.groupby("Facing")["Price_per_SqFt"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=fc.index, y=fc.values, ax=ax)
    ax.set_title("Q15. Avg price per sq ft by facing direction")
    save(fig, "q15_ppsf_by_facing")
    note("Q15. How does price per sq ft vary by property facing direction?",
         "Mean ₹/sq ft: " + ", ".join(f"{k} ₹{v:,.0f}" for k, v in fc.items()) + ".", "q15_ppsf_by_facing")

    # ---------------- 16-20 Investment / amenities / ownership
    for qn, col, name in [("Q16", "Owner_Type", "q16_owner_type"),
                          ("Q17", "Availability_Status", "q17_availability")]:
        vc = df[col].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=vc.index, y=vc.values, ax=ax)
        ax.set_title(f"{qn}. Properties per {col.replace('_', ' ').lower()}")
        save(fig, name)
        q = ("How many properties belong to each owner type?" if qn == "Q16"
             else "How many properties are available under each availability status?")
        note(f"{qn}. {q}", ", ".join(f"{k}: {v:,}" for k, v in vc.items()) + ".", name)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Parking_Space", y="Price_in_Lakhs", ax=ax)
    ax.set_title("Q18. Price with / without parking")
    save(fig, "q18_parking")
    pk = df.groupby("Parking_Space")["Price_in_Lakhs"].mean()
    note("Q18. Does parking space affect property price?",
         "Mean price: " + ", ".join(f"{k} ₹{v:.1f}L" for k, v in pk.items()) + " — negligible difference.", "q18_parking")

    am = df.groupby("Amenity_Count")["Price_per_SqFt"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=am.index, y=am.values, ax=ax)
    ax.set_title("Q19. Avg price per sq ft by number of amenities")
    save(fig, "q19_amenities")
    note("Q19. How do amenities affect price per sq ft?",
         "Mean ₹/sq ft by amenity count: " + ", ".join(f"{k}: ₹{v:,.0f}" for k, v in am.items())
         + ". Amenities matter more for appreciation (and our investment score) than for current price.", "q19_amenities")

    tp = df.groupby("Public_Transport_Accessibility").agg(
        PPSF=("Price_per_SqFt", "mean"), Good=("Good_Investment", "mean"),
        Growth=("Expected_Growth_Rate", "mean")).reindex(["Low", "Medium", "High"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sns.barplot(x=tp.index, y=tp["PPSF"], ax=axes[0]); axes[0].set_title("Avg ₹/sq ft")
    sns.barplot(x=tp.index, y=tp["Good"] * 100, ax=axes[1]); axes[1].set_title("% Good Investment")
    fig.suptitle("Q20. Public transport accessibility vs price & investment potential")
    save(fig, "q20_transport")
    note("Q20. How does public transport accessibility relate to price per sq ft or investment potential?",
         "\n".join(f"- {k}: ₹{r.PPSF:,.0f}/sq ft, {r.Good:.1%} good investments, {r.Growth:.2%} expected growth"
                   for k, r in tp.iterrows()), "q20_transport")

    # ---------------- Extra: investment-focused questions from the brief's Step 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cg = df.groupby("City")["Expected_Growth_Rate"].mean().sort_values(ascending=False) * 100
    sns.barplot(x=cg.values[:15], y=cg.index[:15], ax=axes[0]); axes[0].set_title("Top-15 cities by expected growth %")
    sns.scatterplot(data=sample, x="Infrastructure_Score", y="Expected_Growth_Rate", alpha=0.15, s=8, ax=axes[1])
    axes[1].set_title("Infrastructure score vs expected growth")
    save(fig, "x1_growth_infrastructure")
    note("Extra. Price trends by city & infrastructure vs resale value",
         f"Fastest expected appreciation: {', '.join(cg.index[:3])}. Infrastructure score correlates "
         f"r = {df['Infrastructure_Score'].corr(df['Expected_Growth_Rate']):.2f} with growth rate. "
         f"Size vs 5-year return: r = {df['Size_in_SqFt'].corr(df['Future_Price_5Y'] / df['Price_in_Lakhs']):.3f}. "
         "(The dataset has no crime-rate column, so that brief question cannot be answered.)",
         "x1_growth_infrastructure")

    md = ["# EDA Findings\n", f"Dataset: {len(df):,} cleaned listings, {df['City'].nunique()} cities, "
          f"{df['State'].nunique()} states. Good-investment share: {df['Good_Investment'].mean():.1%}.\n"] + FINDINGS
    (REPORTS_DIR / "eda_findings.md").write_text("\n".join(md), encoding="utf-8")
    print("Wrote", REPORTS_DIR / "eda_findings.md")


if __name__ == "__main__":
    main()
