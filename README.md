# 🏠 Real Estate Investment Advisor
### Predicting Property Profitability & Future Value

A machine-learning application that helps investors decide:

1. **Is this property a "Good Investment"?** — binary classification
2. **What will it be worth in 5 years?** — regression (`Future_Price_5Y`)

It covers data preprocessing, EDA (20 questions), 12 trained models tracked in **MLflow** (best models registered in the MLflow Model Registry), and a **Streamlit** app with predictions, filters and visual insights.

---

## 📁 Project structure

```
project 3/
├── app.py                     # Streamlit application
├── run_pipeline.ps1           # one-shot: preprocess → EDA → train → app
├── requirements.txt
├── data/
│   ├── raw/india_housing_prices.csv          # original dataset (250,000 rows × 23 cols)
│   └── processed/cleaned_housing.csv         # cleaned + engineered + targets  (deliverable)
│       processed/lookups.json                # city medians / growth rates used by the app
├── src/
│   ├── config.py              # paths, constants, feature lists
│   ├── features.py            # feature engineering + target rules (shared with the app)
│   ├── preprocess.py          # Step 1 – cleaning, outliers, features, targets
│   ├── eda.py                 # Step 2 – 20 EDA questions → charts + findings
│   └── train.py               # Steps 3-4 – 6 classifiers + 6 regressors, MLflow, registry
├── models/                    # champion models (joblib) + model_metadata.json
├── reports/
│   ├── eda/*.png              # one chart per EDA question
│   ├── eda_findings.md        # written answers to the 20 questions
│   ├── classification_results.csv
│   └── regression_results.csv
├── mlflow.db  /  mlruns/      # MLflow tracking store, registry and artifacts
```

## ▶️ How to run

Requires Python 3.10–3.12.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src/preprocess.py      # → data/processed/
python src/eda.py             # → reports/eda/, reports/eda_findings.md
python src/train.py           # → MLflow runs + registry, models/   (add --sample 50000 for a quick run)
streamlit run app.py          # → http://localhost:8501

mlflow ui --backend-store-uri sqlite:///mlflow.db   # → http://127.0.0.1:5000
```

Or run everything at once: `.\run_pipeline.ps1`

---

## 🔬 Methodology

### Step 1 — Data preprocessing (`src/preprocess.py`)
| Task | What is done |
|---|---|
| Types & whitespace | Strip strings, coerce numerics |
| Duplicates | Drop rows identical on every column except `ID` |
| Missing values | Median for numeric, mode for categorical, empty string for `Amenities` |
| Invalid rows | Drop non-positive size / price / BHK |
| Outliers | IQR rule on price, size and ₹/sq ft → `Outlier_*` flag columns (kept, not dropped — they are plausible listings and tree models are robust) |
| Scaling / encoding | `StandardScaler` on numeric, `OneHotEncoder` on categorical (inside the sklearn `Pipeline`, so the app reuses it) |

**Data-quality fix:** the raw `Price_per_SqFt` is *lakhs per sq ft rounded to 2 dp* (values such as 0.05), which loses almost all information. It is recomputed in **₹ per sq ft** = `Price_in_Lakhs × 100000 / Size_in_SqFt`.

**Engineered features**

| Feature | Definition |
|---|---|
| `Amenity_Count` | Number of amenities listed (Gym, Pool, Garden, Clubhouse, Playground) |
| `Amenity_Density_Score` | Amenities per 1,000 sq ft |
| `Transport_Score` | Low/Medium/High → 1/2/3 |
| `Infrastructure_Score` | Mean of normalised schools, hospitals and transport (0–1) |
| `Floor_Ratio` | Floor_No / Total_Floors (capped at 1) |
| `Parking`, `Has_Security` | Yes/No → 1/0 |
| `PPSF_vs_City_Median` | Property ₹/sq ft ÷ city median ₹/sq ft |
| `Price_vs_City_Median` | Price ÷ city median price |
| `City_Growth_Rate` | City-level appreciation rate (see below) |

### Target engineering
The dataset has no historical prices, so both targets are built from the domain rules suggested in the brief.

**Regression — `Future_Price_5Y = Price × (1 + r)^5`**, combining the brief's three approaches ("fixed rate", "by location/type", "feature-based"):

```
r = City_Growth_Rate                         # 6%–10%: cheaper ("developing") cities grow faster, centred on 8%
  + type premium                             # Apartment 0, Independent House +0.25%, Villa +0.5%
  + 2% × (Infrastructure_Score − 0.5)
  + 0.1% × Amenity_Count + 0.2% × Has_Security
  + 1% if Under_Construction                 # pre-completion buyers capture build-out appreciation
  − 0.04% × Age_of_Property
  + market noise N(0, 0.3%)                  # so the model can't simply reproduce a formula
r is clipped to [2%, 18%]
```

**Classification — `Good_Investment`** (multi-factor score, brief's 3rd option). One point each for:
1. ₹/sq ft ≤ city median
2. Price ≤ city median
3. BHK ≥ 3
4. Ready to move
5. Infrastructure score ≥ median
6. 4 or more amenities

**Score ≥ 4 / 6 → Good Investment.**

> ⚠️ Because both targets are derived from the input features by rule, the models learn to reproduce those rules. Very high accuracy and R² are therefore expected and **do not** mean the models predict real-world appreciation. With real transaction histories the same pipeline would be retrained on observed 5-year prices.

### Step 2 — EDA (`src/eda.py`)
All 20 questions from the brief, plus the investment questions (growth by city, infrastructure vs resale value, size vs return), are answered with a chart and figures in **`reports/eda_findings.md`**. The dataset has no crime-rate column, so that question in the brief is skipped.

### Step 3 — Models (`src/train.py`)
80/20 stratified train/test split, with the same preprocessing pipeline for every model.

| Classification (target `Good_Investment`) | Regression (target `Future_Price_5Y`) |
|---|---|
| Logistic Regression | Linear Regression |
| Decision Tree | Ridge |
| Random Forest | Decision Tree |
| HistGradientBoosting | Random Forest |
| XGBoost | XGBoost |
| LightGBM | LightGBM |
| *Metrics:* Accuracy, Precision, Recall, F1, ROC-AUC, Confusion matrix | *Metrics:* RMSE, MSE, MAE, R², MAPE |

### Step 4 — MLflow
* Tracking store: `sqlite:///mlflow.db`, with artifacts in `mlruns/`
* Two experiments: `RealEstate_GoodInvestment_Classification` and `RealEstate_FuturePrice5Y_Regression`
* Each run logs: hyper-parameters, all metrics, confusion matrix or actual-vs-predicted plot, a feature-importance plot and JSON, and the full sklearn pipeline with its signature and input example
* **Model Registry:** the best classifier (by F1) is registered as `GoodInvestmentClassifier`, and the best regressor (by RMSE) as `FuturePrice5YRegressor`. Both carry the alias **`champion`**, so they load with `mlflow.pyfunc.load_model("models:/GoodInvestmentClassifier@champion")`

### Step 5 — Streamlit app (`app.py`)
| Tab | Contents |
|---|---|
| 🔮 **Predict** | Property form → *"Is this a Good Investment?"* with a confidence score, *"Estimated price after 5 years"* with a ±RMSE range, implied CAGR, a 5-year value projection chart, the 6-point investment checklist and the top feature importances |
| 🔎 **Explore listings** | Filter by city, property type, BHK, price and size, or show good investments only; KPIs, ranked table and CSV download |
| 📊 **Market insights** | City × property-type ₹/sq ft heatmap, city growth-vs-price bubble chart, state ranking, price trend by build year, good-investment share by BHK and transport access |
| 🧠 **Model performance** | Leaderboards for all 12 models, confusion matrix, feature importance for both champions |

---

## 📈 Results & key findings
See `reports/eda_findings.md`, `reports/classification_results.csv` and `reports/regression_results.csv` (generated by the pipeline).

**Data:** 250,000 listings across 42 cities and 20 states. There were no missing values or duplicates. 7.9% of rows are flagged as ₹/sq ft outliers. 36.7% of listings are labelled *Good Investment*, and the mean expected growth rate is 8.6% p.a.

**Classification (test set, 50,000 rows)**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8839 | 0.8507 | 0.8294 | 0.8399 | 0.9561 |
| **Decision Tree** 🏆 | **0.99996** | **0.99995** | **0.99995** | **0.99995** | 0.99996 |
| Random Forest | 0.99948 | 0.99967 | 0.99891 | 0.99929 | 1.0000 |
| HistGradientBoosting | 0.99914 | 0.99918 | 0.99848 | 0.99883 | 1.0000 |
| XGBoost | 0.99972 | 0.99929 | 0.99995 | 0.99962 | 1.0000 |
| LightGBM | 0.99956 | 0.99929 | 0.99951 | 0.99940 | 1.0000 |

**Regression — Future_Price_5Y (₹ Lakhs; price range 10–500 L)**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 12.51 | 9.32 | 0.9966 |
| Ridge | 12.52 | 9.33 | 0.9966 |
| Decision Tree | 10.63 | 7.53 | 0.9976 |
| Random Forest | 8.02 | 5.65 | 0.9986 |
| **XGBoost** 🏆 | **6.32** | **4.46** | **0.9991** |
| LightGBM | 6.34 | 4.49 | 0.9991 |


**Key findings**
* The linear model is clearly weaker at classification (F1 0.84) because the rule uses *city-relative* thresholds and step functions, which trees capture exactly.
* The most important features for the investment decision are: price relative to the city median (₹/sq ft and total price), amenity count, availability status, BHK and infrastructure score.
* In the raw data, price is almost independent of size, location, amenities, schools, hospitals, furnishing, facing and parking (all |r| < 0.01). City averages differ by only ~3%. The dataset is synthetic, so the targets carry the domain logic.
* The fastest expected appreciation is in cheaper, developing cities (e.g. Bilaspur, Trivandrum, Jamshedpur). Better public-transport access raises the share of good investments.

## 💼 Business use cases
* **Investors** — screen a listing in seconds: a buy/avoid signal plus a 5-year value estimate.
* **Home buyers** — use the Explore tab to find high-return properties in developing (faster-growing) cities that fit their budget and BHK needs.
* **Real-estate platforms** — the registered MLflow models can score every listing automatically and show a "Good Investment" badge.
* **Trust & transparency** — the checklist, confidence scores and feature importance show *why* a property was rated, rather than giving a black-box verdict.

## ⚠️ Limitations
* The dataset appears synthetic: features are nearly independent of price (for example, size and price have r ≈ 0), and floor numbers can exceed total floors.
* Targets are rule-based proxies (see above). Real 5-year resale data would be needed for genuine forecasting.
* No crime, RERA or macro-economic data is available. These would be valuable additions.
