"""Central configuration: paths, constants and feature lists shared by all scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "india_housing_prices.csv"
PROCESSED_CSV = ROOT / "data" / "processed" / "cleaned_housing.csv"
LOOKUPS_JSON = ROOT / "data" / "processed" / "lookups.json"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
EDA_DIR = REPORTS_DIR / "eda"

# MLflow: SQLite backend (needed for the model registry) + local artifact folder
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_ROOT = (ROOT / "mlruns").as_uri()
EXPERIMENT_CLF = "RealEstate_GoodInvestment_Classification"
EXPERIMENT_REG = "RealEstate_FuturePrice5Y_Regression"
REGISTERED_CLF = "GoodInvestmentClassifier"
REGISTERED_REG = "FuturePrice5YRegressor"

RANDOM_STATE = 42
CURRENT_YEAR = 2025          # reference year used by the dataset's Age_of_Property
HORIZON_YEARS = 5
BASE_GROWTH_RATE = 0.08      # 8% p.a. baseline appreciation (brief's example)

AMENITY_LIST = ["Gym", "Pool", "Garden", "Clubhouse", "Playground"]
TRANSPORT_MAP = {"Low": 1, "Medium": 2, "High": 3}

# Model inputs (what a user can supply in the app + engineered features)
NUMERIC_FEATURES = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property",
    "Floor_No", "Total_Floors", "Nearby_Schools", "Nearby_Hospitals",
    "Transport_Score", "Parking", "Has_Security", "Amenity_Count",
    "Amenity_Density_Score", "Infrastructure_Score", "Floor_Ratio",
    "PPSF_vs_City_Median", "Price_vs_City_Median", "City_Growth_Rate",
]
CATEGORICAL_FEATURES = [
    "State", "City", "Property_Type", "Furnished_Status", "Facing",
    "Owner_Type", "Availability_Status",
]
CLF_TARGET = "Good_Investment"
REG_TARGET = "Future_Price_5Y"
