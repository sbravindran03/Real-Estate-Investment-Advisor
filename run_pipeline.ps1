# Runs the full pipeline: preprocess -> EDA -> train (with MLflow) -> launch the app
python src/preprocess.py; if (-not $?) { exit 1 }
python src/eda.py;        if (-not $?) { exit 1 }
python src/train.py;      if (-not $?) { exit 1 }
streamlit run app.py
