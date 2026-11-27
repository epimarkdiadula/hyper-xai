import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
MODEL_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, SRC_DIR)

from preprocess import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES

app = FastAPI(title="OSMun-XAI ML Service", version="1.0.0")

MODEL = None
PREPROCESSOR = None
METADATA = None


def load_artifacts():
    global MODEL, PREPROCESSOR, METADATA
    try:
        MODEL = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
        PREPROCESSOR = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
        with open(os.path.join(MODEL_DIR, "model_metadata.json"), "r", encoding="utf-8") as f:
            METADATA = json.load(f)
    except Exception as e:
        MODEL = None
        PREPROCESSOR = None
        METADATA = None
        raise RuntimeError(f"Model artifacts could not be loaded: {e}")


@app.on_event("startup")
def startup_event():
    try:
        load_artifacts()
    except Exception:
        # API still starts; endpoints will report model unavailable
        pass


class PatientFeatures(BaseModel):
    RIDAGEYR: Optional[float] = None
    RIAGENDR: Optional[float] = None
    RIDRETH1: Optional[float] = None
    DMDEDUC2: Optional[float] = None
    DMDMARTZ: Optional[float] = None
    DMDHHSIZ: Optional[float] = None
    INDFMPIR: Optional[float] = None
    BMXBMI: Optional[float] = None
    BMXWT: Optional[float] = None
    BMXHT: Optional[float] = None
    BMXWAIST: Optional[float] = None
    BMXHIP: Optional[float] = None
    BMXARMC: Optional[float] = None
    BPQ020: Optional[float] = None
    BPQ150: Optional[float] = None
    BPQ080: Optional[float] = None
    BPQ101D: Optional[float] = None


def _to_frame(data: PatientFeatures) -> pd.DataFrame:
    row = {f: getattr(data, f) for f in FEATURES}
    return pd.DataFrame([row])


def _predict_proba(df: pd.DataFrame) -> float:
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model is not available.")
    Xp = PREPROCESSOR.transform(df[FEATURES])
    return float(MODEL.predict_proba(Xp)[0, 1])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None and PREPROCESSOR is not None,
        "model_version": METADATA.get("model_name") if METADATA else None,
    }


@app.post("/predict")
def predict(data: PatientFeatures):
    try:
        df = _to_frame(data)
        prob = _predict_proba(df)
        label = "High blood pressure status" if prob >= 0.5 else "Not high blood pressure status"
        return {
            "prediction": label,
            "probability": prob,
            "model_version": METADATA.get("model_name") if METADATA else "unknown",
            "disclaimer": "This is a screening support model, not a medical diagnosis.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/explain")
def explain(data: PatientFeatures):
    try:
        if MODEL is None or PREPROCESSOR is None:
            raise HTTPException(status_code=503, detail="Model is not available.")
        import shap
        df = _to_frame(data)
        Xp = PREPROCESSOR.transform(df[FEATURES])
        feature_names = list(PREPROCESSOR.get_feature_names_out())
        explainer = shap.TreeExplainer(MODEL)
        shap_values = explainer.shap_values(Xp)
        if isinstance(shap_values, list):
            sv = np.asarray(shap_values[1])[0]
            base = explainer.expected_value[1]
        else:
            sv = np.asarray(shap_values)[0]
            base = explainer.expected_value

        row = Xp[0]
        if hasattr(row, "toarray"):
            row = row.toarray()[0]
        row = np.asarray(row).ravel()
        sv = np.asarray(sv).ravel()

        contributions = []
        for name, value, shap_val in zip(feature_names, row, sv):
            contributions.append({
                "feature": name,
                "value": float(value),
                "shap": float(shap_val),
            })
        contributions.sort(key=lambda x: abs(x["shap"]), reverse=True)

        return {
            "base_value": float(np.asarray(base).ravel()[0]),
            "prediction_probability": float(MODEL.predict_proba(Xp)[0, 1]),
            "contributions": contributions,
            "model_version": METADATA.get("model_name") if METADATA else "unknown",
            "disclaimer": "SHAP shows model feature influence, not medical causation.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
