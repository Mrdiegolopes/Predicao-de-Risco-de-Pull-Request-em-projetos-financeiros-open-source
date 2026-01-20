"""
PMS Services (API) - pipeline.py

Serviço de inferência via API que encapsula o binário do modelo treinado (.pkl).
- Carrega um modelo sklearn/imblearn pipeline serializado (pickle)
- Recebe features numéricas via JSON
- Retorna: predição (0/1), score contínuo (probabilidade/decision_function) e metadados

Como executar:
  export MODEL_PATH="model_LRC_SMOTE.pkl"
  pip install -r requirements.txt
  uvicorn pipeline:app --host 0.0.0.0 --port 8000
"""

import os
import json
import pickle
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# CONFIG
# ============================================================
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "model_LRC_SMOTE.pkl")
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

# Se você quiser exigir um conjunto fixo de features, defina via env:
# export FEATURE_COLUMNS='["f1","f2","f3"]'
FEATURE_COLUMNS_ENV = os.getenv("FEATURE_COLUMNS", "").strip()
FEATURE_COLUMNS: Optional[List[str]] = None
if FEATURE_COLUMNS_ENV:
    try:
        FEATURE_COLUMNS = json.loads(FEATURE_COLUMNS_ENV)
        if not isinstance(FEATURE_COLUMNS, list) or not all(isinstance(c, str) for c in FEATURE_COLUMNS):
            raise ValueError
    except Exception:
        raise RuntimeError("FEATURE_COLUMNS deve ser um JSON array de strings, ex: '[\"f1\",\"f2\"]'.")


# ============================================================
# APP
# ============================================================
app = FastAPI(
    title="PMS Model Service",
    description="Serviço de inferência para PMS (Predição de Mudança de Software) via API.",
    version="1.0.0"
)

_MODEL = None
_MODEL_PATH = None


# ============================================================
# INPUT SCHEMAS
# ============================================================
class PredictRequest(BaseModel):
    """
    Para PMS: features podem ser métricas por arquivo em um período:
    Ex.: changes_count, lines_added, lines_removed, churn, complexity_sum, etc.

    O serviço aceita qualquer dicionário {feature: valor}.
    """
    features: Dict[str, float] = Field(..., description="Dicionário de features numéricas (feature -> valor).")
    threshold: Optional[float] = Field(None, description="Threshold opcional para classificar score em 0/1.")


class PredictBatchRequest(BaseModel):
    items: List[PredictRequest]


# ============================================================
# UTIL
# ============================================================
def _load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo não encontrado em: {path}")

    with open(path, "rb") as f:
        model = pickle.load(f)

    return model


def _ensure_model_loaded():
    global _MODEL, _MODEL_PATH
    if _MODEL is None or _MODEL_PATH != DEFAULT_MODEL_PATH:
        _MODEL = _load_model(DEFAULT_MODEL_PATH)
        _MODEL_PATH = DEFAULT_MODEL_PATH


def _to_dataframe(features: Dict[str, float]) -> pd.DataFrame:
    """
    Converte features (dict) -> DataFrame 1 linha
    Se FEATURE_COLUMNS estiver definido, garante exatamente essas colunas (ordem + faltantes = 0).
    Caso contrário, usa as chaves recebidas.
    """
    if FEATURE_COLUMNS is not None:
        row = {c: float(features.get(c, 0.0)) for c in FEATURE_COLUMNS}
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    # modo flexível (não recomendado se o modelo exigir colunas fixas)
    return pd.DataFrame([features]).apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _continuous_score(model, X_df: pd.DataFrame) -> np.ndarray:
    """
    Score contínuo para classificação:
    - predict_proba -> retorna probabilidade da classe 1
    - decision_function -> retorna score
    - fallback predict -> retorna label (0/1) como score
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_df)
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_df)
    return model.predict(X_df)


def _predict_one(features: Dict[str, float], threshold: float) -> Dict[str, Any]:
    _ensure_model_loaded()

    X_df = _to_dataframe(features)

    try:
        score = _continuous_score(_MODEL, X_df)
        score_val = float(score[0])

        # label
        pred = int(score_val >= threshold)

        # também retorna o "pred" original se existir
        try:
            raw_pred = int(_MODEL.predict(X_df)[0])
        except Exception:
            raw_pred = pred

        return {
            "prediction": pred,
            "raw_prediction": raw_pred,
            "score": score_val,
            "threshold": threshold,
            "n_features": int(X_df.shape[1]),
            "model_path": _MODEL_PATH,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao prever: {e}")


# ============================================================
# ROUTES
# ============================================================
@app.get("/health")
def health():
    """
    Health check básico.
    """
    try:
        _ensure_model_loaded()
        return {"status": "ok", "model_path": _MODEL_PATH}
    except Exception as e:
        return {"status": "error", "detail": str(e), "model_path": DEFAULT_MODEL_PATH}


@app.get("/schema")
def schema():
    """
    Exibe schema das features (se FEATURE_COLUMNS estiver definido).
    """
    return {
        "feature_columns": FEATURE_COLUMNS,
        "threshold_default": DEFAULT_THRESHOLD,
        "model_path": DEFAULT_MODEL_PATH
    }


@app.post("/predict")
def predict(req: PredictRequest):
    th = DEFAULT_THRESHOLD if req.threshold is None else float(req.threshold)
    return _predict_one(req.features, th)


@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest):
    outputs = []
    for item in req.items:
        th = DEFAULT_THRESHOLD if item.threshold is None else float(item.threshold)
        outputs.append(_predict_one(item.features, th))
    return {"results": outputs, "count": len(outputs)}
