"""
test.py - Testes unitários para o PMS Model Service (pipeline.py)

Como rodar:
  python -m unittest -v test.py

Requisitos:
  pip install fastapi uvicorn httpx pydantic numpy pandas scikit-learn imbalanced-learn xgboost

Observações:
- Não precisa subir servidor (usa TestClient do FastAPI).
- O carregamento do modelo (.pkl) é mockado, então não precisa de arquivo real.
- Testa: /health, /schema, /predict, /predict_batch + utilitários internos.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from fastapi.testclient import TestClient

import pipeline  # seu arquivo do serviço (pipeline.py)


# ----------------------------
# Modelo fake para testes
# ----------------------------
class FakeModelProba:
    """Modelo fake com predict_proba."""
    def __init__(self, proba=0.8):
        self._proba = float(proba)

    def predict_proba(self, X):
        n = len(X)
        # retorna prob da classe 1 constante
        return np.column_stack([np.full(n, 1 - self._proba), np.full(n, self._proba)])

    def predict(self, X):
        # label baseado em 0.5 para consistência
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class FakeModelDecision:
    """Modelo fake com decision_function."""
    def __init__(self, score=0.2):
        self._score = float(score)

    def decision_function(self, X):
        return np.full(len(X), self._score)

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


class FakeModelPredictOnly:
    """Modelo fake apenas com predict."""
    def predict(self, X):
        return np.ones(len(X), dtype=int)


class TestPMSServices(unittest.TestCase):
    def setUp(self):
        # garante envs limpos (importante pois o pipeline lê env)
        os.environ.pop("FEATURE_COLUMNS", None)
        os.environ.pop("MODEL_PATH", None)
        os.environ.pop("THRESHOLD", None)

        # reset globals do módulo para evitar bleed entre testes
        pipeline.FEATURE_COLUMNS = None
        pipeline.DEFAULT_MODEL_PATH = "model_LRC_SMOTE.pkl"
        pipeline.DEFAULT_THRESHOLD = 0.5
        pipeline._MODEL = None
        pipeline._MODEL_PATH = None

        self.client = TestClient(pipeline.app)

    # ----------------------------
    # /schema
    # ----------------------------
    def test_schema_without_feature_columns(self):
        r = self.client.get("/schema")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("threshold_default", data)
        self.assertEqual(data["feature_columns"], None)

    def test_schema_with_feature_columns(self):
        pipeline.FEATURE_COLUMNS = ["f1", "f2"]
        r = self.client.get("/schema")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["feature_columns"], ["f1", "f2"])

    # ----------------------------
    # /health
    # ----------------------------
    def test_health_ok_when_model_loads(self):
        with patch.object(pipeline, "_load_model", return_value=FakeModelProba(0.7)):
            r = self.client.get("/health")
            self.assertEqual(r.status_code, 200)
            data = r.json()
            self.assertEqual(data["status"], "ok")
            self.assertIn("model_path", data)

    def test_health_error_when_model_missing(self):
        with patch.object(pipeline, "_load_model", side_effect=FileNotFoundError("nope")):
            r = self.client.get("/health")
            self.assertEqual(r.status_code, 200)
            data = r.json()
            self.assertEqual(data["status"], "error")
            self.assertIn("detail", data)

    # ----------------------------
    # util: _to_dataframe
    # ----------------------------
    def test_to_dataframe_flexible_mode(self):
        pipeline.FEATURE_COLUMNS = None
        df = pipeline._to_dataframe({"a": 1, "b": 2.5})
        self.assertEqual(list(df.columns), ["a", "b"])
        self.assertEqual(df.shape, (1, 2))

    def test_to_dataframe_fixed_columns_fills_missing_with_zero(self):
        pipeline.FEATURE_COLUMNS = ["f1", "f2", "f3"]
        df = pipeline._to_dataframe({"f1": 10, "f3": 5})
        self.assertEqual(list(df.columns), ["f1", "f2", "f3"])
        self.assertEqual(float(df.loc[0, "f2"]), 0.0)

    # ----------------------------
    # util: _continuous_score precedence
    # ----------------------------
    def test_continuous_score_predict_proba(self):
        model = FakeModelProba(0.9)
        df = pipeline._to_dataframe({"x": 1})
        score = pipeline._continuous_score(model, df)
        self.assertTrue(np.allclose(score, 0.9))

    def test_continuous_score_decision_function(self):
        model = FakeModelDecision(0.33)
        df = pipeline._to_dataframe({"x": 1})
        score = pipeline._continuous_score(model, df)
        self.assertTrue(np.allclose(score, 0.33))

    def test_continuous_score_fallback_predict(self):
        model = FakeModelPredictOnly()
        df = pipeline._to_dataframe({"x": 1})
        score = pipeline._continuous_score(model, df)
        self.assertTrue(np.allclose(score, 1.0))

    # ----------------------------
    # /predict
    # ----------------------------
    def test_predict_uses_default_threshold(self):
        # score=0.8 => pred=1 com threshold default 0.5
        with patch.object(pipeline, "_load_model", return_value=FakeModelProba(0.8)):
            r = self.client.post("/predict", json={"features": {"f1": 1.0}})
            self.assertEqual(r.status_code, 200)
            data = r.json()
            self.assertEqual(data["prediction"], 1)
            self.assertAlmostEqual(data["score"], 0.8, places=6)
            self.assertEqual(data["threshold"], 0.5)

    def test_predict_override_threshold(self):
        # score=0.6 e threshold=0.7 => pred=0
        with patch.object(pipeline, "_load_model", return_value=FakeModelProba(0.6)):
            r = self.client.post("/predict", json={"features": {"f1": 1.0}, "threshold": 0.7})
            self.assertEqual(r.status_code, 200)
            data = r.json()
            self.assertEqual(data["prediction"], 0)
            self.assertAlmostEqual(data["score"], 0.6, places=6)
            self.assertEqual(data["threshold"], 0.7)

    def test_predict_fixed_feature_columns_applied(self):
        pipeline.FEATURE_COLUMNS = ["f1", "f2"]
        with patch.object(pipeline, "_load_model", return_value=FakeModelProba(0.8)):
            r = self.client.post("/predict", json={"features": {"f1": 1.0}})
            self.assertEqual(r.status_code, 200)
            data = r.json()
            self.assertEqual(data["n_features"], 2)  # f2 preenchido com 0

    def test_predict_returns_http_400_on_failure(self):
        class BadModel:
            def predict_proba(self, X):
                raise RuntimeError("boom")

        with patch.object(pipeline, "_load_model", return_value=BadModel()):
            r = self.client.post("/predict", json={"features": {"f1": 1.0}})
            self.assertEqual(r.status_code, 400)
            data = r.json()
            self.assertIn("detail", data)

    # ----------------------------
    # /predict_batch
    # ----------------------------
    def test_predict_batch_outputs_count(self):
        with patch.object(pipeline, "_load_model", return_value=FakeModelProba(0.8)):
            r = self.client.post("/predict_batch", json={
                "items": [
                    {"features": {"f1": 1.0}},
                    {"features": {"f1": 2.0}, "threshold": 0.9}
                ]
            })
            self.assertEqual(r.status_code, 200)
            data = r.json()
            self.assertEqual(data["count"], 2)
            self.assertEqual(len(data["results"]), 2)
            self.assertEqual(data["results"][0]["prediction"], 1)
            # segundo: threshold 0.9 com score 0.8 => 0
            self.assertEqual(data["results"][1]["prediction"], 0)


if __name__ == "__main__":
    unittest.main()
