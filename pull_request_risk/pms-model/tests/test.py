"""
test.py - Testes unitários para o pipeline de modelagem (pipeline.py)

Como rodar:
  python -m unittest -v test.py

Premissas:
- O arquivo do pipeline de modelagem se chama "pipeline.py" e está no mesmo diretório.
- Estes testes NÃO dependem de arquivos CSV reais: criam dados sintéticos e usam arquivos temporários.
- Para acelerar e tornar determinístico, alguns testes "reduzem" os grids e N_SPLITS via patch.
"""

import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

import pipeline  # seu arquivo do pipeline de modelagem


class TestModelPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.workdir = self.base

        # dataset sintético (binário e estratificado)
        rng = np.random.default_rng(0)
        n = 120
        X = pd.DataFrame({
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
        })
        # cria um sinal fraco + desbalanceamento leve
        logits = 0.8 * X["f1"] - 0.4 * X["f2"] + 0.1 * rng.normal(size=n)
        p = 1 / (1 + np.exp(-logits))
        y = (p > np.quantile(p, 0.75)).astype(int)  # ~25% positivos

        self.X = X
        self.y = y

        # escreve CSVs para testar load_xy
        self.x_path = self.base / "X_features.csv"
        self.y_path = self.base / "y_target.csv"
        X.to_csv(self.x_path, index=False)
        pd.DataFrame({"y": y}).to_csv(self.y_path, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    # ----------------------------
    # setup_logger
    # ----------------------------
    def test_setup_logger_creates_logger(self):
        # escreve o log no tmp
        with patch("pipeline.logging.FileHandler") as fh_mock:
            fh_mock.return_value = MagicMock()
            logger = pipeline.setup_logger()
            self.assertIsNotNone(logger)

    # ----------------------------
    # load_xy
    # ----------------------------
    def test_load_xy_filters_nan_and_casts_int(self):
        # adiciona um NaN para garantir filtragem
        X2 = self.X.copy()
        X2.loc[0, "f1"] = np.nan
        X2.to_csv(self.x_path, index=False)

        X_loaded, y_loaded = pipeline.load_xy(str(self.x_path), str(self.y_path))
        self.assertEqual(len(X_loaded), len(self.X) - 1)
        self.assertTrue(pd.api.types.is_integer_dtype(y_loaded))

    # ----------------------------
    # make_pipeline
    # ----------------------------
    def test_make_pipeline_linear_none_has_scaler_no_smote(self):
        pipe_obj = pipeline.make_pipeline("LRC", "NONE")
        step_names = [name for name, _ in pipe_obj.steps]
        self.assertIn("scaler", step_names)
        self.assertNotIn("smote", step_names)
        self.assertIn("model", step_names)

    def test_make_pipeline_linear_smote_has_scaler_and_smote(self):
        pipe_obj = pipeline.make_pipeline("SVC", "SMOTE")
        step_names = [name for name, _ in pipe_obj.steps]
        self.assertIn("scaler", step_names)
        self.assertIn("smote", step_names)
        self.assertIn("model", step_names)

    def test_make_pipeline_tree_smote_has_smote_no_scaler(self):
        pipe_obj = pipeline.make_pipeline("RFC", "SMOTE")
        step_names = [name for name, _ in pipe_obj.steps]
        self.assertNotIn("scaler", step_names)
        self.assertIn("smote", step_names)

    # ----------------------------
    # continuous_score
    # ----------------------------
    def test_continuous_score_predict_proba(self):
        class M:
            def predict_proba(self, X):
                return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

        s = pipeline.continuous_score(M(), self.X)
        self.assertEqual(s.shape[0], len(self.X))
        self.assertTrue(np.allclose(s, 1.0))

    def test_continuous_score_decision_function(self):
        class M:
            def decision_function(self, X):
                return np.arange(len(X))

        s = pipeline.continuous_score(M(), self.X)
        self.assertEqual(s.shape[0], len(self.X))
        self.assertEqual(s[0], 0)

    def test_continuous_score_fallback_predict(self):
        class M:
            def predict(self, X):
                return np.zeros(len(X))

        s = pipeline.continuous_score(M(), self.X)
        self.assertEqual(s.shape[0], len(self.X))
        self.assertTrue(np.allclose(s, 0.0))

    # ----------------------------
    # compute_metrics
    # ----------------------------
    def test_compute_metrics_keys(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_score = np.array([0.1, 0.6, 0.8, 0.9])

        out = pipeline.compute_metrics(y_true, y_pred, y_score)
        for k in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
            self.assertIn(k, out)

    # ----------------------------
    # run_experiments (com patch para ser rápido)
    # ----------------------------
    @patch.object(pipeline, "N_SPLITS", 3)
    def test_run_experiments_returns_dataframe(self):
        # reduz modelos e grids para rodar rápido
        small_models = {"LRC": pipeline.BASE_MODELS["LRC"], "RFC": pipeline.BASE_MODELS["RFC"]}
        small_grids = {
            "LRC": {"model__C": [0.1]},
            "RFC": {"model__n_estimators": [50], "model__max_depth": [None]},
        }

        with patch.object(pipeline, "BASE_MODELS", small_models), \
             patch.object(pipeline, "PARAM_GRIDS", small_grids), \
             patch.object(pipeline, "BALANCE_MODES", ["NONE", "SMOTE"]):

            logger = MagicMock()
            df = pipeline.run_experiments(self.X, self.y, logger)

            self.assertIsInstance(df, pd.DataFrame)
            # 2 modelos x 2 balanceamentos = 4 linhas
            self.assertEqual(len(df), 4)
            self.assertIn("roc_auc", df.columns)
            self.assertIn("model", df.columns)
            self.assertIn("balance", df.columns)

    # ----------------------------
    # train_champion (gera .pkl)
    # ----------------------------
    def test_train_champion_writes_pickle(self):
        # reduz modelo e grid para acelerar
        small_models = {"LRC": pipeline.BASE_MODELS["LRC"]}
        small_grids = {"LRC": {"model__C": [0.1]}}

        champion_row = {"model": "LRC", "balance": "NONE"}

        with patch.object(pipeline, "BASE_MODELS", small_models), \
             patch.object(pipeline, "PARAM_GRIDS", small_grids), \
             patch("pipeline.open", create=True) as open_mock, \
             patch("pipeline.pickle.dump") as dump_mock:

            # simulando um "file handle" válido
            open_mock.return_value.__enter__.return_value = MagicMock()

            logger = MagicMock()
            pipeline.train_champion(self.X, self.y, champion_row, logger)

            dump_mock.assert_called_once()
            logger.debug.assert_any_call("Saved champion: model_LRC_NONE.pkl")

    # ----------------------------
    # start (integração leve) - sem rodar pesado
    # ----------------------------
    @patch.object(pipeline, "run_experiments")
    @patch.object(pipeline, "train_champion")
    def test_start_calls_run_and_train(self, mock_train, mock_run):
        # mocka retorno do run_experiments
        mock_run.return_value = pd.DataFrame([
            {"roc_auc": 0.9, "model": "LRC", "balance": "NONE", "f1": 0.5, "precision": 0.5, "recall": 0.5, "accuracy": 0.7},
            {"roc_auc": 0.8, "model": "RFC", "balance": "SMOTE", "f1": 0.4, "precision": 0.4, "recall": 0.4, "accuracy": 0.6},
        ])

        # joga o CSV no tmp sem poluir diretório real
        with patch("pipeline.setup_logger") as log_mock, \
             patch("pipeline.load_xy") as load_mock, \
             patch("pipeline.pd.DataFrame.to_csv") as to_csv_mock:

            log_mock.return_value = MagicMock()
            load_mock.return_value = (self.X, self.y)

            pipeline.start(str(self.x_path), str(self.y_path))

            mock_run.assert_called_once()
            to_csv_mock.assert_called_once()  # metrics_comparison.csv
            mock_train.assert_called_once()

            # confere se o campeão escolhido foi o de roc_auc maior (0.9)
            args, kwargs = mock_train.call_args
            champion_row = args[2]
            self.assertEqual(champion_row["model"], "LRC")
            self.assertEqual(champion_row["balance"], "NONE")


if __name__ == "__main__":
    unittest.main()
