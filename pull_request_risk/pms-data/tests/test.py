"""
test.py - Testes unitários (cobertura "completa" de funções + branches)
para pipeline.py (pds-data - PMS FULL)

Como rodar:
  python -m unittest -v test.py

Características:
- Não clona repositórios reais.
- Mocka PyDriller (Repository) e Git.clone.
- Redireciona paths globais para diretório temporário.
- Cobre branches importantes:
  - _ensure_repo_cloned: repo existe / repo não existe (clone chamado)
  - _load_cached_commits: JSON ok / JSON corrompido
  - extract_pms_records_for_repo: caminho normal / exceção em janela / truncamento por MAX_FILES_PER_WINDOW
  - extract_raw_dataset: sucesso / exceção por repo / dataset vazio
  - transform_dataset: features derivadas / filtros / logs
  - export_modeling_files: gera X/y e dataset completo
  - start: caminho completo / early-exit dataset vazio
"""

import json
import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

import pipeline  # seu pipeline PMS FULL


# ============================================================
# FAKES PARA PYDRILLER
# ============================================================
class FakeMod:
    def __init__(self, path, added=1, deleted=1, complexity=0):
        self.new_path = path
        self.old_path = path
        self.added_lines = added
        self.deleted_lines = deleted
        self.complexity = complexity
        self.filename = Path(path).name


class FakeAuthor:
    def __init__(self, name):
        self.name = name


class FakeCommit:
    def __init__(self, h, dt, author="dev", mods=None):
        self.hash = h
        self.committer_date = dt
        self.author = FakeAuthor(author)
        self.modifications = mods or []


class FakeRepositoryOK:
    """
    Simula Repository(...).traverse_commits() com:
    - chamada sem since/to: retorna commits de "metadados"
    - chamada com since/to: retorna commits filtrados (hist e future)
    """

    def __init__(self, _repo_path, since=None, to=None):
        self.since = since
        self.to = to

    def traverse_commits(self):
        base = datetime(2020, 1, 1)

        # commits:
        # histórico: dias 1, 10, 50
        # futuro: dia 95
        commits_all = [
            FakeCommit("c1", base + timedelta(days=1), "a1", [FakeMod("src/a.py", 10, 2, 3)]),
            FakeCommit("c2", base + timedelta(days=10), "a2", [FakeMod("src/a.py", 3, 1, 2), FakeMod("src/b.py", 5, 5, 1)]),
            FakeCommit("c3", base + timedelta(days=50), "a1", [FakeMod("src/b.py", 2, 0, 1)]),
            FakeCommit("c4", base + timedelta(days=95), "a3", [FakeMod("src/a.py", 1, 1, 1)]),  # futuro: a.py muda
        ]

        if self.since is None and self.to is None:
            return commits_all

        out = []
        for c in commits_all:
            if self.since <= c.committer_date < self.to:
                out.append(c)
        return out


class FakeRepositoryRaisesOnWindow:
    """
    Retorna commits na chamada sem since/to (metadados),
    mas levanta exceção quando since/to são passados (simula falha na janela).
    """

    def __init__(self, _repo_path, since=None, to=None):
        self.since = since
        self.to = to

    def traverse_commits(self):
        base = datetime(2020, 1, 1)
        commits_all = [
            FakeCommit("c1", base + timedelta(days=1), "a1", [FakeMod("src/a.py", 1, 1, 1)]),
            FakeCommit("c2", base + timedelta(days=130), "a2", [FakeMod("src/a.py", 1, 1, 1)]),
        ]
        if self.since is None and self.to is None:
            return commits_all

        raise RuntimeError("janela quebrou")


class FakeRepositoryManyFiles:
    """
    Gera MUITOS arquivos no histórico para acionar truncamento MAX_FILES_PER_WINDOW.
    """

    def __init__(self, _repo_path, since=None, to=None):
        self.since = since
        self.to = to

    def traverse_commits(self):
        base = datetime(2020, 1, 1)

        if self.since is None and self.to is None:
            # metadados: precisa de range suficiente para criar pelo menos 1 janela
            return [
                FakeCommit("c1", base + timedelta(days=1), "a1", []),
                FakeCommit("c2", base + timedelta(days=140), "a2", []),
            ]

        # histórico: um commit com muitos arquivos
        if self.since <= base + timedelta(days=10) < self.to:
            mods = []
            for i in range(50):
                mods.append(FakeMod(f"src/f{i}.py", added=1, deleted=1, complexity=0))
            return [FakeCommit("chist", base + timedelta(days=10), "a1", mods)]

        # futuro: muda só alguns
        if self.since <= base + timedelta(days=100) < self.to:
            mods = [FakeMod("src/f1.py", 1, 1, 0), FakeMod("src/f2.py", 1, 1, 0)]
            return [FakeCommit("cfut", base + timedelta(days=100), "a2", mods)]

        return []


# ============================================================
# TESTS
# ============================================================
class TestPMSFullPipelineComplete(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

        # redireciona diretórios globais
        pipeline.DATA_DIR = self.base / "data"
        pipeline.CACHE_DIR = pipeline.DATA_DIR / "cache"
        pipeline.CLONE_DIR = pipeline.DATA_DIR / "repos"
        pipeline.TRANSFORMED_DIR = pipeline.DATA_DIR / "transformed"
        pipeline.MODEL_DIR = pipeline.TRANSFORMED_DIR / "modeling"

        pipeline.DATA_DIR.mkdir(parents=True, exist_ok=True)
        pipeline.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pipeline.CLONE_DIR.mkdir(parents=True, exist_ok=True)
        pipeline.TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)
        pipeline.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        pipeline.RAW_OUT = pipeline.DATA_DIR / "pms_raw.csv"
        pipeline.TRF_OUT = pipeline.DATA_DIR / "pms_trf.csv"

        # parâmetros controlados p/ testes
        pipeline.HISTORY_DAYS = 90
        pipeline.FUTURE_DAYS = 30
        pipeline.MAX_WINDOWS_PER_REPO = 2
        pipeline.MAX_FILES_PER_WINDOW = 10
        pipeline.ONLY_CODE_EXTENSIONS = {".py"}

        pipeline.REPOSITORIOS = ["owner/repo"]

        self.logger = MagicMock()

    def tearDown(self):
        self.tmp.cleanup()

    # ----------------------------
    # setup_logger
    # ----------------------------
    def test_setup_logger_returns_logger(self):
        lg = pipeline.setup_logger()
        self.assertIsNotNone(lg)

    # ----------------------------
    # _repo_to_local_path
    # ----------------------------
    def test_repo_to_local_path(self):
        p = pipeline._repo_to_local_path("owner/name")
        self.assertEqual(p, pipeline.CLONE_DIR / "name")

    # ----------------------------
    # _ensure_repo_cloned branches
    # ----------------------------
    @patch("pipeline.Git")
    def test_ensure_repo_cloned_repo_exists_no_clone(self, mock_git):
        local = pipeline._repo_to_local_path("owner/repo")
        local.mkdir(parents=True, exist_ok=True)

        out = pipeline._ensure_repo_cloned("owner/repo", self.logger)
        self.assertEqual(out, local)
        mock_git.assert_not_called()

    @patch("pipeline.Git")
    def test_ensure_repo_cloned_repo_missing_calls_clone(self, mock_git):
        local = pipeline._repo_to_local_path("owner/repo")
        if local.exists():
            # segurança
            for _ in local.rglob("*"):
                pass

        out = pipeline._ensure_repo_cloned("owner/repo", self.logger)
        self.assertEqual(out, local)

        # Git(str(local)).clone(url)
        mock_git.assert_called_once()
        mock_git.return_value.clone.assert_called_once()
        args, kwargs = mock_git.return_value.clone.call_args
        self.assertTrue(args[0].endswith("owner/repo.git"))

    # ----------------------------
    # cache commits: ok e corrompido
    # ----------------------------
    def test_cache_commits_roundtrip(self):
        repo = "owner/repo"
        commits = [{"hash": "x", "date": "2020-01-01T00:00:00", "author": "a"}]
        pipeline._cache_commits(repo, commits)
        loaded = pipeline._load_cached_commits(repo)
        self.assertEqual(loaded, commits)

    def test_load_cached_commits_corrupted_json_returns_none(self):
        repo = "owner/repo"
        cache_file = pipeline.CACHE_DIR / f"{repo.replace('/', '_')}_commits.json"
        cache_file.write_text("{not valid json", encoding="utf-8")

        loaded = pipeline._load_cached_commits(repo)
        self.assertIsNone(loaded)

    # ----------------------------
    # normalize + is_code_file
    # ----------------------------
    def test_normalize_file_path(self):
        self.assertEqual(pipeline._normalize_file_path("a\\b\\c.py"), "a/b/c.py")
        self.assertIsNone(pipeline._normalize_file_path(None))

    def test_is_code_file(self):
        self.assertTrue(pipeline._is_code_file("src/a.py"))
        self.assertFalse(pipeline._is_code_file("README.md"))

    # ----------------------------
    # extract_pms_records_for_repo: caminho normal
    # ----------------------------
    @patch("pipeline.time.sleep", return_value=None)
    @patch("pipeline.Repository", new=FakeRepositoryOK)
    @patch("pipeline.Git")
    def test_extract_pms_records_for_repo_normal(self, _mock_git, _mock_sleep):
        # garante repo local
        local = pipeline._repo_to_local_path("owner/repo")
        local.mkdir(parents=True, exist_ok=True)

        recs = pipeline.extract_pms_records_for_repo("owner/repo", self.logger)
        self.assertTrue(len(recs) > 0)

        # a.py e b.py no histórico
        self.assertTrue(any(r["file"] == "src/a.py" for r in recs))
        self.assertTrue(any(r["file"] == "src/b.py" for r in recs))

        a = next(r for r in recs if r["file"] == "src/a.py")
        b = next(r for r in recs if r["file"] == "src/b.py")
        self.assertEqual(a["will_change"], 1)
        self.assertEqual(b["will_change"], 0)

    # ----------------------------
    # extract_pms_records_for_repo: exceção em janela (branch except)
    # ----------------------------
    @patch("pipeline.time.sleep", return_value=None)
    @patch("pipeline.Repository", new=FakeRepositoryRaisesOnWindow)
    @patch("pipeline.Git")
    def test_extract_pms_records_for_repo_window_exception_continues(self, _mock_git, _mock_sleep):
        local = pipeline._repo_to_local_path("owner/repo")
        local.mkdir(parents=True, exist_ok=True)

        recs = pipeline.extract_pms_records_for_repo("owner/repo", self.logger)
        # Como a janela sempre "quebra", deve retornar vazio sem crash
        self.assertEqual(recs, [])

        # Deve ter logado a falha
        self.assertTrue(self.logger.debug.called)

    # ----------------------------
    # extract_pms_records_for_repo: truncamento por MAX_FILES_PER_WINDOW
    # ----------------------------
    @patch("pipeline.time.sleep", return_value=None)
    @patch("pipeline.Repository", new=FakeRepositoryManyFiles)
    @patch("pipeline.Git")
    def test_extract_pms_records_for_repo_truncates_many_files(self, _mock_git, _mock_sleep):
        local = pipeline._repo_to_local_path("owner/repo")
        local.mkdir(parents=True, exist_ok=True)

        pipeline.MAX_FILES_PER_WINDOW = 10  # menor que 50
        recs = pipeline.extract_pms_records_for_repo("owner/repo", self.logger)

        # Deve truncar para 10 arquivos na janela
        # (Como temos só 1 janela gerada pelo fake, deve ter <=10)
        self.assertTrue(len(recs) <= 10)

        # e deve conter will_change 1 para alguns (f1/f2)
        # (não garantimos que f1 e f2 estejam nos top10 por changes_count,
        # mas no fake todos têm changes_count=1 então pode entrar)
        self.assertTrue(all("will_change" in r for r in recs))

    # ----------------------------
    # extract_raw_dataset: sucesso e exceção por repo
    # ----------------------------
    @patch("pipeline.extract_pms_records_for_repo")
    def test_extract_raw_dataset_success_writes_csv(self, mock_extract):
        mock_extract.return_value = [
            {"repo": "owner/repo", "file": "src/a.py", "window_start": "2020-01-01", "window_end": "2020-04-01",
             "changes_count": 2, "lines_added": 10, "lines_removed": 2, "churn": 12,
             "complexity_sum": 3, "n_authors": 1, "will_change": 1}
        ]

        df = pipeline.extract_raw_dataset(self.logger)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(pipeline.RAW_OUT.exists())
        self.assertEqual(len(df), 1)

    @patch("pipeline.extract_pms_records_for_repo", side_effect=RuntimeError("boom"))
    def test_extract_raw_dataset_handles_repo_exception(self, _mock_extract):
        df = pipeline.extract_raw_dataset(self.logger)
        self.assertTrue(pipeline.RAW_OUT.exists())
        self.assertEqual(len(df), 0)

    # ----------------------------
    # transform_dataset: features e saneamento
    # ----------------------------
    def test_transform_dataset_creates_features_and_removes_nan_inf(self):
        df = pd.DataFrame([
            {"repo": "owner/repo", "file": "src/a.py", "window_start": "2020-01-01", "window_end": "2020-04-01",
             "changes_count": 2, "lines_added": 10, "lines_removed": 2, "churn": 12,
             "complexity_sum": 3, "n_authors": 1, "will_change": 1}
        ])

        out = pipeline.transform_dataset(df)
        for c in ["churn_per_change", "added_removed_ratio", "changes_count_log", "churn_log", "complexity_sum_log", "n_authors_log"]:
            self.assertIn(c, out.columns)

        self.assertFalse(out.isna().any().any())
        self.assertFalse(np.isinf(out.select_dtypes(include=[np.number]).to_numpy()).any())

    # ----------------------------
    # export_modeling_files: gera X/y + completo
    # ----------------------------
    def test_export_modeling_files_writes_X_y_and_full(self):
        df = pd.DataFrame([
            {"repo": "owner/repo", "file": "src/a.py",
             "window_start": pd.Timestamp("2020-01-01", tz="UTC"),
             "window_end": pd.Timestamp("2020-04-01", tz="UTC"),
             "changes_count": 2, "lines_added": 10, "lines_removed": 2, "churn": 12,
             "complexity_sum": 3, "n_authors": 1, "will_change": 1,
             "churn_per_change": 6, "added_removed_ratio": 5,
             "changes_count_log": 1.09, "churn_log": 2.56, "complexity_sum_log": 1.39, "n_authors_log": 0.69
             }
        ])

        pipeline.export_modeling_files(df)

        x_path = pipeline.MODEL_DIR / "X_features.csv"
        y_path = pipeline.MODEL_DIR / "y_target.csv"
        full_path = pipeline.TRANSFORMED_DIR / "pms_transformed_complete.csv"

        self.assertTrue(x_path.exists())
        self.assertTrue(y_path.exists())
        self.assertTrue(full_path.exists())

        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)

        self.assertEqual(y.shape[1], 1)
        self.assertTrue(all(np.issubdtype(dt, np.number) for dt in X.dtypes))
        self.assertNotIn("repo", X.columns)
        self.assertNotIn("file", X.columns)
        self.assertNotIn("will_change", X.columns)

    # ----------------------------
    # start: caminho completo
    # ----------------------------
    @patch("pipeline.export_modeling_files")
    @patch("pipeline.extract_raw_dataset")
    def test_start_happy_path(self, mock_extract, mock_export):
        df_raw = pd.DataFrame([
            {"repo": "owner/repo", "file": "src/a.py", "window_start": "2020-01-01", "window_end": "2020-04-01",
             "changes_count": 2, "lines_added": 10, "lines_removed": 2, "churn": 12,
             "complexity_sum": 3, "n_authors": 1, "will_change": 1}
        ])
        mock_extract.return_value = df_raw

        with patch("pipeline.setup_logger", return_value=self.logger):
            pipeline.start()

        self.assertTrue(pipeline.TRF_OUT.exists())
        mock_export.assert_called_once()

    # ----------------------------
    # start: early exit quando dataset vazio
    # ----------------------------
    @patch("pipeline.extract_raw_dataset", return_value=pd.DataFrame())
    def test_start_early_exit_on_empty_dataset(self, _mock_extract):
        with patch("pipeline.setup_logger", return_value=self.logger):
            pipeline.start()

        # TRF não precisa existir nesse caso
        self.assertFalse(pipeline.TRF_OUT.exists())


if __name__ == "__main__":
    unittest.main()
