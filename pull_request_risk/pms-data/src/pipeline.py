"""
pipeline.py (pds-data - PMS FULL)
Pipeline de Extração + Transformação para Predição de Mudança de Software (PMS)

- Unidade de análise: ARQUIVO
- Step-1: Extrair métricas por arquivo em janela histórica -> RAW CSV
- Step-2: Rotular se o arquivo muda na janela futura -> label will_change
- Step-3: Transformar dataset -> TRF CSV
- Step-4: Exportar X_features.csv e y_target.csv (para pds-model)
- Extras: cache, logs, tratamento de erros, export dataset completo

Executar:
  python3 pipeline.py
"""

import os
import time
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydriller import Repository, Git

# ============================================================
# CONFIG
# ============================================================
REPOSITORIOS = [
    "freqtrade/freqtrade",
    "quantopian/zipline",
    "ta4j/ta4j",
    "ccxt/ccxt",
]

CLONE_DIR = Path("data/repos")
CACHE_DIR = Path("data/cache")
DATA_DIR = Path("data")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
CLONE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUT = DATA_DIR / "pms_raw.csv"
TRF_OUT = DATA_DIR / "pms_trf.csv"

TRANSFORMED_DIR = DATA_DIR / "transformed"
MODEL_DIR = TRANSFORMED_DIR / "modeling"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Janela temporal (PMS)
HISTORY_DAYS = 90     # passado (features)
FUTURE_DAYS = 30      # futuro (label)

# Controle de volume (evita explodir)
MAX_WINDOWS_PER_REPO = 40          # limita quantidade de janelas por repositório
MAX_FILES_PER_WINDOW = 5000       # evita janela absurda (projetos muito grandes)
ONLY_CODE_EXTENSIONS = {".py", ".java", ".js", ".ts", ".go", ".rb", ".php", ".cs", ".cpp", ".c"}

# ============================================================
# LOGGER (pipeline.log no estilo PDS)
# ============================================================
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("pms-pipeline")
    logger.setLevel(logging.DEBUG)

    for h in list(logger.handlers):
        logger.removeHandler(h)

    fh = logging.FileHandler("pipeline.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(fh)

    return logger


# ============================================================
# HELPERS
# ============================================================
def _repo_to_local_path(repo_slug: str) -> Path:
    # repo_slug: "owner/name"
    repo_name = repo_slug.split("/")[1]
    return CLONE_DIR / repo_name


def _ensure_repo_cloned(repo_slug: str, logger: logging.Logger) -> Path:
    local_path = _repo_to_local_path(repo_slug)
    if local_path.exists():
        return local_path

    url = f"https://github.com/{repo_slug}.git"
    logger.debug(f"Clonando repo {repo_slug} em {local_path} ...")
    Git(str(local_path)).clone(url)
    return local_path


def _load_cached_commits(repo_slug: str) -> Optional[List[Dict]]:
    cache_file = CACHE_DIR / f"{repo_slug.replace('/', '_')}_commits.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return None
    return None


def _cache_commits(repo_slug: str, commits: List[Dict]):
    cache_file = CACHE_DIR / f"{repo_slug.replace('/', '_')}_commits.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(commits, f, default=str, ensure_ascii=False)


def _normalize_file_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    p = p.replace("\\", "/")
    return p.strip()


def _is_code_file(path: str) -> bool:
    path = path.lower()
    for ext in ONLY_CODE_EXTENSIONS:
        if path.endswith(ext):
            return True
    return False


# ============================================================
# STEP-1: EXTRACT RAW (PMS)
# ============================================================
def extract_pms_records_for_repo(repo_slug: str, logger: logging.Logger) -> List[Dict]:
    """
    Gera registros PMS por arquivo e janela temporal.
    """
    local_path = _ensure_repo_cloned(repo_slug, logger)

    # Carregar commits (cache)
    cached = _load_cached_commits(repo_slug)
    if cached is not None:
        logger.debug(f"[CACHE] Commits carregados do cache ({repo_slug}): {len(cached)}")
        commits_meta = cached
    else:
        logger.debug(f"Extraindo commits via PyDriller: {repo_slug}")
        commits_meta = []
        for c in Repository(str(local_path)).traverse_commits():
            commits_meta.append({
                "hash": c.hash,
                "date": c.committer_date.isoformat(),
                "author": c.author.name if c.author else "unknown"
            })
        _cache_commits(repo_slug, commits_meta)

    if not commits_meta:
        return []

    # Ordena por data
    commits_meta.sort(key=lambda x: x["date"])
    min_date = datetime.fromisoformat(commits_meta[0]["date"])
    max_date = datetime.fromisoformat(commits_meta[-1]["date"])

    records: List[Dict] = []
    current = min_date
    windows_done = 0

    while (current + timedelta(days=HISTORY_DAYS + FUTURE_DAYS) <= max_date) and (windows_done < MAX_WINDOWS_PER_REPO):
        history_start = current
        history_end = current + timedelta(days=HISTORY_DAYS)
        future_end = history_end + timedelta(days=FUTURE_DAYS)

        file_stats: Dict[str, Dict] = {}
        future_changed_files = set()

        # Para performance: percorre commits na faixa
        # Re-traversal por intervalo de data usando PyDriller (filtro por since/to)
        # Nota: PyDriller usa "since/to" com datetime
        try:
            # -------------------------
            # HIST WINDOW: features
            # -------------------------
            for commit in Repository(str(local_path), since=history_start, to=history_end).traverse_commits():
                for mod in commit.modifications:
                    fname = _normalize_file_path(mod.new_path or mod.old_path)
                    if not fname or not _is_code_file(fname):
                        continue

                    fs = file_stats.setdefault(fname, {
                        "changes_count": 0,
                        "lines_added": 0,
                        "lines_removed": 0,
                        "complexity_sum": 0,
                        "authors": set(),
                    })

                    fs["changes_count"] += 1
                    fs["lines_added"] += int(mod.added_lines or 0)
                    fs["lines_removed"] += int(mod.deleted_lines or 0)
                    if mod.complexity is not None:
                        fs["complexity_sum"] += int(mod.complexity)
                    fs["authors"].add(commit.author.name if commit.author else "unknown")

            # -------------------------
            # FUT WINDOW: label
            # -------------------------
            for commit in Repository(str(local_path), since=history_end, to=future_end).traverse_commits():
                for mod in commit.modifications:
                    fname = _normalize_file_path(mod.new_path or mod.old_path)
                    if not fname or not _is_code_file(fname):
                        continue
                    future_changed_files.add(fname)

        except Exception as e:
            logger.debug(f"Falha ao analisar janela {history_start} - {history_end}: {e}")
            current += timedelta(days=FUTURE_DAYS)
            windows_done += 1
            continue

        # limita número de arquivos por janela (evita explosão)
        if len(file_stats) > MAX_FILES_PER_WINDOW:
            logger.debug(f"Janela com muitos arquivos ({len(file_stats)}). Limitando a {MAX_FILES_PER_WINDOW}.")
            # pega top por changes_count
            items = sorted(file_stats.items(), key=lambda kv: kv[1]["changes_count"], reverse=True)[:MAX_FILES_PER_WINDOW]
        else:
            items = list(file_stats.items())

        for fname, stats in items:
            lines_added = stats["lines_added"]
            lines_removed = stats["lines_removed"]
            churn = lines_added + lines_removed

            records.append({
                "repo": repo_slug,
                "file": fname,
                "window_start": history_start.isoformat(),
                "window_end": history_end.isoformat(),

                # features PMS
                "changes_count": stats["changes_count"],
                "lines_added": lines_added,
                "lines_removed": lines_removed,
                "churn": churn,
                "complexity_sum": stats["complexity_sum"],
                "n_authors": len(stats["authors"]),

                # label PMS
                "will_change": int(fname in future_changed_files),
            })

        logger.debug(
            f"[{repo_slug}] Janela {windows_done+1}: {history_start.date()}..{history_end.date()} "
            f"| arquivos={len(items)} | positivos={sum(1 for r in records[-len(items):] if r['will_change']==1)}"
        )

        current += timedelta(days=FUTURE_DAYS)
        windows_done += 1

        time.sleep(0.1)

    return records


def extract_raw_dataset(logger: logging.Logger) -> pd.DataFrame:
    all_records: List[Dict] = []
    inicio = time.time()

    for idx, repo_slug in enumerate(REPOSITORIOS):
        logger.debug(f"[Step-1] ({idx+1}/{len(REPOSITORIOS)}) Repo: {repo_slug}")
        try:
            recs = extract_pms_records_for_repo(repo_slug, logger)
            all_records.extend(recs)
        except Exception as e:
            logger.debug(f"Erro repo {repo_slug}: {e}")
            continue

    df_raw = pd.DataFrame(all_records)
    df_raw.to_csv(RAW_OUT, index=False, encoding="utf-8")

    logger.debug(f"RAW salvo em {RAW_OUT} ({len(df_raw)} linhas)")
    logger.debug(f"Tempo total Step-1: {(time.time()-inicio)/60:.2f} min")
    return df_raw


# ============================================================
# STEP-2: TRANSFORM (robusto, estilo do seu)
# ============================================================
def transform_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # dedup
    out = out.drop_duplicates(subset=["repo", "file", "window_start", "window_end"], keep="last")

    # datas
    out["window_start"] = pd.to_datetime(out["window_start"], errors="coerce", utc=True)
    out["window_end"] = pd.to_datetime(out["window_end"], errors="coerce", utc=True)

    # numeric
    numeric_cols = ["changes_count", "lines_added", "lines_removed", "churn", "complexity_sum", "n_authors", "will_change"]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    out["will_change"] = out["will_change"].astype(int)

    # filtros mínimos
    out = out[out["changes_count"] >= 1]

    # outliers p99.5 (como você fazia)
    for col in ["changes_count", "lines_added", "lines_removed", "churn", "complexity_sum"]:
        if col in out.columns and out[col].notna().any():
            lim = out[col].quantile(0.995)
            out = out[out[col] <= lim]

    # features derivadas extras (estilo seu)
    out["churn_per_change"] = np.where(out["changes_count"] > 0, out["churn"] / out["changes_count"], 0)
    out["added_removed_ratio"] = np.where(out["lines_removed"] > 0, out["lines_added"] / out["lines_removed"], out["lines_added"])

    # logs
    for col in ["changes_count", "churn", "complexity_sum", "n_authors"]:
        out[f"{col}_log"] = np.log1p(out[col].clip(lower=0))

    out = out.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    return out.reset_index(drop=True)


# ============================================================
# STEP-3: EXPORT MODEL FILES + FULL DATASET
# ============================================================
def export_modeling_files(df_trf: pd.DataFrame):
    """
    Exporta X_features.csv e y_target.csv (padrão do pds-model)
    e também salva o dataset completo transformado.
    """
    drop_cols = {"repo", "file", "window_start", "window_end"}
    df = df_trf.copy()

    y = df["will_change"].astype(int)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + ["will_change"], errors="ignore")
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    # alinhar y com X
    y = y.loc[X.index].reset_index(drop=True)
    X = X.reset_index(drop=True)

    TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X.to_csv(MODEL_DIR / "X_features.csv", index=False)
    y.to_csv(MODEL_DIR / "y_target.csv", index=False)

    df_trf.to_csv(TRANSFORMED_DIR / "pms_transformed_complete.csv", index=False, encoding="utf-8")

    print(f"X_features.csv salvo em: {MODEL_DIR / 'X_features.csv'}")
    print(f"y_target.csv salvo em: {MODEL_DIR / 'y_target.csv'}")
    print(f"Dataset completo salvo em: {TRANSFORMED_DIR / 'pms_transformed_complete.csv'}")


# ============================================================
# START
# ============================================================
def start():
    logger = setup_logger()

    logger.debug("[Step-1] Extraindo dataset RAW (PMS)")
    df_raw = extract_raw_dataset(logger)
    if df_raw.empty:
        logger.debug("Nenhum dado extraído. Encerrando.")
        return

    logger.debug("[Step-2] Transformando dataset RAW -> TRF")
    df_trf = transform_dataset(df_raw)
    df_trf.to_csv(TRF_OUT, index=False, encoding="utf-8")
    logger.debug(f"TRF salvo em {TRF_OUT} ({len(df_trf)} linhas)")

    logger.debug("[Step-3] Exportando arquivos de modelagem (X/y)")
    export_modeling_files(df_trf)
    logger.debug("Pipeline PMS concluído com sucesso.")


if __name__ == "__main__":
    try:
        start()
    except Exception as e:
        print("Erro fatal no pipeline:", e)
        print(traceback.format_exc())
