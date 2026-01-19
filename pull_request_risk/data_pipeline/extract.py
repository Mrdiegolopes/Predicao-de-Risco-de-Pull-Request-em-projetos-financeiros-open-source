"""
Modulo de Extracao de Dados de Pull Requests para Predicao de Risco em projetos financeiros

"""

import os
import time
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from pydriller import Repository, Git
import json
from pathlib import Path
import traceback

# configs
GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = os.getenv ("##########") 

# Repositórios financeiros
REPOSITORIOS_FINANCEIROS = [
    "freqtrade/freqtrade",         # Bot trading (Python) - MAIS ATIVO
    "quantopian/zipline",          # Backtesting trading  
    "ta4j/ta4j",                   # Análise técnica (Java)
    "enigmampc/catalyst",          # Crypto backtesting
    "binance/binance-connector-python",
    "ccxt/ccxt",
    "stripe/stripe-python",
    "paypal/checkout-python-sdk" ]

# Limites
LIMITE_PRS_POR_REPO = 800
ARQUIVO_SAIDA = "data/pull_requests_financeiros_final.csv"
CACHE_DIR = Path("data/cache")

# CLASSES 

class GitHubExtractor:
    def __init__(self, token: str = None, cache_enabled: bool = True):
        self.token = token or GITHUB_TOKEN
        self.headers = self._build_headers()
        self.cache_enabled = cache_enabled
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
    def _build_headers(self) -> dict:
        """Constrói headers para API do GitHub"""
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token and self.token.startswith("ghp_"):
            headers["Authorization"] = f"token {self.token}"
        return headers
    
    def _get_cached_prs(self, repo: str) -> Optional[List]:
        """Recupera PRs do cache local"""
        if not self.cache_enabled:
            return None
        
        cache_file = CACHE_DIR / f"{repo.replace('/', '_')}_prs.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _cache_prs(self, repo: str, prs: List[Dict]):
        """Salva PRs no cache local"""
        if not self.cache_enabled:
            return
        
        cache_file = CACHE_DIR / f"{repo.replace('/', '_')}_prs.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(prs, f, default=str, ensure_ascii=False)
    
    def fetch_pull_requests(self, repo: str, state: str = "closed") -> List[Dict]:
        """
        Obtém Pull Requests de um repositório.
        """
        print(f" Buscando PRs de {repo}...")
        
        # Verifica cache primeiro
        cached = self._get_cached_prs(repo)
        if cached:
            print(f"  Usando cache ({len(cached)} PRs)")
            return cached[:LIMITE_PRS_POR_REPO]
        
        prs = []
        page = 1
        max_pages = 10
        
        while len(prs) < LIMITE_PRS_POR_REPO and page <= max_pages:
            url = f"{GITHUB_API_URL}/repos/{repo}/pulls"
            params = {
                "state": state,
                "per_page": 100,
                "page": page,
                "sort": "updated",
                "direction": "desc"
            }
            
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                
                if response.status_code == 403:
                    print(f"  Rate limit atingido. Aguardando 60s...")
                    time.sleep(60)
                    continue
                elif response.status_code != 200:
                    print(f" Erro {response.status_code} ao buscar PRs")
                    break
                
                batch = response.json()
                if not batch:
                    break
                
                # Filtra por data
                for pr in batch:
                    created_at = datetime.fromisoformat(pr["created_at"].replace("Z", ""))
                    if created_at >= datetime(2018, 1, 1):  # PRs desde 2018
                        prs.append(pr)
                
                print(f"    Página {page}: +{len(batch)} PRs (total: {len(prs)})")
                
                page += 1
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                print(f" Erro na página {page}: {e}")
                break
        
        # Salva no cache
        self._cache_prs(repo, prs)
        
        print(f" Encontrados {len(prs)} PRs de {repo}")
        return prs[:LIMITE_PRS_POR_REPO]
    
    def fetch_pr_details(self, repo: str, pr_number: int) -> Dict:
        """
        Obtém detalhes completos de um PR específico.
        """
        cache_file = CACHE_DIR / f"{repo.replace('/', '_')}_pr_{pr_number}.json"
        
        # Verifica cache
        if self.cache_enabled and cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass  # Se erro no cache, busca da API
        
        # Busca da API
        url = f"{GITHUB_API_URL}/repos/{repo}/pulls/{pr_number}"
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code != 200:
                print(f"   PR #{pr_number}: Erro {response.status_code}")
                return {}
            
            pr_details = response.json()
            
            # Salva no cache
            if self.cache_enabled:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(pr_details, f, default=str, ensure_ascii=False)
            
            time.sleep(0.2)
            return pr_details
            
        except Exception as e:
            print(f"Erro ao buscar PR #{pr_number}: {e}")
            return {}

class CodeMetricsExtractor:
    """Extrator de métricas de código com fallback robusto"""
    
    def __init__(self, clone_dir: str = "data/repos"):
        self.clone_dir = Path(clone_dir)
        self.clone_dir.mkdir(parents=True, exist_ok=True)
        self.cloned_repos = set()
    
    def extract_pr_metrics(self, repo: str, pr_data: Dict) -> Dict:
        """
        Tenta extrair métricas com PyDriller, fallback para métricas básicas.
        """
        try:
        # Gate mínimo para PyDriller
            if pr_data.get("commits", 0) < 2 or pr_data.get("changed_files", 0) < 2:
                return self._get_basic_metrics(pr_data)

            return self._try_pydriller_analysis(repo, pr_data)

        except Exception as e:
            print(f"  PyDriller falhou, fallback: {str(e)[:60]}...")
            return self._get_basic_metrics(pr_data)
        
    def _try_pydriller_analysis(self, repo: str, pr_data: Dict) -> Dict:
        """Tenta análise com PyDriller"""
        repo_name = repo.split("/")[1]
        local_path = self.clone_dir / repo_name
        
        # Clona apenas uma vez por repositório
        if repo not in self.cloned_repos:
            print(f"  Preparando repositório local para {repo}...")
            try:
                if not local_path.exists():
                    repo_url = f"https://github.com/{repo}.git"
                    Git(str(local_path)).clone(repo_url)
                self.cloned_repos.add(repo)
            except Exception as e:
                raise Exception(f"Falha ao preparar repositório: {e}")
        
        # Verifica SHAs
        base_sha = pr_data.get("base", {}).get("sha")
        head_sha = pr_data.get("head", {}).get("sha")
        
        if not base_sha or not head_sha:
            raise Exception("SHAs não disponíveis")
        
        # Análise com PyDriller
        metrics = self._analyze_with_pydriller(str(local_path), base_sha, head_sha)
        
        
        return metrics
    
    def _analyze_with_pydriller(self, repo_path: str, base_sha: str, head_sha: str) -> Dict:
        """Análise real com PyDriller"""
        metrics = {
            "pr_commits": 0,
            "pr_lines_added": 0,
            "pr_lines_removed": 0,
            "pr_files_changed": 0,
            "pr_complexity_changes": 0,
            "test_files_changed": 0,
            "config_files_changed": 0,
        }
        
        try:
            for commit in Repository(
                repo_path,
                from_commit=base_sha,
                to_commit=head_sha
            ).traverse_commits():
                
                metrics["pr_commits"] += 1
                
                for mod in commit.modifications:
                    metrics["pr_lines_added"] += mod.added_lines
                    metrics["pr_lines_removed"] += mod.deleted_lines
                    metrics["pr_files_changed"] += 1
                    
                    if mod.complexity is not None:
                        metrics["pr_complexity_changes"] += mod.complexity
                    
                    # Categorização
                    filename = mod.filename.lower()
                    if any(ext in filename for ext in [".test.", "_test.", "test_", "spec."]):
                        metrics["test_files_changed"] += 1
                    elif any(term in filename for term in ["config", "settings", ".env", ".ini", ".toml"]):
                        metrics["config_files_changed"] += 1
            
        except Exception as e:
            print(f"  Erro no PyDriller: {e}")
        
        return metrics
    
    def _analyze_modified_files(self, pr_data: Dict) -> Dict:
        files_info = {
            "total_files": 0,
            "py_files": 0,
            "java_files": 0,
            "js_files": 0,
            "md_files": 0,
            "touches_security": 0,
            "touches_financial": 0
        }

        try:
            files_url = pr_data.get("url", "") + "/files"
            headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

            response = requests.get(files_url, headers=headers, timeout=15)
            if response.status_code != 200:
                return files_info

            files = response.json() or []
            files_info["total_files"] = len(files)

            security_keywords = ["auth", "security", "password", "token", "key", "crypto"]
            financial_keywords = ["payment", "transaction", "money", "account", "balance", "price"]

            for file in files:
                filename = file.get("filename", "").lower()

                if filename.endswith(".py"):
                    files_info["py_files"] += 1
                elif filename.endswith(".java"):
                    files_info["java_files"] += 1
                elif filename.endswith(".js") or filename.endswith(".ts"):
                    files_info["js_files"] += 1
                elif filename.endswith(".md"):
                    files_info["md_files"] += 1

                if any(k in filename for k in security_keywords):
                    files_info["touches_security"] += 1
                if any(k in filename for k in financial_keywords):
                    files_info["touches_financial"] += 1

        except Exception as e:
            print(f"Erro análise arquivos: {e}")

        return files_info

 
    
    def _get_basic_metrics(self, pr_data: Dict) -> Dict:
        """Métricas básicas quando PyDriller falha"""
        return {
            "pr_commits": pr_data.get("commits", 0),
            "pr_lines_added": pr_data.get("additions", 0),
            "pr_lines_removed": pr_data.get("deletions", 0),
            "pr_files_changed": pr_data.get("changed_files", 0),
            "pr_complexity_changes": 0,
            "test_files_changed": 0,
            "config_files_changed": 0,
            "py_files": 0,
            "java_files": 0,
            "js_files": 0,
            "md_files": 0,
            "touches_security": 0,
            "touches_financial": 0
        }

#FUNÇÕES AUXILIARES

def create_risk_label(pr_data: Dict, metrics: Dict) -> Dict:
    """
    Cria labels para treinamento supervisionado.
    """
    label_data = {}
    risky_conditions = []
    
    # Condição 1: PR não foi merged
    label_data["is_merged"] = int(pr_data.get("merged_at") is not None)

    
    # Condição 2: Muitos comentários
    if pr_data.get("comments", 0) > 15:
        risky_conditions.append("many_comments")
    
    # Condição 3: Toca arquivos sensíveis
    if metrics.get("touches_security", 0) > 0:
        risky_conditions.append("touches_security")
    if metrics.get("touches_financial", 0) > 0:
        risky_conditions.append("touches_financial")
    
    # Condição 4: Muitas alterações
    if metrics.get("pr_files_changed", 0) > 25:
        risky_conditions.append("many_files")
    
    # Condição 5: Termos no título
    title = pr_data.get("title", "").lower()
    if any(term in title for term in ["security", "auth", "crypto", "password", "token"]):
        risky_conditions.append("title_security")
    if any(term in title for term in ["payment", "transaction", "money", "account", "balance", "trade"]):
        risky_conditions.append("title_financial")
    
    # Define label
    label_data["is_risky"] = 1 if len(risky_conditions) > 0 else 0
    label_data["risk_reasons"] = ", ".join(risky_conditions) if risky_conditions else "none"
    
    # Tempo de merge
    if pr_data.get("merged_at"):
        try:
            created = datetime.fromisoformat(pr_data["created_at"].replace("Z", ""))
            merged = datetime.fromisoformat(pr_data["merged_at"].replace("Z", ""))
            label_data["merge_time_hours"] = (merged - created).total_seconds() / 3600
        except:
            label_data["merge_time_hours"] = -1
    else:
        label_data["merge_time_hours"] = -1
    
    return label_data

def executar_extracao_final():
    """Pipeline principal de extração"""
    print("Extração de Dados de PRs Financeiros")
    print(f"Token: {GITHUB_TOKEN[:10]}...")
    
    # Inicializa extratores
    gh_extractor = GitHubExtractor(token=GITHUB_TOKEN, cache_enabled=True)
    code_extractor = CodeMetricsExtractor()
    
    registros = []
    inicio_total = time.time()
    
    for repo_idx, repo in enumerate(REPOSITORIOS_FINANCEIROS):
        
        print(f"[{repo_idx+1}/{len(REPOSITORIOS_FINANCEIROS)}] PROCESSANDO: {repo}")
        
        
        inicio_repo = time.time()
        
        # Busca PRs
        prs = gh_extractor.fetch_pull_requests(repo, state="closed")
        
        if not prs:
            print(f"  Nenhum PR encontrado para {repo}")
            continue
        
        print(f" Processando {len(prs)} PRs...")
        
        # Processa cada PR
        for i, pr in enumerate(prs):
            try:
                print(f"    PR #{pr['number']} ({i+1}/{len(prs)})...")
                
                # Busca detalhes
                pr_details = gh_extractor.fetch_pr_details(repo, pr["number"])
                if not pr_details:
                    print(f"  Sem detalhes, pulando...")
                    continue
                
                # Extrai métricas (com fallback)
                metrics = code_extractor.extract_pr_metrics(repo, pr_details)
                
                # Cria label
                labels = create_risk_label(pr_details, metrics)
                
                # Constrói registro
                registro = {
                    # Identificação
                    "repo": repo,
                    "pr_id": pr["number"],
                    "pr_url": pr.get("html_url", ""),
                    
                    # Metadados básicos
                    "state": "merged" if pr_details.get("merged_at") else "closed",
                    "created_at": pr_details.get("created_at"),
                    "merged_at": pr_details.get("merged_at"),
                    "closed_at": pr_details.get("closed_at"),
                    "author": pr_details.get("user", {}).get("login", ""),
                    
                    # Engajamento
                    "comments": pr_details.get("comments", 0),
                    "review_comments": pr_details.get("review_comments", 0),
                    "commits": pr_details.get("commits", 0),
                    "additions": pr_details.get("additions", 0),
                    "deletions": pr_details.get("deletions", 0),
                    "changed_files": pr_details.get("changed_files", 0),
                    
                    # Conteúdo
                    "title": pr_details.get("title", ""),
                    "body_length": len(pr_details.get("body", "")),
                    "has_security_terms": int(any(term in pr_details.get("title", "").lower() 
                                                for term in ["security", "auth", "crypto", "encrypt", "token"])),
                    "has_financial_terms": int(any(term in pr_details.get("title", "").lower()
                                                  for term in ["payment", "transaction", "money", "price", "trade", "account"])),
                    
                    # Métricas de código
                    "pr_commits": metrics.get("pr_commits", 0),
                    "pr_lines_added": metrics.get("pr_lines_added", 0),
                    "pr_lines_removed": metrics.get("pr_lines_removed", 0),
                    "pr_files_changed": metrics.get("pr_files_changed", 0),
                    "pr_complexity_changes": metrics.get("pr_complexity_changes", 0),
                    "test_files_changed": metrics.get("test_files_changed", 0),
                    "config_files_changed": metrics.get("config_files_changed", 0),
                    
                    # Tipos de arquivos
                    "py_files": metrics.get("py_files", 0),
                    "java_files": metrics.get("java_files", 0),
                    "js_files": metrics.get("js_files", 0),
                    "md_files": metrics.get("md_files", 0),
                    "touches_security": metrics.get("touches_security", 0),
                    "touches_financial": metrics.get("touches_financial", 0),
                    
                    # Labels
                    "is_risky": labels.get("is_risky", 0),
                    "risk_reasons": labels.get("risk_reasons", ""),
                    "merge_time_hours": labels.get("merge_time_hours"),
                }
                
                registros.append(registro)
                
                # Rate limiting mais leve
                if i % 10 == 0:  # Apenas a cada 10 PRs
                    time.sleep(0.2)
                
            except KeyboardInterrupt:
                print(f"\n Execução interrompida pelo usuário.")
                raise
            except Exception as e:
                print(f"  Erro no PR #{pr.get('number')}: {str(e)[:50]}")
                continue
        
        tempo_repo = time.time() - inicio_repo
        print(f"  {repo}: {len([r for r in registros if r['repo'] == repo])} PRs processados em {tempo_repo/60:.1f}min")
    
    # Salva resultados
    if registros:
        df = pd.DataFrame(registros)
        
        # Cria diretório
        output_dir = Path("data")
        output_dir.mkdir(exist_ok=True)
        
        # Salva CSV
        df.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8")
        
        tempo_total = time.time() - inicio_total
        
        # Relatório final
        print(f"   Total de PRs: {len(df)}")
        print(f"   Repositórios: {df['repo'].nunique()}")
        print(f"   Período: {df['created_at'].min()} a {df['created_at'].max()}")
        print(f"   PRs Arriscados: {df['is_risky'].sum()} ({df['is_risky'].mean()*100:.1f}%)")
        print(f"   Tempo total: {tempo_total/60:.1f} minutos")
        print(f"   Arquivo salvo: {ARQUIVO_SAIDA}")
        
        # Estatísticas por repositório
        print(f"\n POR REPOSITÓRIO:")
        for repo in df['repo'].unique():
            repo_df = df[df['repo'] == repo]
            print(f"   {repo}: {len(repo_df)} PRs ({repo_df['is_risky'].mean()*100:.1f}% arriscados)")
        
        # Top razões de risco
        if df['is_risky'].sum() > 0:
            print(f"\n PRINCIPAIS RAZÕES DE RISCO:")
            reasons = df[df['risk_reasons'] != 'none']['risk_reasons']
            all_reasons = []
            for r in reasons:
                all_reasons.extend([rs.strip() for rs in r.split(',')])
            reason_counts = pd.Series(all_reasons).value_counts().head(5)
            for reason, count in reason_counts.items():
                print(f"   {reason}: {count} PRs")
        
        return df
    else:
        print("Nenhum dado extraído.")
        return pd.DataFrame()

# EXECUÇÃO PRINCIPAL 

if __name__ == "__main__":
    print("SISTEMA DE PREDIÇÃO DE RISCO DE PULL REQUESTS")
    print(f"\nConfiguração:")
    print(f"  Repositórios: {len(REPOSITORIOS_FINANCEIROS)}")
    print(f"  Limite por repo: {LIMITE_PRS_POR_REPO} PRs")
    print(f"  Total estimado: ~{LIMITE_PRS_POR_REPO * len(REPOSITORIOS_FINANCEIROS)} PRs") 
    

df = executar_extracao_final()