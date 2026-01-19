"""
transform.py - Módulo de Transformação de Dados para Pipeline de PRs Financeiros

Responsabilidades:
1. Limpeza e tratamento de dados
2. Feature engineering
3. Enriquecimento com novas features
4. Normalização e codificação
5. Preparação para modelagem/análise
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataTransformer:
    """Classe principal para transformação de dados de PRs"""
    
    def __init__(self, data_path: str = "data/pull_requests_financeiros_final.csv"):
        self.data_path = Path(data_path)
        self.df = None
        self.transformed_df = None
        
    def load_data(self) -> pd.DataFrame:
        """Carrega dados do arquivo CSV"""
        logger.info(f"Carregando dados de {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path, low_memory=False)
        logger.info(f"Dados carregados: {len(self.df)} registros, {len(self.df.columns)} colunas")
        
        return self.df
    
    def basic_cleanup(self) -> 'DataTransformer':
        """Executa limpeza básica dos dados"""
        logger.info("Iniciando limpeza básica dos dados...")
        
        if self.df is None:
            self.load_data()
        
        # 1. Remover duplicatas
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['repo', 'pr_id'], keep='first')
        logger.info(f"Duplicatas removidas: {initial_count - len(self.df)}")
        
        # 2. Tratar valores nulos
        self._handle_missing_values()
        
        # 3. Converter tipos de dados
        self._convert_data_types()
        
        # 4. Correções básicas
        self._apply_basic_fixes()
        
        return self
    
    def _handle_missing_values(self):
        """Trata valores nulos de forma estratégica"""
        logger.info("Tratando valores nulos...")
        
        # Colunas onde null significa 0
        zero_if_null = ['comments', 'review_comments', 'commits', 'additions', 
                       'deletions', 'changed_files', 'body_length',
                       'pr_commits', 'pr_lines_added', 'pr_lines_removed', 
                       'pr_files_changed', 'pr_complexity_changes',
                       'test_files_changed', 'config_files_changed',
                       'py_files', 'java_files', 'js_files', 'md_files',
                       'touches_security', 'touches_financial']
        
        for col in zero_if_null:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)
        
        # Colunas de texto
        text_cols = ['title', 'risk_reasons', 'author']
        for col in text_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('')
        
        # Colunas booleanas/int
        if 'is_risky' in self.df.columns:
            self.df['is_risky'] = self.df['is_risky'].fillna(0).astype(int)
        
        if 'has_security_terms' in self.df.columns:
            self.df['has_security_terms'] = self.df['has_security_terms'].fillna(0).astype(int)
        
        if 'has_financial_terms' in self.df.columns:
            self.df['has_financial_terms'] = self.df['has_financial_terms'].fillna(0).astype(int)
        
        # Tempo de merge: -1 para não merged
        if 'merge_time_hours' in self.df.columns:
            self.df['merge_time_hours'] = self.df['merge_time_hours'].fillna(-1)
        
        logger.info("Valores nulos tratados")
    
    def _convert_data_types(self):
        """Converte colunas para tipos apropriados"""
        logger.info("Convertendo tipos de dados...")
        
        # Converter datas
        date_cols = ['created_at', 'merged_at', 'closed_at']
        for col in date_cols:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                except:
                    logger.warning(f"Falha ao converter {col} para datetime")
        
        # Converter colunas numéricas
        numeric_cols = ['comments', 'review_comments', 'commits', 'additions', 
                       'deletions', 'changed_files', 'body_length',
                       'pr_commits', 'pr_lines_added', 'pr_lines_removed', 
                       'pr_files_changed', 'merge_time_hours']
        
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def _apply_basic_fixes(self):
        """Aplica correções básicas"""
        # Garantir que state seja 'merged' ou 'closed'
        if 'state' in self.df.columns:
            self.df['state'] = self.df['state'].apply(
                lambda x: 'merged' if str(x).strip().lower() == 'merged' else 'closed'
            )
        
        # Limpar author (remover [bot])
        if 'author' in self.df.columns:
            self.df['author'] = self.df['author'].str.replace(r'\[bot\]', '', regex=True).str.strip()
        
        # Limpar risk_reasons
        if 'risk_reasons' in self.df.columns:
            self.df['risk_reasons'] = self.df['risk_reasons'].apply(
                lambda x: str(x).strip().lower() if pd.notna(x) else 'none'
            )
    
    def engineer_features(self) -> 'DataTransformer':
        """Cria novas features baseadas nos dados existentes"""
        logger.info("Criando novas features...")
        
        # 1. Features temporais
        self._create_time_features()
        
        # 2. Features de tamanho e complexidade
        self._create_size_complexity_features()
        
        # 3. Features de engajamento
        self._create_engagement_features()
        
        # 4. Features de conteúdo
        self._create_content_features()
        
        # 5. Features de arquivos
        self._create_file_features()
        
        # 6. Features de risco aprimoradas
        self._create_enhanced_risk_features()
        
        # 7. Features categóricas
        self._create_categorical_features()
        
        logger.info(f"Total de features após engenharia: {len(self.df.columns)}")
        return self
    
    def _create_time_features(self):
        """Cria features baseadas em tempo"""
        logger.info("  Criando features temporais...")
        
        if 'created_at' in self.df.columns and pd.api.types.is_datetime64_any_dtype(self.df['created_at']):
            # Extrair componentes da data
            self.df['created_year'] = self.df['created_at'].dt.year
            self.df['created_month'] = self.df['created_at'].dt.month
            self.df['created_weekday'] = self.df['created_at'].dt.weekday
            self.df['created_hour'] = self.df['created_at'].dt.hour
            
            # Indicador de fim de semana
            self.df['is_weekend'] = self.df['created_weekday'].apply(lambda x: 1 if x >= 5 else 0)
            
            # Indicador de horário comercial (9-17)
            self.df['is_business_hours'] = self.df['created_hour'].apply(
                lambda x: 1 if 9 <= x <= 17 else 0
            )
        
        # Tempo até fechamento (para PRs fechados)
        if all(col in self.df.columns for col in ['created_at', 'closed_at']):
            mask = self.df['closed_at'].notna() & self.df['created_at'].notna()
            self.df.loc[mask, 'time_to_close_hours'] = (
                (self.df.loc[mask, 'closed_at'] - self.df.loc[mask, 'created_at']).dt.total_seconds() / 3600
            )
            self.df['time_to_close_hours'] = self.df['time_to_close_hours'].fillna(-1)
        
        # Idade do PR (em dias desde criação)
        if 'created_at' in self.df.columns:
            latest_date = self.df['created_at'].max()
            self.df['pr_age_days'] = (latest_date - self.df['created_at']).dt.days
    
    def _create_size_complexity_features(self):
        """Cria features relacionadas ao tamanho e complexidade"""
        logger.info("  Criando features de tamanho e complexidade...")
        
        # Tamanho total do PR
        self.df['total_lines_changed'] = self.df['additions'] + self.df['deletions']
        
        # Densidade de mudança (linhas por arquivo)
        self.df['lines_per_file'] = np.where(
            self.df['changed_files'] > 0,
            self.df['total_lines_changed'] / self.df['changed_files'],
            0
        )
        
        # Net lines (additions - deletions)
        self.df['net_lines'] = self.df['additions'] - self.df['deletions']
        
        # Razão de adições/deleções
        self.df['add_delete_ratio'] = np.where(
            self.df['deletions'] > 0,
            self.df['additions'] / self.df['deletions'],
            self.df['additions']  # Se deletions = 0, usar apenas additions
        )
        
        # Tamanho relativo (normalizado por repo)
        for repo in self.df['repo'].unique():
            repo_mask = self.df['repo'] == repo
            repo_mean = self.df.loc[repo_mask, 'total_lines_changed'].mean()
            if repo_mean > 0:
                self.df.loc[repo_mask, 'relative_pr_size'] = (
                    self.df.loc[repo_mask, 'total_lines_changed'] / repo_mean
                )
            else:
                self.df.loc[repo_mask, 'relative_pr_size'] = 0
        
        # Complexidade por linha
        self.df['complexity_per_line'] = np.where(
            self.df['total_lines_changed'] > 0,
            self.df['pr_complexity_changes'] / self.df['total_lines_changed'],
            0
        )
    
    def _create_engagement_features(self):
        """Cria features relacionadas ao engajamento"""
        logger.info("  Criando features de engajamento...")
        
        # Total de comentários
        self.df['total_comments'] = self.df['comments'] + self.df['review_comments']
        
        # Densidade de comentários (comentários por linha)
        self.df['comments_per_line'] = np.where(
            self.df['total_lines_changed'] > 0,
            self.df['total_comments'] / self.df['total_lines_changed'],
            0
        )
        
        # Densidade de comentários por arquivo
        self.df['comments_per_file'] = np.where(
            self.df['changed_files'] > 0,
            self.df['total_comments'] / self.df['changed_files'],
            0
        )
        
        # Engajamento por commit
        self.df['comments_per_commit'] = np.where(
            self.df['commits'] > 0,
            self.df['total_comments'] / self.df['commits'],
            0
        )
        
        # Indicador de discussão intensa
        self.df['has_intense_discussion'] = self.df['total_comments'].apply(
            lambda x: 1 if x > 10 else 0
        )
    
    def _create_content_features(self):
        """Cria features baseadas no conteúdo do PR"""
        logger.info("  Criando features de conteúdo...")
        
        # Features do título
        if 'title' in self.df.columns:
            self.df['title_length'] = self.df['title'].str.len()
            self.df['title_word_count'] = self.df['title'].str.split().str.len()
            
            # Presença de termos especiais
            self.df['title_has_feature'] = self.df['title'].str.contains(
                r'\bfeat(ure)?\b|\benhance\b|\badd\b', case=False, regex=True
            ).astype(int)
            
            self.df['title_has_fix'] = self.df['title'].str.contains(
                r'\bfix\b|\bbug\b|\berror\b', case=False, regex=True
            ).astype(int)
            
            self.df['title_has_refactor'] = self.df['title'].str.contains(
                r'\brefactor\b|\bcleanup\b|\boptimiz', case=False, regex=True
            ).astype(int)
        
        # Features do body
        if 'body_length' in self.df.columns:
            # Densidade de informação (tamanho relativo ao código)
            self.df['body_code_ratio'] = np.where(
                self.df['total_lines_changed'] > 0,
                self.df['body_length'] / self.df['total_lines_changed'],
                0
            )
            
            # Categorias de tamanho do body
            self.df['body_size_category'] = pd.cut(
                self.df['body_length'],
                bins=[-1, 0, 100, 500, 1000, float('inf')],
                labels=['empty', 'tiny', 'small', 'medium', 'large']
            )
    
    def _create_file_features(self):
        """Cria features relacionadas aos tipos de arquivos"""
        logger.info("  Criando features de tipos de arquivos...")
        
        # Proporção de tipos de arquivos
        file_type_cols = ['py_files', 'java_files', 'js_files', 'md_files']
        total_files = self.df[file_type_cols].sum(axis=1)
        
        for col in file_type_cols:
            if col in self.df.columns:
                self.df[f'{col}_ratio'] = np.where(
                    total_files > 0,
                    self.df[col] / total_files,
                    0
                )
        
        # Diversidade de tipos de arquivos
        self.df['file_type_diversity'] = self.df[file_type_cols].gt(0).sum(axis=1)
        
        # Indicadores de arquivos sensíveis
        self.df['has_security_files'] = (self.df['touches_security'] > 0).astype(int)
        self.df['has_financial_files'] = (self.df['touches_financial'] > 0).astype(int)
        
        # Proporção de arquivos de teste/config
        self.df['test_file_ratio'] = np.where(
            self.df['changed_files'] > 0,
            self.df['test_files_changed'] / self.df['changed_files'],
            0
        )
        
        self.df['config_file_ratio'] = np.where(
            self.df['changed_files'] > 0,
            self.df['config_files_changed'] / self.df['changed_files'],
            0
        )
        
        # Indicador de PR focado (apenas um tipo de arquivo)
        self.df['is_focused_pr'] = (self.df['file_type_diversity'] == 1).astype(int)
    
    def _create_enhanced_risk_features(self):
        """Cria features aprimoradas de risco"""
        logger.info("  Criando features aprimoradas de risco...")
        
        # Score de risco baseado em múltiplos fatores
        risk_factors = []
        
        # Fator 1: Tamanho extremo
        q95_lines = self.df['total_lines_changed'].quantile(0.95)
        risk_factors.append((self.df['total_lines_changed'] > q95_lines).astype(int) * 2)
        
        # Fator 2: Muitos arquivos
        q95_files = self.df['changed_files'].quantile(0.95)
        risk_factors.append((self.df['changed_files'] > q95_files).astype(int) * 2)
        
        # Fator 3: Arquivos sensíveis
        risk_factors.append(self.df['has_security_files'] * 3)
        risk_factors.append(self.df['has_financial_files'] * 3)
        
        # Fator 4: Discussão intensa
        risk_factors.append(self.df['has_intense_discussion'] * 1)
        
        # Fator 5: Complexidade alta
        q90_complexity = self.df['complexity_per_line'].quantile(0.90)
        risk_factors.append((self.df['complexity_per_line'] > q90_complexity).astype(int) * 2)
        
        # Calcular score total
        if risk_factors:
            risk_score = sum(risk_factors)
            self.df['risk_score_raw'] = risk_score
            
            # Normalizar para 0-10
            max_score = risk_score.max()
            if max_score > 0:
                self.df['risk_score_norm'] = (risk_score / max_score * 10).round(2)
            else:
                self.df['risk_score_norm'] = 0
            
            # Categorias de risco
            self.df['risk_category'] = pd.cut(
                self.df['risk_score_norm'],
                bins=[-0.1, 0, 3, 7, 10],
                labels=['none', 'low', 'medium', 'high']
            )
        
        # Features específicas das razões de risco
        if 'risk_reasons' in self.df.columns:
            # Contar número de razões
            self.df['risk_reason_count'] = self.df['risk_reasons'].apply(
                lambda x: len(str(x).split(',')) if x != 'none' else 0
            )
            
            # Indicadores para cada razão
            common_reasons = ['many_files', 'touches_financial', 'touches_security', 
                             'many_comments', 'title_financial', 'title_security']
            
            for reason in common_reasons:
                self.df[f'has_risk_reason_{reason}'] = self.df['risk_reasons'].str.contains(
                    reason, case=False
                ).astype(int)
    
    def _create_categorical_features(self):
        """Cria features categóricas codificadas"""
        logger.info("  Criando features categóricas...")
        
        # Codificar repositório (one-hot encoding reduzido)
        repo_dummies = pd.get_dummies(self.df['repo'], prefix='repo', dtype=int)
        # Manter apenas as colunas mais comuns
        if len(repo_dummies.columns) > 10:
            top_repos = self.df['repo'].value_counts().head(10).index
            repo_dummies = pd.get_dummies(
                self.df['repo'].apply(lambda x: x if x in top_repos else 'other'),
                prefix='repo',
                dtype=int
            )
        
        self.df = pd.concat([self.df, repo_dummies], axis=1)
        
        # Codificar autor (apenas autores mais ativos)
        if 'author' in self.df.columns:
            top_authors = self.df['author'].value_counts().head(20).index
            self.df['author_category'] = self.df['author'].apply(
                lambda x: x if x in top_authors else 'other'
            )
            author_dummies = pd.get_dummies(
                self.df['author_category'], 
                prefix='author_top',
                dtype=int
            )
            self.df = pd.concat([self.df, author_dummies], axis=1)
        
        # Codificar mês
        if 'created_month' in self.df.columns:
            month_dummies = pd.get_dummies(
                self.df['created_month'], 
                prefix='month',
                dtype=int
            )
            self.df = pd.concat([self.df, month_dummies], axis=1)
    
    def create_analytical_dataset(self) -> pd.DataFrame:
        """Cria dataset otimizado para análise"""
        logger.info("Criando dataset analítico...")
        
        # Definir colunas relevantes para análise
        analytical_cols = [
            # Identificadores
            'repo', 'pr_id', 'pr_url', 'author',
            
            # Temporal
            'created_at', 'created_year', 'created_month', 'created_weekday',
            'created_hour', 'is_weekend', 'is_business_hours',
            'time_to_close_hours', 'pr_age_days',
            
            # Tamanho e complexidade
            'total_lines_changed', 'lines_per_file', 'net_lines',
            'add_delete_ratio', 'relative_pr_size', 'complexity_per_line',
            'changed_files',
            
            # Engajamento
            'total_comments', 'comments_per_line', 'comments_per_file',
            'comments_per_commit', 'has_intense_discussion',
            
            # Conteúdo
            'title_length', 'title_word_count', 'title_has_feature',
            'title_has_fix', 'title_has_refactor', 'body_length',
            'body_code_ratio', 'body_size_category',
            
            # Arquivos
            'py_files_ratio', 'java_files_ratio', 'js_files_ratio',
            'md_files_ratio', 'file_type_diversity', 'has_security_files',
            'has_financial_files', 'test_file_ratio', 'config_file_ratio',
            'is_focused_pr',
            
            # Risco
            'is_risky', 'risk_score_raw', 'risk_score_norm', 'risk_category',
            'risk_reason_count', 'risk_reasons',
            'has_risk_reason_many_files', 'has_risk_reason_touches_financial',
            'has_risk_reason_touches_security', 'has_risk_reason_many_comments',
            
            # Estado
            'state', 'merge_time_hours'
        ]
        
        # Filtrar colunas existentes
        existing_cols = [col for col in analytical_cols if col in self.df.columns]
        analytical_df = self.df[existing_cols].copy()
        
        # Adicionar colunas one-hot dinamicamente
        one_hot_prefixes = ['repo_', 'author_top_', 'month_']
        for prefix in one_hot_prefixes:
            prefix_cols = [col for col in self.df.columns if col.startswith(prefix)]
            if prefix_cols:
                analytical_df = pd.concat([analytical_df, self.df[prefix_cols]], axis=1)
        
        logger.info(f"Dataset analítico criado: {len(analytical_df)} registros, {len(analytical_df.columns)} colunas")
        return analytical_df
    
    def create_modeling_dataset(self, target_col: str = 'is_risky') -> Tuple[pd.DataFrame, pd.Series]:
        """Cria dataset otimizado para modelagem de ML"""
        logger.info(f"Criando dataset para modelagem (target: {target_col})...")
        
        if target_col not in self.df.columns:
            raise ValueError(f"Coluna alvo '{target_col}' não encontrada")
        
        # Selecionar features para modelagem
        feature_cols = self._select_modeling_features()
        
        # Garantir que target não esteja nas features
        feature_cols = [col for col in feature_cols if col != target_col]
        
        # Criar X (features) e y (target)
        X = self.df[feature_cols].copy()
        y = self.df[target_col].copy()
        
        # Tratar valores infinitos
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Preencher valores nulos restantes
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else '')
        
        logger.info(f"Dataset de modelagem criado: X={X.shape}, y={y.shape}")
        return X, y
    
    def _select_modeling_features(self) -> List[str]:
        """Seleciona features relevantes para modelagem"""
        # Excluir colunas não úteis para modelagem
        exclude_cols = [
            'repo', 'pr_id', 'pr_url', 'author',  # Identificadores
            'created_at', 'merged_at', 'closed_at',  # Datas brutas
            'risk_reasons', 'title',  # Texto livre
            'author_category', 'body_size_category',  # Categóricas já codificadas
            'risk_category'  # Derivada do target
        ]
        
        # Todas as colunas exceto as excluídas
        all_cols = list(self.df.columns)
        feature_cols = [col for col in all_cols if col not in exclude_cols]
        
        # Priorizar colunas numéricas e one-hot
        numeric_cols = self.df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        return numeric_cols
    
    def export_transformed_data(self, output_dir: str = "data/transformed"):
        """Exporta os dados transformados"""
        logger.info(f"Exportando dados transformados para {output_dir}...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Salvar dataset completo transformado
        output_file = output_path / "prs_transformed_complete.csv"
        self.df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"  Dataset completo salvo: {output_file}")
        
        # Salvar dataset analítico
        analytical_df = self.create_analytical_dataset()
        analytical_file = output_path / "prs_analytical.csv"
        analytical_df.to_csv(analytical_file, index=False, encoding='utf-8')
        logger.info(f"  Dataset analítico salvo: {analytical_file}")
        
        # Salvar dataset para modelagem
        try:
            X, y = self.create_modeling_dataset()
            modeling_path = output_path / "modeling"
            modeling_path.mkdir(exist_ok=True)
            
            X.to_csv(modeling_path / "X_features.csv", index=False)
            y.to_csv(modeling_path / "y_target.csv", index=False)
            
            logger.info(f"  Datasets para modelagem salvos em: {modeling_path}")
        except Exception as e:
            logger.warning(f"  Não foi possível salvar dataset de modelagem: {e}")
        
        # Salvar metadados
        self._save_metadata(output_path)
        
        logger.info("Exportação concluída!")
        return output_path
    
    def _save_metadata(self, output_path: Path):
        """Salva metadados sobre a transformação"""
        metadata = {
            'transformation_date': datetime.now().isoformat(),
            'original_records': len(self.df),
            'original_columns': len([c for c in self.df.columns if not c.startswith(('repo_', 'author_', 'month_'))]),
            'total_columns_after_transform': len(self.df.columns),
            'risky_prs_count': int(self.df['is_risky'].sum()),
            'risky_prs_percentage': float(self.df['is_risky'].mean() * 100),
            'repositories': self.df['repo'].unique().tolist(),
            'date_range': {
                'min': self.df['created_at'].min().isoformat() if pd.notna(self.df['created_at'].min()) else None,
                'max': self.df['created_at'].max().isoformat() if pd.notna(self.df['created_at'].max()) else None
            },
            'new_features_created': [col for col in self.df.columns if col not in [
                'repo', 'pr_id', 'pr_url', 'state', 'created_at', 'merged_at', 
                'closed_at', 'author', 'comments', 'review_comments', 'commits',
                'additions', 'deletions', 'changed_files', 'title', 'body_length',
                'has_security_terms', 'has_financial_terms', 'pr_commits',
                'pr_lines_added', 'pr_lines_removed', 'pr_files_changed',
                'pr_complexity_changes', 'test_files_changed', 'config_files_changed',
                'py_files', 'java_files', 'js_files', 'md_files', 'touches_security',
                'touches_financial', 'is_risky', 'risk_reasons', 'merge_time_hours'
            ]]
        }
        
        import json
        metadata_file = output_path / "transformation_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"  Metadados salvos: {metadata_file}")

def run_transformation_pipeline(input_file: str = None, output_dir: str = None):
    """Pipeline completo de transformação"""
    
    # Inicializar transformador
    transformer = DataTransformer(input_file or "data/pull_requests_financeiros_final.csv")
    
    try:
        # 1. Carregar dados
        transformer.load_data()
        
        # 2. Limpeza básica
        transformer.basic_cleanup()
        
        # 3. Engenharia de features
        transformer.engineer_features()
        
        # 4. Exportar dados transformados
        output_path = transformer.export_transformed_data(output_dir or "data/transformed")
        
        
        logger.info("PIPELINE DE TRANSFORMAÇÃO CONCLUÍDO COM SUCESSO!")
        logger.info(f"Dados exportados para: {output_path}")
        
        return transformer
        
    except Exception as e:
        logger.error(f"Erro no pipeline de transformação: {e}")
        logger.error(traceback.format_exc())
        raise

def generate_transformation_report(transformer: DataTransformer):
    """Gera um relatório detalhado da transformação"""
    
    logger.info("RELATÓRIO DE TRANSFORMAÇÃO")

    
    if transformer.df is None:
        logger.error("Nenhum dado disponível para relatório")
        return
    
    df = transformer.df
    
    # Estatísticas básicas
    logger.info(f"Total de registros: {len(df):,}")
    logger.info(f"Total de colunas: {len(df.columns)}")
    logger.info(f"PRs riscosos: {df['is_risky'].sum():,} ({df['is_risky'].mean()*100:.1f}%)")
    logger.info(f"Repositórios únicos: {df['repo'].nunique()}")
    
    # Novas features criadas
    new_features = [col for col in df.columns if any(col.startswith(prefix) 
                    for prefix in ['created_', 'is_', 'total_', 'has_', 'risk_', 
                                  'comments_', 'file_', 'body_', 'title_', 'lines_'])]
    
    logger.info(f"\nNovas features criadas: {len(new_features)}")
    for i, feat in enumerate(sorted(new_features), 1):
        if i <= 20:  # Mostrar apenas as primeiras 20
            logger.info(f"  {i:2}. {feat}")
        elif i == 21:
            logger.info(f"  ... e mais {len(new_features) - 20} features")
    
    # Distribuição do score de risco
    if 'risk_score_norm' in df.columns:
        logger.info(f"\nDistribuição do Score de Risco:")
        logger.info(f"  Média: {df['risk_score_norm'].mean():.2f}")
        logger.info(f"  Mediana: {df['risk_score_norm'].median():.2f}")
        logger.info(f"  Min: {df['risk_score_norm'].min():.2f}")
        logger.info(f"  Max: {df['risk_score_norm'].max():.2f}")
    
    if 'risk_category' in df.columns:
        logger.info(f"\nCategorias de Risco:")
        for category in df['risk_category'].cat.categories:
            count = (df['risk_category'] == category).sum()
            logger.info(f"  {category}: {count} PRs ({count/len(df)*100:.1f}%)")
    
    # Correlação com risco
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    top_correlations = {}
    
    for col in numeric_cols:
        if col != 'is_risky' and df[col].nunique() > 1:
            corr = df[col].corr(df['is_risky'])
            if not np.isnan(corr):
                top_correlations[col] = abs(corr)
    
    # Top 10 correlações
    top_10 = sorted(top_correlations.items(), key=lambda x: x[1], reverse=True)[:10]
    
    logger.info(f"\nTop 10 Features mais correlacionadas com risco:")
    for col, corr in top_10:
        direction = "positiva" if df[col].corr(df['is_risky']) > 0 else "negativa"
        logger.info(f"  {col}: {corr:.3f} ({direction})")
    
    
    logger.info("FIM DO RELATÓRIO")
    

# Execução principal
if __name__ == "__main__":
    # Executar pipeline completo
    transformer = run_transformation_pipeline()
    
    # Gerar relatório
    generate_transformation_report(transformer)
    
