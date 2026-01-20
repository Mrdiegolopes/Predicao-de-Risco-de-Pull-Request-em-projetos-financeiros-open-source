PIPELINE DE MODELAGEM – DETECÇÃO DE PRs ARRISCADOS

Este projeto implementa um pipeline completo de modelagem preditiva
para classificação de Pull Requests (PRs) potencialmente arriscados,
utilizando dados previamente preparados pelo pipeline de dados.

O objetivo é comparar diferentes algoritmos de classificação,
avaliar o impacto do balanceamento de classes (SMOTE vs NONE) e
selecionar automaticamente o modelo campeão com base em métricas
de desempenho.

------------------------------------------------------------
ENTRADA DO PIPELINE
------------------------------------------------------------

O pipeline espera dois arquivos CSV gerados pelo pds-data:

- X_features.csv → matriz de features numéricas
- y_target.csv   → variável alvo binária (is_risky ∈ {0,1})

Por padrão, os arquivos são lidos de:

  src/transformed/modeling/X_features.csv
  src/transformed/modeling/y_target.csv

------------------------------------------------------------
MODELOS AVALIADOS
------------------------------------------------------------

O pipeline avalia os seguintes algoritmos:

- Logistic Regression (LRC)
- Linear Support Vector Classifier (SVC)
- Random Forest Classifier (RFC)
- Gradient Boosting Classifier (GBC)
- Balanced Random Forest (BRF)
- HistGradientBoosting Classifier (HGB)
- XGBoost Classifier (XGB)

Modelos lineares (LRC e SVC) utilizam padronização de features
(StandardScaler).

------------------------------------------------------------
BALANCEAMENTO DE CLASSES
------------------------------------------------------------

São avaliados dois cenários experimentais:

- NONE  → sem balanceamento
- SMOTE → oversampling da classe minoritária

O SMOTE é aplicado dentro de um Pipeline (imblearn.pipeline.Pipeline),
evitando vazamento de dados durante a validação cruzada.

------------------------------------------------------------
METODOLOGIA EXPERIMENTAL
------------------------------------------------------------

- Validação cruzada estratificada (StratifiedKFold)
- Ajuste de hiperparâmetros via GridSearchCV
- Métrica principal de seleção: ROC-AUC
- Métricas reportadas:
  - ROC-AUC
  - F1-score
  - Precision
  - Recall
  - Accuracy

Cada combinação (modelo × balanceamento) é avaliada de forma
independente, e os resultados médios são comparados.

------------------------------------------------------------
SAÍDA DO PIPELINE
------------------------------------------------------------

Arquivos gerados:

1) metrics_comparison.csv
   → tabela comparativa com métricas médias para todos os modelos
     e modos de balanceamento

2) model_<MODELO>_<BALANCEAMENTO>.pkl
   → modelo campeão treinado no dataset completo

3) pipeline.log
   → log detalhado da execução, incluindo:
     - modelos avaliados
     - modo de balanceamento
     - ROC-AUC médio
     - hiperparâmetros selecionados

------------------------------------------------------------
COMO EXECUTAR
------------------------------------------------------------

Instale as dependências:

  pip install -r requirements.txt

Execute o pipeline:

  python3 pipeline.py

Ou explicitando os arquivos de entrada:

  python3 pipeline.py caminho/X_features.csv caminho/y_target.csv

------------------------------------------------------------
INTERPRETAÇÃO DOS RESULTADOS
------------------------------------------------------------

- O modelo campeão é aquele com maior ROC-AUC médio.
- A comparação SMOTE vs NONE permite avaliar se o balanceamento
  melhora ou piora o desempenho para cada algoritmo.
- O arquivo metrics_comparison.csv pode ser usado diretamente
  em relatórios, artigos ou apresentações.