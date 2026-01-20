PMS MODEL SERVICE (API) - pms-services

Este módulo é um "Módulo de Serviço" com acesso via API que encapsula o binário
do modelo treinado (.pkl) e executa a tarefa/problema escolhida (PMS).

PMS (Predição de Mudança de Software):
- Entrada: métricas/atributos de um arquivo em um período (features numéricas)
- Saída: previsão binária (0/1) indicando se o arquivo será alterado no futuro próximo
  (ex.: próxima release), além de um score contínuo para decisão.

------------------------------------------------------------
REQUISITOS
------------------------------------------------------------
- Python 3.9+
- Um arquivo de modelo treinado (.pkl), ex:
  model_LRC_SMOTE.pkl
  model_RFC_NONE.pkl
  etc.

Instalação de dependências:
  pip install -r requirements.txt

Sugestão de requirements para o serviço:
  fastapi
  uvicorn
  numpy
  pandas
  scikit-learn
  imbalanced-learn
  xgboost

------------------------------------------------------------
COMO RODAR O SERVIÇO
------------------------------------------------------------

1) Defina o caminho do modelo (binário .pkl):

Linux/macOS:
  export MODEL_PATH="model_LRC_SMOTE.pkl"

Windows PowerShell:
  setx MODEL_PATH "model_LRC_SMOTE.pkl"

2) (Opcional) Fixar as colunas de features esperadas pelo modelo:
   Isso é recomendado para evitar erro de mismatch de colunas.

Linux/macOS:
  export FEATURE_COLUMNS='["f1","f2","f3"]'

3) Suba a API:

  uvicorn pipeline:app --host 0.0.0.0 --port 8000

A API ficará disponível em:
  http://localhost:8000

Docs (Swagger):
  http://localhost:8000/docs

------------------------------------------------------------
ENDPOINTS
------------------------------------------------------------

GET /health
- Verifica se o modelo carregou corretamente.

GET /schema
- Mostra o MODEL_PATH, threshold padrão e (se definido) as FEATURE_COLUMNS.

POST /predict
- Predição para 1 item.

POST /predict_batch
- Predição para lista de itens.

------------------------------------------------------------
DEMO RÁPIDA (cURL)
------------------------------------------------------------

1) Health check:
  curl http://localhost:8000/health

2) Predição simples:
  curl -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{
      "features": {
        "changes_count": 12,
        "lines_added": 340,
        "lines_removed": 120,
        "complexity_sum": 55,
        "churn_total": 460
      },
      "threshold": 0.5
    }'

Resposta esperada (exemplo):
{
  "prediction": 1,
  "raw_prediction": 1,
  "score": 0.73,
  "threshold": 0.5,
  "n_features": 5,
  "model_path": "model_LRC_SMOTE.pkl"
}

------------------------------------------------------------
OBSERVAÇÕES IMPORTANTES (para a apresentação)
------------------------------------------------------------
- Este serviço encapsula o binário do modelo treinado (.pkl) e o disponibiliza via API,
  caracterizando um "Módulo de Serviço" conforme solicitado.
- A demonstração pode ser feita:
  (i) subindo o serviço com uvicorn
  (ii) chamando /health e /predict
  (iii) exibindo a resposta com prediction + score
- Para evitar inconsistência de features, recomenda-se definir FEATURE_COLUMNS com a
  mesma lista/ordem usada no treino do modelo.

------------------------------------------------------------
