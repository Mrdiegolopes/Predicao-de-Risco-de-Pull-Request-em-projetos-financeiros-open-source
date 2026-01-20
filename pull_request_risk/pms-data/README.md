PDS-DATA (PMS FULL) - PIPELINE DE PREDIÇÃO DE MUDANÇA DE SOFTWARE
(Prediction of Software Change - PMS)

Este módulo implementa um pipeline no estilo PDS para preparar um dataset
de PMS (Predição de Mudança de Software), usando histórico Git via PyDriller.

A unidade de análise é o ARQUIVO. Para cada arquivo, o pipeline extrai métricas
de mudanças em uma janela histórica (passado) e rotula se o arquivo mudou
em uma janela futura (ex.: próxima release/período).

------------------------------------------------------------
REQUISITOS
------------------------------------------------------------
- Python 3.9+
- Git instalado (para clonagem local, se necessário)

Instale dependências:
  pip install -r requirements.txt

Principais pacotes usados:
- numpy
- pandas
- pydriller

------------------------------------------------------------
COMO EXECUTAR
------------------------------------------------------------

Execute o pipeline:
  python3 pipeline.py

O pipeline executa os passos:

Step-1: Extração (RAW)
- Clona (se necessário) os repositórios em: data/repos/
- Percorre commits e gera registros por arquivo e janela temporal
- Salva o dataset bruto em:
  data/pms_raw.csv

Step-2: Transformação (TRF)
- Limpa dados, converte tipos, remove outliers (p99.5)
- Cria features derivadas e log-transform
- Salva dataset transformado em:
  data/pms_trf.csv

Step-3: Exportação para Modelagem (pds-model)
- Exporta:
  data/transformed/modeling/X_features.csv
  data/transformed/modeling/y_target.csv
- Salva também o dataset completo transformado:
  data/transformed/pms_transformed_complete.csv

------------------------------------------------------------
CONFIGURAÇÃO PRINCIPAL
------------------------------------------------------------

No início do pipeline.py:

- REPOSITORIOS: lista de repositórios GitHub (owner/repo)
- HISTORY_DAYS: tamanho da janela histórica (features)
- FUTURE_DAYS: tamanho da janela futura (label)
- MAX_WINDOWS_PER_REPO: limita janelas por repositório
- ONLY_CODE_EXTENSIONS: extensões de código consideradas (.py, .java, ...)

Exemplo de PMS:
- HISTORY_DAYS=90  (métricas dos últimos 90 dias)
- FUTURE_DAYS=30   (label: mudou nos 30 dias seguintes?)

------------------------------------------------------------
CACHE E PERFORMANCE
------------------------------------------------------------

- Commits são cacheados em:
  data/cache/<repo>_commits.json

- Repositórios clonados em:
  data/repos/<repo_name>/

O cache acelera execuções futuras.

------------------------------------------------------------
SAÍDA / LABEL
------------------------------------------------------------

Label de PMS:
- will_change = 1  → arquivo sofreu ao menos 1 modificação na janela futura
- will_change = 0  → arquivo não sofreu alterações na janela futura

------------------------------------------------------------
INTEGRAÇÃO COM PDS-MODEL E SERVICE
------------------------------------------------------------

Os arquivos X_features.csv e y_target.csv gerados aqui são a entrada do pipeline
de modelagem (pds-model). O modelo treinado (.pkl) pode ser servido via API
no módulo de serviço (FastAPI) já implementado.

------------------------------------------------------------
COMO RODAR OS TESTES
------------------------------------------------------------

Rodar testes unitários:
  python -m unittest -v test.py

Os testes não clonam repositórios reais e não fazem tráfego de rede.
PyDriller e Git clone são mockados.

------------------------------------------------------------
