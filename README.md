# TURMA T3 - PÓS-GRADUAÇÃO SISTEMAS E AGENTES INTELIGENTES - UFG

# Aluno: RENAN XAVIER FERNANDES

---

## Relatório Técnico: Solução de Machine Learning para Predição de Churn (Evasão de Clientes)

## Como Executar o Projeto

### 1. Instalar as dependências obrigatórias
Crie e ative seu ambiente virtual (`venv`) e instale os pacotes necessários:
```bash
python -m venv .venv
# No Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Instalar requisitos:
pip install -r requirements.txt
```

### 2. Executar a Análise Exploratória de Dados (EDA)
Para gerar os gráficos de análise de dados e salvá-los na pasta `outputs/`:
```bash
python src/eda_plots.py
```

### 3. Executar o Pipeline de Machine Learning
Execute os scripts na sequência correta para preparar os dados, treinar os modelos e obter as predições finais:

```bash
# 1. Ingestão e divisão estratificada dos dados (gera train.csv, val.csv e test.csv)
python src/data_preparation.py

# 2. Treinamento e validação dos modelos (Random Forest e XGBoost)
python src/train.py

# 3. Inferência final usando o modelo vencedor na base de teste
python src/inference.py
```

### 4. Executar e Rastrear Experimentos com o MLflow
Para treinar os modelos registrando todos os hiperparâmetros, artefatos e métricas de validação de forma centralizada:

```bash
# Executa o treino integrando ao MLflow
python run_mlflow.py

# Inicia o servidor web do MLflow para análise comparativa
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Depois, abra o seu navegador no endereço: [http://localhost:5000](http://localhost:5000).

---

# Trabalho Realizado:

### 1. Introdução e Definição do Problema
Este relatório apresenta o desenvolvimento, otimização e avaliação de uma solução de Inteligência Artificial voltada para a predição de evasão de clientes (*Churn*) utilizando a base de dados pública *Telco Customer Churn*. O objetivo principal é identificar antecipadamente clientes com alta probabilidade de cancelar os serviços, permitindo que a empresa adote ações preventivas e estratégias personalizadas de retenção.

### 2. Engenharia e Preparação dos Dados
Seguindo as boas práticas do ciclo de vida de projetos de IA (MLOps), os dados brutos passaram pelas etapas de limpeza, transformação vetorizada e particionamento através do script [src/data_preparation.py](src/data_preparation.py).

A execução do pipeline de dados no terminal validou o tratamento do volume amostral:

```bash
python src/data_preparation.py
--- Iniciando a Preparação dos Dados ---
Dados originais carregados: 7043 linhas e 21 colunas.
-> Separação concluída:
   Treino: 4225 registros | Validação: 1409 | Teste: 1409
-> Arquivos de treino, validação e teste salvos em 'data/'.
```

#### Detalhamento das Etapas:
* **Volumetria e Ingestão**: O dataset original conta com 7.043 registros, superando amplamente o requisito mínimo de 1.000 linhas estipulado para a atividade.
* **Tratamento e Limpeza**: Conversão de valores na coluna `TotalCharges` com coerção de caracteres inválidos/espaços para numéricos e imputação robusta através da mediana.
* **Redução de Dimensionalidade & Ruído**: Remoção da coluna de identificação `customerID` para evitar *data leakage* e sobreajuste (*overfitting*).
* **Codificação (Encoding)**: Mapeamento binário da variável dependente `Churn` (1 para evasão, 0 para permanência).
* **Divisão Estratificada dos Dados**: Segregação estrita em três partições independentes arquivadas na pasta `data/`:
  * **Treino (4.225 registros - 60%)**: Utilizado para o aprendizado e ajuste de pesos dos algoritmos.
  * **Validação (1.409 registros - 20%)**: Utilizado para comparação de hiperparâmetros e seleção do modelo campeão.
  * **Teste (1.409 registros - 20%)**: Reservado exclusivamente para simulação cega de inferência em ambiente de produção.

### 3. Modelagem e Algoritmos Utilizados
Foram implementados, otimizados e comparados dois algoritmos distintos baseados em árvores de decisão:
1. **Random Forest (Floresta Aleatória)**: Método de Ensemble baseado em *Bagging*, robusto a ruídos e com baixa tendência a sobreajuste. Configurado com paralelismo multi-core (`n_jobs=-1`).
2. **XGBoost (Extreme Gradient Boosting)**: Algoritmo de *Gradient Boosting* sequencial que otimiza de forma agressiva a função de perda (*logloss*), comumente associado ao estado da arte em dados tabulares.

### 4. Avaliação e Comparação de Resultados
Ambos os modelos foram treinados sobre a base de treino e submetidos aos mesmos dados de validação através do script [src/train.py](src/train.py). Os resultados obtidos estão consolidados abaixo:

| Métrica de Avaliação | Random Forest (Validação) | XGBoost (Validação) | Modelo Vencedor |
|---|:---:|:---:|:---:|
| **Accuracy (Acurácia)** | 78.71% | **78.78%** | **XGBoost** |
| **Precision (Precisão)** | **63.39%** | 62.05% | Random Forest |
| **Recall (Sensibilidade)** | 49.34% | **54.35%** | **XGBoost** (+5.01%) |
| **F1-Score (Harmônica)** | 55.49% | **57.95%** | **XGBoost** (+2.46%) |
| **Tempo de Treinamento** | ~0.33s | **~0.13s** | **XGBoost** (2.5x mais rápido) |

```bash
python src/train.py
--- Iniciando o Treinamento dos Modelos ---

Treinando o modelo: Random Forest (paralelismo habilitado)...
Resultados de Random Forest na Validação (treinado em 0.33s):
   Accuracy:  0.7871
   Precision: 0.6339
   Recall:    0.4934
   F1-Score:  0.5549
-> Modelo salvo em: models\random_forest_model.pkl

Treinando o modelo: XGBoost (paralelismo habilitado)...
Resultados de XGBoost na Validação (treinado em 0.13s):
   Accuracy:  0.7878
   Precision: 0.6205
   Recall:    0.5435
   F1-Score:  0.5795
-> Modelo salvo em: models\xgboost_model.pkl
```

#### Análise das Métricas de Validação:
* **F1-Score (Métrica Global)**: Em problemas de churn, a distribuição das classes é desbalanceada (apenas ~26.5% dos clientes evadem). O XGBoost superou o Random Forest obtendo um **F1-Score de 57.95%**, consolidando-se como o melhor classificador geral.
* **Trade-off entre Precision e Recall**: O XGBoost obteve um **Recall significativamente superior (54.35% vs 49.34%)**. Na visão de negócio, capturar o maior número possível de clientes em risco real de evasão (alto Recall) é prioritário, pois o Custo de Aquisição de Clientes (CAC) é muito superior ao custo de ações preventivas de retenção.

---

### 🚀 Validação Final e Inferência (Base de Teste Cega)
Após a seleção do **XGBoost** como modelo vencedor na fase de validação, executou-se o script [src/inference.py](src/inference.py) para simular o comportamento em produção real com os 1.409 clientes da base de teste (`test.csv`):

```bash
python src/inference.py
--- Iniciando a Inferência (Previsão) com o Modelo Vencedor ---
Carregando o modelo: models\xgboost_model.pkl

=========================================
  RESULTADOS FINAIS (BASE DE TESTE) - XGBOOST
=========================================
Accuracy:  0.7977
Precision: 0.6419
Recall:    0.5335
F1-Score:  0.5827
Tempo de inferência: 47.79 ms (29481 amostras/s)
=========================================

-> Arquivo com as previsões salvo em: data\previsoes_finais.csv
```

Os resultados na partição de teste ratificaram a alta capacidade de generalização do algoritmo:
- **Acurácia:** subiu para **79.77%**.
- **Precisão:** subiu para **64.19%**.
- **F1-Score:** manteve-se estável em **58.27%**.
- **Performance de Produção:** inferência em lote executada em **47.79 ms** (vazão de mais de **29.000 clientes/segundo**), confirmando que a solução é altamente escalável para operação em tempo real ou em lote.

---

### 5. Rastreamento e MLOps com MLflow
O pipeline foi integrado ao **MLflow** via [run_mlflow.py](run_mlflow.py) utilizando banco SQLite local (`sqlite:///mlflow.db`). O rastreamento centralizado permite:
- Versionamento automático de parâmetros dos estimadores (`n_estimators`, `max_depth`, `learning_rate`, etc.).
- Registro de artefatos de visualização (curvas ROC, Matrizes de Confusão).
- Comparação histórica de métricas de validação sob o experimento `Predicao_Churn_UFG`.

---

### 6. Conclusão
O modelo **XGBoost** foi eleito o classificador oficial para produção nesta solução corporativa devido ao seu equilíbrio ótimo entre sensibilidade e precisão, somado à sua velocidade superior de treinamento e inferência. A arquitetura cumpre rigorosamente os padrões de engenharia de software para IA, segregando responsabilidades em dados (`data/`), modelos persistidos (`models/`), saídas analíticas (`outputs/`) e código-fonte modular (`src/`). As predições consolidadas em `data/previsoes_finais.csv` fornecem os insumos ideais para campanhas ativas de retenção pelo time de negócios.
