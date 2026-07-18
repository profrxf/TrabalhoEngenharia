# Desenvolvimento de Solução de Machine Learning para Predição de Churn
# TURMA T3 - PÓS-GRADUAÇÃO SISTEMAS E AGENTES INTELIGENTES - UFG 

## Integrantes da Equipe
* Eduardo de Aguiar Rezende
* Regiane Mendes Barbosa

---

## Relatório Técnico: Solução de Machine Learning para Predição de Churn (Evasão de Clientes)

### 1. Introdução e Definição do Problema
Este relatório apresenta o desenvolvimento e a avaliação de uma solução de Inteligência Artificial voltada para a predição de evasão de clientes (Churn) utilizando a base de dados pública Telco Customer Churn. O objetivo principal é identificar antecipadamente clientes com alta probabilidade de cancelar os serviços, permitindo que a empresa tome ações preventivas de retenção.

### 2. Engenharia e Preparação dos Dados
Seguindo as boas práticas do ciclo de vida de projetos de IA, os dados brutos passaram pelas etapas de limpeza, transformação e particionamento através do script [data_preparation.py](file:///c:/Users/Usu%C3%A1rio/Documents/PSAI2026_UFG/ESIA%20-%20Altino/projeto_churn/src/data_preparation.py).
A execução do pipeline de dados no terminal validou o tratamento do volume amostral em conformidade com as regras de negócio:

```bash
(venv) PS C:\Users\Usuário\Documents\PSAI2026_UFG\ESIA - Altino\projeto_churn> python .\src\data_preparation.py
--- Iniciando a Preparacao dos Dados ---
Dados originais carregados: 7043 linhas.
-> Separacao concluida:
   Treino: 4225 registros | Validacao: 1409 | Teste: 1409
-> Arquivos de treino, validacao e teste salvos em data/.
```

#### Detalhamento das Etapas:
* **Volumetria e Ingestão**: O dataset original conta com 7.043 registros, superando amplamente o requisito mínimo de 1.000 linhas estipulado para a atividade.
* **Tratamento e Limpeza**: Identificação de valores ausentes (como espaços em branco na coluna de faturamento total `TotalCharges`) e conversão para tipos numéricos adequados.
* **Codificação (Encoding)**: Transformação de variáveis categóricas (gênero, tipo de contrato, forma de pagamento) em formato numérico legível por algoritmos.
* **Divisão Estratificada dos Dados**: Para garantir uma avaliação justa e evitar o *data leakage*, o conjunto foi dividido de forma estrita em três partições independentes arquivadas na pasta `data/`:
  * **Treino (4.225 registros)**: Utilizado para o ajuste dos pesos e aprendizado dos algoritmos (60%).
  * **Validação (1.409 registros)**: Utilizado para o ajuste de hiperparâmetros e comparação inicial de desempenho (20%).
  * **Teste (1.409 registros)**: Reservado exclusivamente para a simulação final do modelo em ambiente de produção (inferência) (20%).

### 3. Modelagem e Algoritmos Utilizados
Conforme os requisitos mínimos estabelecidos, foram implementados e comparados dois algoritmos distintos baseados em árvores de decisão:
1. **Random Forest (Floresta Aleatória)**: Um método de Ensemble baseado em Bagging, conhecido por sua robustez e baixa tendência ao overfitting.
2. **XGBoost (Extreme Gradient Boosting)**: Um método baseado em Boosting sequencial que otimiza de forma agressiva a função de perda, comumente associado a alta performance em dados estruturados.

### 4. Avaliação e Comparação de Resultados
Ambos os modelos foram submetidos inicialmente aos mesmos dados de validação para a tomada de decisão. Os resultados obtidos no terminal após a execução do script [train.py](file:///c:/Users/Usu%C3%A1rio/Documents/PSAI2026_UFG/ESIA%20-%20Altino/projeto_churn/src/train.py) estão consolidados na tabela abaixo:

| Métrica de Avaliação | Random Forest (Validação) | XGBoost (Validação) | Melhor Desempenho |
|---|---|---|---|
| Accuracy (Acurácia) | 78.71% | 78.78% | XGBoost (Empate Técnico) |
| Precision (Precisão) | 63.39% | 62.05% | Random Forest |
| Recall (Sensibilidade) | 49.34% | 54.35% | XGBoost |
| F1-Score | 55.49% | 57.95% | XGBoost |

#### Análise das Métricas de Validação:
* **F1-Score (Métrica Global)**: Em problemas de churn, as classes costumam ser altamente desbalanceadas (menos pessoas cancelam do que renovam). O XGBoost superou o Random Forest obtendo um F1-Score de 57.95%, consolidando-se como a melhor solução geral.
* **Trade-off entre Precision e Recall**: O XGBoost obteve um Recall significativamente superior (54.35% contra 49.34%). Na visão de negócio, capturar o maior número possível de clientes em risco de evasão (alto Recall) é prioritário, pois o custo de perder um cliente permanente costuma ser muito superior ao custo de ofertar uma campanha preventiva por engano.

#### 🚀 Validação Final e Inferência (Base de Teste)
Após a seleção do XGBoost como o modelo vencedor na fase de validação, o script [inference.py](file:///c:/Users/Usu%C3%A1rio/Documents/PSAI2026_UFG/ESIA%20-%20Altino/projeto_churn/src/inference.py) foi executado para avaliar o comportamento do algoritmo em um cenário simulado de produção real, utilizando o conjunto de dados de teste (`test.csv`) — dados inéditos que nunca haviam sido expostos ao modelo.
O retorno oficial do terminal confirmou o sucesso da generalização:

```bash
PS C:\Users\Usuário\Documents\PSAI2026_UFG\ESIA - Altino\projeto_churn> python .\src\inference.py
--- Iniciando a Inferencia (Previsao) com o Modelo Vencedor ---
Carregando o modelo: models/xgboost_model.pkl

=========================================
  RESULTADOS FINAIS (BASE DE TESTE) - XGBOOST
=========================================
Accuracy:  0.7977
Precision: 0.6419
Recall:    0.5335
F1-Score:  0.5827
=========================================

-> Arquivo com as previsoes salvo em: data/previsoes_finais.csv
```

Os resultados na base de teste ratificaram a robustez do algoritmo. A acurácia final subiu para 79.77% e o F1-Score se manteve estável em 58.27%, provando estatisticamente que o modelo não sofreu de overfitting (sobreajuste) e está pronto para o uso estratégico de negócio.

### 5. Conclusão
O modelo XGBoost foi selecionado como o modelo oficial de produção para esta solução devido ao seu equilíbrio superior entre capturar clientes propensos à evasão e manter uma taxa aceitável de assertividade. O projeto cumpre rigorosamente os padrões de engenharia de software para IA, segregando as responsabilidades de dados (`data/`), modelos (`models/`) e códigos de execução (`src/`). As saídas geradas em `data/previsoes_finais.csv` fornecem os insumos necessários para que o time de marketing adote ações diretas de retenção de clientes.

---

## Como Executar o Projeto

### 1. Instalar as dependências obrigatórias
Crie e ative seu ambiente virtual (`venv`) e instale os pacotes necessários:
```bash
pip install -r requirements.txt
```

### 2. Executar a Análise Exploratória de Dados (EDA)
Para gerar os gráficos de análise de dados e salvá-los na pasta `outputs/`:
```bash
python src/eda_plots.py
```

### 3. Executar o Pipeline Original de Machine Learning
Execute os scripts na sequência correta para baixar os dados, tratar, treinar os modelos e obter as predições finais:

```bash
# 1. Ingestão e divisão dos dados (gera train.csv, val.csv e test.csv)
python src/data_preparation.py

# 2. Treinamento dos modelos (Random Forest e XGBoost)
python src/train.py

# 3. Inferência final usando o modelo vencedor na base de teste
python src/inference.py
```

### 4. Executar e Rastrear Experimentos com o MLflow
Para treinar os modelos registrando todos os hiperparâmetros, artefatos de visualização e métricas de validação de forma centralizada:

```bash
# Executa o treino integrando ao MLflow
python run_mlflow.py

# Abre a interface visual do MLflow para análise comparativa
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Depois, abra o seu navegador no endereço: [http://localhost:5000](http://localhost:5000).