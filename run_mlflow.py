"""Módulo de Rastreamento de Experimentos com MLflow (MLOps & Experiment Tracking).

Este módulo executa o treinamento dos modelos candidatos (Random Forest e XGBoost)
integrando o registro automático de parâmetros, métricas e artefatos ao MLflow
utilizando um backend SQLite local (`sqlite:///mlflow.db`).

Principais recursos:
- Registro automático (autologging) e explícito de métricas de validação.
- Paralelização multi-thread/multi-core (`n_jobs=-1`) para aceleração do treinamento.
- Correção de compatibilidade com Pandas 3+.
- Medição e registro do tempo de execução por modelo.

Padrões adotados:
- Tipagem estática (PEP 484).
- Documentação no padrão Google Style (PEP 257).
- Arquitetura funcional e modular para fácil extensibilidade.
"""

import os
from pathlib import Path
import time
from typing import Dict, Union

# Configuração do backend 'Agg' do matplotlib para execução headless (evita concorrência e crash de threads do Tkinter/Tcl)
import matplotlib
matplotlib.use("Agg")

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Silencia mensagens informativas repetitivas do MLflow CLI
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

TRACKING_URI: str = "sqlite:///mlflow.db"
EXPERIMENT_NAME: str = "Predicao_Churn_UFG"
DEFAULT_TRAIN_PATH: Path = Path("data/train.csv")
DEFAULT_VAL_PATH: Path = Path("data/val.csv")


def run_mlflow_experiments(
    train_path: Union[str, Path] = DEFAULT_TRAIN_PATH,
    val_path: Union[str, Path] = DEFAULT_VAL_PATH,
    tracking_uri: str = TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
) -> Dict[str, Dict[str, float]]:
    """Configura o MLflow, treina os modelos com paralelismo e registra os resultados.

    Args:
        train_path (Union[str, Path]): Caminho do dataset particionado de treino.
        val_path (Union[str, Path]): Caminho do dataset particionado de validação.
        tracking_uri (str): URI do servidor ou banco de rastreamento do MLflow.
        experiment_name (str): Nome do experimento agrupador no MLflow.

    Returns:
        Dict[str, Dict[str, float]]: Dicionário com métricas registradas de cada modelo.

    Raises:
        FileNotFoundError: Se as partições de treino ou validação não forem localizadas.
    """
    train_file = Path(train_path)
    val_file = Path(val_path)

    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError(
            f"Bases não encontradas em '{train_file}' ou '{val_file}'. "
            f"Execute 'src/data_preparation.py' previamente."
        )

    # 1. Configurar URI e Experimento no MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # 2. Ativar rastreamento automático para scikit-learn e xgboost
    mlflow.sklearn.autolog(log_models=True)
    mlflow.xgboost.autolog(log_models=True)

    print("--- Carregando dados gerados pelo pipeline de dados ---")
    train = pd.read_csv(train_file)
    val = pd.read_csv(val_file)

    X_train, y_train = train.drop(columns=["Churn"]), train["Churn"]
    X_val, y_val = val.drop(columns=["Churn"]), val["Churn"]

    # 3. Pré-processamento padronizado (compatível com Pandas 2 e 3)
    cat_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    models_config = [
        (
            "Random_Forest_Oficial",
            "Random Forest",
            RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1),
        ),
        (
            "XGBoost_Oficial",
            "XGBoost",
            XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1),
        ),
    ]

    summary_metrics: Dict[str, Dict[str, float]] = {}

    for run_name, display_name, estimator in models_config:
        print(f"\n[MLflow] Treinando {display_name} com suporte multi-core...")
        start_time = time.perf_counter()

        with mlflow.start_run(run_name=run_name):
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])
            pipeline.fit(X_train, y_train)

            elapsed = time.perf_counter() - start_time
            y_pred = pipeline.predict(X_val)

            metrics = {
                "val_accuracy": float(accuracy_score(y_val, y_pred)),
                "val_precision": float(precision_score(y_val, y_pred, zero_division=0)),
                "val_recall": float(recall_score(y_val, y_pred, zero_division=0)),
                "val_f1_score": float(f1_score(y_val, y_pred, zero_division=0)),
                "training_time_seconds": float(elapsed),
            }

            # Registro explícito das métricas principais de validação
            for metric_name, metric_val in metrics.items():
                mlflow.log_metric(metric_name, metric_val)

            summary_metrics[display_name] = metrics
            print(
                f"{display_name} - Acurácia: {metrics['val_accuracy']:.4f} | "
                f"F1-Score: {metrics['val_f1_score']:.4f} | "
                f"Tempo: {elapsed:.2f}s"
            )

    print("\n--- Experimentos concluídos com sucesso e salvos no MLflow! ---")
    print(f"Para abrir o painel do MLflow, execute o comando: mlflow ui --backend-store-uri {tracking_uri}")

    return summary_metrics


if __name__ == "__main__":
    run_mlflow_experiments()
