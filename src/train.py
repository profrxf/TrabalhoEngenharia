"""Módulo de Treinamento e Avaliação de Modelos (Model Training & Evaluation).

Este módulo implementa o pipeline de modelagem preditiva, comparando dois algoritmos
baseados em árvores de decisão: Random Forest e XGBoost.

Principais funcionalidades:
- Construção de pipelines com pré-processamento unificado (`ColumnTransformer`,
  `StandardScaler`, `OneHotEncoder`).
- Paralelização multi-core (`n_jobs=-1`) para máxima velocidade de treinamento.
- Correção de avisos de depreciação do Pandas 3+.
- Cálculo das métricas de avaliação: Acurácia, Precisão, Sensibilidade (Recall) e F1-Score.
- Persistência dos modelos treinados em disco via Joblib (`models/*.pkl`).

Padrões adotados:
- Tipagem estática (PEP 484).
- Documentação no padrão Google Style (PEP 257).
- Princípio de Responsabilidade Única (SRP) na decomposição das funções.
"""

from pathlib import Path
import time
from typing import Dict, List, Tuple, Union
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

DATA_DIR: Path = Path("data")
MODELS_DIR: Path = Path("models")


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Identifica os tipos de colunas e constrói o pré-processador de dados.

    Aplica `StandardScaler` às colunas numéricas e `OneHotEncoder` às colunas
    categóricas/strings, com tratamento seguro para categorias desconhecidas.

    Args:
        X (pd.DataFrame): DataFrame contendo apenas as variáveis preditoras (features).

    Returns:
        Tuple[ColumnTransformer, List[str], List[str]]:
            - Pré-processador Scikit-Learn configurado.
            - Lista de nomes de colunas categóricas.
            - Lista de nomes de colunas numéricas.
    """
    # Seleção compatível e à prova de depreciações no Pandas 2/3
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    return preprocessor, cat_cols, num_cols


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Calcula as quatro métricas padronizadas para validação de modelos de classificação.

    Args:
        y_true (pd.Series): Valores reais da classe alvo.
        y_pred (pd.Series): Valores previstos pelo modelo.

    Returns:
        Dict[str, float]: Dicionário com as métricas calculadas (accuracy, precision, recall, f1).
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_models(
    train_path: Union[str, Path] = DATA_DIR / "train.csv",
    val_path: Union[str, Path] = DATA_DIR / "val.csv",
    models_dir: Union[str, Path] = MODELS_DIR,
) -> Dict[str, Dict[str, Union[Pipeline, Dict[str, float]]]]:
    """Treina, avalia e salva os modelos Random Forest e XGBoost.

    Otimizações aplicadas:
    - Uso de todos os núcleos da CPU (`n_jobs=-1`) em ambos os algoritmos.
    - Medição precisa do tempo de treino de cada estimador.
    - Criação de pipeline unificado para prevenir vazamento de dados (data leakage).

    Args:
        train_path (Union[str, Path]): Caminho do CSV de treino.
        val_path (Union[str, Path]): Caminho do CSV de validação.
        models_dir (Union[str, Path]): Diretório para gravação dos artefatos serializados.

    Returns:
        Dict[str, Dict[str, Union[Pipeline, Dict[str, float]]]]:
            Dicionário com o pipeline treinado e as métricas de validação de cada modelo.

    Raises:
        FileNotFoundError: Se os arquivos de dados de treino ou validação não forem encontrados.
    """
    train_file = Path(train_path)
    val_file = Path(val_path)
    out_dir = Path(models_dir)

    print("--- Iniciando o Treinamento dos Modelos ---")
    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError(
            f"Bases de dados não encontradas em '{train_file}' ou '{val_file}'. "
            f"Execute 'src/data_preparation.py' previamente."
        )

    # 1. Carregamento dos dados particionados
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]
    X_val = val_df.drop(columns=["Churn"])
    y_val = val_df["Churn"]

    # 2. Construção do pré-processador
    preprocessor, _, _ = build_preprocessor(X_train)

    # 3. Definição dos modelos com paralelização máxima habilitada (n_jobs=-1)
    models = {
        "Random Forest": RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            n_jobs=-1,  # Utiliza todos os núcleos de CPU disponíveis
        ),
        "XGBoost": XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,  # Otimização de concorrência paralela no treinamento
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Union[Pipeline, Dict[str, float]]]] = {}

    # 4. Treinamento e avaliação iterativa
    for name, model in models.items():
        print(f"\nTreinando o modelo: {name} (paralelismo habilitado)...")
        start_time = time.perf_counter()

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        pipeline.fit(X_train, y_train)

        elapsed = time.perf_counter() - start_time
        y_pred = pipeline.predict(X_val)
        metrics = evaluate_predictions(y_val, y_pred)

        print(f"Resultados de {name} na Validação (treinado em {elapsed:.2f}s):")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1']:.4f}")

        # Salvar o modelo treinado em disco para reuso na inferência
        clean_name = name.lower().replace(" ", "_")
        model_filename = out_dir / f"{clean_name}_model.pkl"
        joblib.dump(pipeline, model_filename)
        print(f"-> Modelo salvo em: {model_filename}")

        results[name] = {"pipeline": pipeline, "metrics": metrics, "elapsed_seconds": elapsed}

    return results


if __name__ == "__main__":
    train_models()
