"""Módulo de Inferência e Avaliação em Produção (Model Inference & Production Scoring).

Este módulo carrega o modelo vencedor persistido em disco (XGBoost) e simula a
execução em ambiente de produção utilizando a base de teste cega (`test.csv`).

Principais funcionalidades:
- Carregamento seguro de modelos serializados (`models/xgboost_model.pkl`).
- Execução de inferência em lote com mensuração de tempo e vazão (throughput).
- Cálculo e exibição das métricas finais de generalização: Acurácia, Precisão, Recall e F1-Score.
- Exportação das predições com os rótulos reais consolidados em `data/previsoes_finais.csv`.

Padrões adotados:
- Tipagem estática (PEP 484).
- Documentação no padrão Google Style (PEP 257).
- Tratamento de exceções e validação de existência dos artefatos.
"""

from pathlib import Path
import time
from typing import Dict, Tuple, Union
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

DEFAULT_TEST_PATH: Path = Path("data/test.csv")
DEFAULT_MODEL_PATH: Path = Path("models/xgboost_model.pkl")
DEFAULT_OUTPUT_PATH: Path = Path("data/previsoes_finais.csv")


def run_inference(
    test_path: Union[str, Path] = DEFAULT_TEST_PATH,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    output_path: Union[str, Path] = DEFAULT_OUTPUT_PATH,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Executa a inferência na partição de teste e calcula as métricas de produção.

    Args:
        test_path (Union[str, Path]): Caminho do dataset de teste não visto pelo modelo.
        model_path (Union[str, Path]): Caminho do arquivo pickle contendo o pipeline treinado.
        output_path (Union[str, Path]): Caminho onde o arquivo CSV de predições será salvo.

    Returns:
        Tuple[pd.DataFrame, Dict[str, float]]:
            - DataFrame de teste acrescido das colunas 'Churn_Real' e 'Churn_Predicao'.
            - Dicionário com as métricas calculadas na base de teste.

    Raises:
        FileNotFoundError: Caso o arquivo de teste ou o modelo treinado não existam.
    """
    test_file = Path(test_path)
    model_file = Path(model_path)
    out_file = Path(output_path)

    print("--- Iniciando a Inferência (Previsão) com o Modelo Vencedor ---")
    if not test_file.exists():
        raise FileNotFoundError(
            f"Arquivo de teste não encontrado: '{test_file}'. "
            f"Execute 'src/data_preparation.py' para gerar a partição."
        )

    if not model_file.exists():
        raise FileNotFoundError(
            f"Modelo treinado não encontrado: '{model_file}'. "
            f"Execute 'src/train.py' para treinar e salvar os estimadores."
        )

    # 1. Carregar os dados de teste (dados inéditos para o modelo)
    test_df = pd.read_csv(test_file)
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    # 2. Carregar o modelo vencedor
    print(f"Carregando o modelo: {model_file}")
    model = joblib.load(model_file)

    # 3. Executar as predições com mensuração de performance
    start_time = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - start_time
    throughput = len(X_test) / inference_time if inference_time > 0 else 0.0

    # 4. Avaliação Final na base de Teste
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    print("\n=========================================")
    print("  RESULTADOS FINAIS (BASE DE TESTE) - XGBOOST")
    print("=========================================")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"Tempo de inferência: {inference_time*1000:.2f} ms ({throughput:.0f} amostras/s)")
    print("=========================================")

    # 5. Salvar previsões consolidando com os atributos originais
    out_file.parent.mkdir(parents=True, exist_ok=True)
    output_df = X_test.copy()
    output_df["Churn_Real"] = y_test
    output_df["Churn_Predicao"] = y_pred

    output_df.to_csv(out_file, index=False)
    print(f"\n-> Arquivo com as previsões salvo em: {out_file}")

    return output_df, metrics


if __name__ == "__main__":
    run_inference()
