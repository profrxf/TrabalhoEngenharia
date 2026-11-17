"""Módulo de Preparação e Engenharia de Dados (Data Preparation & Preprocessing).

Este módulo executa as etapas de limpeza, conversão de tipos, tratamento de valores
ausentes, codificação da variável alvo e particionamento dos dados em conjuntos
de Treino (60%), Validação (20%) e Teste (20%).

Padrões adotados:
- Tipagem estática (PEP 484).
- Documentação no padrão Google Style (PEP 257).
- Otimização com conversão vetorizada em C via `pd.to_numeric(errors='coerce')`.
- Tratamento de exceções e verificação de integridade de caminhos.
"""

from pathlib import Path
from typing import Tuple, Union
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_INPUT_PATH: Path = Path("data/Telco-Customer-Churn.csv")
DATA_DIR: Path = Path("data")


def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpeza e pré-processamento estrutural nos dados brutos.

    Transformações executadas:
    1. Conversão vetorizada de `TotalCharges` para float, convertendo espaços em branco
       em `NaN` e imputando com a mediana (robusta a outliers).
    2. Remoção do identificador de cliente `customerID` (evita vazamento de dados e ruído).
    3. Mapeamento binário da variável dependente `Churn` ('Yes' -> 1, 'No' -> 0).

    Args:
        df (pd.DataFrame): DataFrame bruto original.

    Returns:
        pd.DataFrame: DataFrame transformado e pronto para o particionamento.
    """
    processed = df.copy()

    # Otimização: conversão direta em C de valores numéricos com coerção de caracteres inválidos
    processed["TotalCharges"] = pd.to_numeric(processed["TotalCharges"], errors="coerce")
    median_total = processed["TotalCharges"].median()
    processed["TotalCharges"] = processed["TotalCharges"].fillna(median_total)

    # Remoção do identificador único (não agrega poder preditivo)
    if "customerID" in processed.columns:
        processed = processed.drop(columns=["customerID"])

    # Codificação binária da classe alvo
    if "Churn" in processed.columns:
        processed["Churn"] = processed["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    return processed


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Particiona o dataset em três conjuntos independentes: Treino, Validação e Teste.

    A proporção final padrão é 60% Treino, 20% Validação e 20% Teste:
    - 1º split: 80% (Treino + Validação) e 20% (Teste)
    - 2º split: dos 80%, separa-se 25% para Validação (0.80 * 0.25 = 0.20) e o restante (60%) para Treino.

    Args:
        df (pd.DataFrame): Dataset preparado.
        test_size (float): Proporção para a partição de teste final. Padrão: 0.20.
        val_size (float): Proporção da partição restante destinada à validação. Padrão: 0.25.
        random_state (int): Semente pseudoaleatória para reprodutibilidade. Padrão: 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames de (treino, validação, teste).
    """
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=random_state)

    return train_df, val_df, test_df


def prepare_data(
    input_path: Union[str, Path] = DEFAULT_INPUT_PATH,
    output_dir: Union[str, Path] = DATA_DIR,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Orquestra o pipeline completo de preparação e persistência dos dados.

    Args:
        input_path (Union[str, Path]): Caminho do arquivo bruto CSV.
        output_dir (Union[str, Path]): Diretório onde os arquivos resultantes serão salvos.
        random_state (int): Semente para o particionamento.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tupla com os DataFrames (treino, val, teste).

    Raises:
        FileNotFoundError: Caso o arquivo original de dados não seja localizado.
    """
    file_path = Path(input_path)
    out_dir = Path(output_dir)

    print("--- Iniciando a Preparação dos Dados ---")
    if not file_path.exists():
        raise FileNotFoundError(
            f"Arquivo de dados não encontrado: '{file_path}'. "
            f"Execute 'python data_preparation.py' para realizar a ingestão inicial."
        )

    df = pd.read_csv(file_path)
    print(f"Dados originais carregados: {df.shape[0]} linhas e {df.shape[1]} colunas.")

    cleaned_df = clean_and_transform(df)
    train_df, val_df, test_df = split_data(cleaned_df, random_state=random_state)

    print("-> Separação concluída:")
    print(
        f"   Treino: {train_df.shape[0]} registros | "
        f"Validação: {val_df.shape[0]} | "
        f"Teste: {test_df.shape[0]}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    print(f"-> Arquivos de treino, validação e teste salvos em '{out_dir}/'.")

    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare_data()
