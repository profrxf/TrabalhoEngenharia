"""Módulo de Ingestão de Dados (Data Ingestion).

Este módulo é responsável por realizar a ingestão do dataset público Telco Customer Churn
a partir do repositório de dados oficial da IBM, armazenando-o no diretório local de dados.

Padrões adotados:
- Tipagem estática (PEP 484).
- Documentação no padrão Google Style (PEP 257).
- Otimização de I/O com cache local para evitar downloads redundantes.
"""

from pathlib import Path
from typing import Union
import pandas as pd

DATA_URL: str = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
DEFAULT_OUTPUT_DIR: Path = Path("data")
DEFAULT_OUTPUT_FILE: Path = DEFAULT_OUTPUT_DIR / "Telco-Customer-Churn.csv"


def ingest_data(
    url: str = DATA_URL,
    output_path: Union[str, Path] = DEFAULT_OUTPUT_FILE,
    force_download: bool = False,
) -> pd.DataFrame:
    """Realiza o download e a ingestão do dataset bruto de Churn.

    Verifica se o arquivo já existe localmente para otimizar tempo de execução e
    tráfego de rede. Caso já exista e `force_download` seja False, carrega
    diretamente do disco.

    Args:
        url (str): URL de origem do arquivo CSV bruto. Padrão: IBM GitHub Repo.
        output_path (Union[str, Path]): Caminho de destino para salvar o arquivo CSV.
        force_download (bool): Se True, força o download mesmo que o arquivo já exista.

    Returns:
        pd.DataFrame: DataFrame contendo os dados brutos carregados.

    Raises:
        urllib.error.URLError: Se houver falha de conexão durante o download.
        pd.errors.EmptyDataError: Se o arquivo baixado estiver vazio.
    """
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not force_download:
        print(f"-> Arquivo já existente em cache: {target_path}. Carregando localmente...")
        return pd.read_csv(target_path)

    print(f"-> Iniciando o download dos dados a partir de:\n   {url}")
    df = pd.read_csv(url)

    df.to_csv(target_path, index=False)
    print(f"-> Download concluído com sucesso ({df.shape[0]} linhas, {df.shape[1]} colunas).")
    print(f"-> Arquivo salvo em: {target_path}")

    return df


if __name__ == "__main__":
    ingest_data()
