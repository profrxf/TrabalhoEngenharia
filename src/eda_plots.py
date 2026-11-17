"""Módulo de Análise Exploratória de Dados (Exploratory Data Analysis - EDA).

Este módulo gera visualizações gráficas para apoiar a compreensão do comportamento
das variáveis mais representativas e sua correlação com a evasão de clientes (Churn).

Gráficos gerados:
1. Distribuição Geral da Variável Alvo (Churn).
2. Proporção de Churn por Tipo de Contrato (Contract).
3. Incidência de Churn por Método de Pagamento (PaymentMethod).
4. Dispersão das Cobranças Mensais (MonthlyCharges) por Status de Churn.

Padrões adotados:
- Backend 'Agg' não-interativo do Matplotlib para renderização headless rápida e sem vazamento de memória.
- Tipagem estática (PEP 484).
- Documentação no padrão Google Style (PEP 257).
- Interface orientada a objetos do Matplotlib (fig, ax).
"""

from pathlib import Path
from typing import Union
import matplotlib

# Configuração do backend Agg antes de carregar o pyplot (otimização de renderização e compatibilidade headless)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DEFAULT_DATA_PATH: Path = Path("data/Telco-Customer-Churn.csv")
DEFAULT_OUTPUT_DIR: Path = Path("outputs")


def generate_eda(
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    dpi: int = 300,
) -> None:
    """Gera e salva todos os gráficos da Análise Exploratória de Dados (EDA).

    Utiliza Seaborn e Matplotlib para gerar artefatos visuais de alta qualidade,
    salvando-os no diretório especificado em formato PNG.

    Args:
        data_path (Union[str, Path]): Caminho do dataset CSV para leitura.
        output_dir (Union[str, Path]): Diretório de destino para salvar os gráficos.
        dpi (int): Resolução (dots per inch) das figuras exportadas. Padrão: 300.

    Raises:
        FileNotFoundError: Se o arquivo CSV informado não for localizado no caminho especificado.
    """
    file_path = Path(data_path)
    out_dir = Path(output_dir)

    print("--- Iniciando a Geração de Gráficos de EDA ---")
    if not file_path.exists():
        raise FileNotFoundError(
            f"Erro: O arquivo '{file_path}' não foi encontrado. "
            f"Por favor, execute o script de preparação/ingestão primeiro."
        )

    df = pd.read_csv(file_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # --- GRÁFICO 1: Distribuição de Churn ---
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Churn", hue="Churn", palette="Set2", legend=False, ax=ax)
    ax.set_title("Distribuição de Churn (Evasão de Clientes)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Quantidade de Clientes")
    fig.tight_layout()
    fig.savefig(out_dir / "distribuicao_churn.png", dpi=dpi)
    plt.close(fig)
    print(f"-> Gráfico 1 salvo: {out_dir / 'distribuicao_churn.png'}")

    # --- GRÁFICO 2: Churn por Tipo de Contrato ---
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="Contract", hue="Churn", palette="viridis", ax=ax)
    ax.set_title("Churn por Tipo de Contrato", fontsize=12, fontweight="bold")
    ax.set_xlabel("Tipo de Contrato")
    ax.set_ylabel("Quantidade de Clientes")
    ax.legend(title="Churn")
    fig.tight_layout()
    fig.savefig(out_dir / "churn_por_contrato.png", dpi=dpi)
    plt.close(fig)
    print(f"-> Gráfico 2 salvo: {out_dir / 'churn_por_contrato.png'}")

    # --- GRÁFICO 3: Churn por Método de Pagamento ---
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, y="PaymentMethod", hue="Churn", palette="magma", ax=ax)
    ax.set_title("Churn por Método de Pagamento", fontsize=12, fontweight="bold")
    ax.set_xlabel("Quantidade de Clientes")
    ax.set_ylabel("Método de Pagamento")
    ax.legend(title="Churn")
    fig.tight_layout()
    fig.savefig(out_dir / "churn_por_pagamento.png", dpi=dpi)
    plt.close(fig)
    print(f"-> Gráfico 3 salvo: {out_dir / 'churn_por_pagamento.png'}")

    # --- GRÁFICO 4: Distribuição de Cobranças Mensais por Churn ---
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn", palette="coolwarm", legend=False, ax=ax)
    ax.set_title("Distribuição de Cobranças Mensais por Status de Churn", fontsize=12, fontweight="bold")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Cobrança Mensal ($)")
    fig.tight_layout()
    fig.savefig(out_dir / "cobrancas_mensais_por_churn.png", dpi=dpi)
    plt.close(fig)
    print(f"-> Gráfico 4 salvo: {out_dir / 'cobrancas_mensais_por_churn.png'}")

    print(f"--- Todos os gráficos de EDA foram salvos na pasta '{out_dir}/'! ---")


if __name__ == "__main__":
    generate_eda()
