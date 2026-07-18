import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_eda():
    print("--- Iniciando a Geracao de Graficos de EDA ---")
    
    # 1. Carregar os dados originais
    data_path = 'data/Telco-Customer-Churn.csv'
    if not os.path.exists(data_path):
        print(f"Erro: O arquivo {data_path} nao foi encontrado. Por favor, execute data_preparation.py primeiro.")
        return
        
    df = pd.read_csv(data_path)
    
    # Garantir que a pasta outputs existe
    os.makedirs('outputs', exist_ok=True)
    
    # Configurar estilo dos graficos
    sns.set_theme(style="whitegrid")
    
    # --- GRAFICO 1: Distribuicao de Churn ---
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn', hue='Churn', palette='Set2', legend=False)
    plt.title('Distribuicao de Churn (Evasao de Clientes)')
    plt.xlabel('Churn')
    plt.ylabel('Quantidade de Clientes')
    plt.tight_layout()
    plt.savefig('outputs/distribuicao_churn.png', dpi=300)
    plt.close()
    print("-> Grafico 1 salvo: outputs/distribuicao_churn.png")
    
    # --- GRAFICO 2: Churn por Tipo de Contrato ---
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Contract', hue='Churn', palette='viridis')
    plt.title('Churn por Tipo de Contrato')
    plt.xlabel('Tipo de Contrato')
    plt.ylabel('Quantidade de Clientes')
    plt.legend(title='Churn')
    plt.tight_layout()
    plt.savefig('outputs/churn_por_contrato.png', dpi=300)
    plt.close()
    print("-> Grafico 2 salvo: outputs/churn_por_contrato.png")
    
    # --- GRAFICO 3: Churn por Metodo de Pagamento ---
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, y='PaymentMethod', hue='Churn', palette='magma')
    plt.title('Churn por Metodo de Pagamento')
    plt.xlabel('Quantidade de Clientes')
    plt.ylabel('Metodo de Pagamento')
    plt.legend(title='Churn')
    plt.tight_layout()
    plt.savefig('outputs/churn_por_pagamento.png', dpi=300)
    plt.close()
    print("-> Grafico 3 salvo: outputs/churn_por_pagamento.png")

    # --- GRAFICO 4: Distribuicao de Cobrancas Mensais por Churn ---
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges', hue='Churn', palette='coolwarm', legend=False)
    plt.title('Distribuicao de Cobrancas Mensais por Status de Churn')
    plt.xlabel('Churn')
    plt.ylabel('Cobranca Mensal ($)')
    plt.tight_layout()
    plt.savefig('outputs/cobrancas_mensais_por_churn.png', dpi=300)
    plt.close()
    print("-> Grafico 4 salvo: outputs/cobrancas_mensais_por_churn.png")
    
    print("--- Todos os graficos de EDA foram salvos na pasta outputs/! ---")

if __name__ == "__main__":
    generate_eda()
