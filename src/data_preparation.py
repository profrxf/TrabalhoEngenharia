import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    print("--- Iniciando a Preparacao dos Dados ---")
    df = pd.read_csv('data/Telco-Customer-Churn.csv')
    print(f"Dados originais carregados: {df.shape[0]} linhas.")
    
    df['TotalCharges'] = df['TotalCharges'].replace(" ", None)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)
    
    print(f"-> Separacao concluida:")
    print(f"   Treino: {train_df.shape[0]} registros | Validacao: {val_df.shape[0]} | Teste: {test_df.shape[0]}")
    
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    print("-> Arquivos de treino, validacao e teste salvos em data/.")

if __name__ == "__main__":
    prepare_data()