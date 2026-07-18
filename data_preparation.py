import os
import pandas as pd

def ingest_data():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    # Cria a pasta data
    os.makedirs('data', exist_ok=True)
    
    print("Iniciando o download dos dados...")
    df = pd.read_csv(url)
    
    output_path = 'data/Telco-Customer-Churn.csv'
    df.to_csv(output_path, index=False)
    
    print("Dados baixados e salvos com sucesso!")
    return df

if __name__ == "__main__":
    ingest_data()