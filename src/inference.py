import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_inference():
    print("--- Iniciando a Inferencia (Previsao) com o Modelo Vencedor ---")
    
    # 1. Carregar os dados de teste (dados ineditos para o modelo)
    test_df = pd.read_csv('data/test.csv')
    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']
    
    # 2. Carregar o modelo vencedor (XGBoost) que foi salvo no treino
    model_path = 'models/xgboost_model.pkl'
    print(f"Carregando o modelo: {model_path}")
    model = joblib.load(model_path)
    
    # 3. Executar as prediçoes
    y_pred = model.predict(X_test)
    
    # 4. Avaliacao Final na base de Teste (Requisito do Relatorio)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n=========================================")
    print("  RESULTADOS FINAIS (BASE DE TESTE) - XGBOOST")
    print("=========================================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=========================================")
    
    # 5. Salvar um exemplo das prediçoes junto com os dados reais
    output_df = X_test.copy()
    output_df['Churn_Real'] = y_test
    output_df['Churn_Predicao'] = y_pred
    
    output_path = 'data/previsoes_finais.csv'
    output_df.to_csv(output_path, index=False)
    print(f"\n-> Arquivo com as previsoes salvo em: {output_path}")

if __name__ == "__main__":
    run_inference()