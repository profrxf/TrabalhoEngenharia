import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def train_models():
    print("--- Iniciando o Treinamento dos Modelos ---")
    
    # 1. Carregar os dados que preparamos
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')
    
    # Separar as features (X) e o alvo (y)
    X_train = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn']
    X_val = val_df.drop(columns=['Churn'])
    y_val = val_df['Churn']
    
    # 2. Identificar colunas numericas e categoricas para o Pipeline
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Pre-processamento: Normalizar numeros e codificar textos em numeros (OneHot)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
    
    # 3. Definir os dois modelos que serao comparados
    models = {
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    # Garantir que a pasta para salvar os modelos existe
    os.makedirs('models', exist_ok=True)
    
    # 4. Treinar e avaliar cada modelo
    for name, model in models.items():
        print(f"\nTreinando o modelo: {name}...")
        
        # Criar um pipeline que junta o pre-processamento e o algoritmo
        clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('classifier', model)])
        
        # Treinar o modelo
        clf_pipeline.fit(X_train, y_train)
        
        # Fazer predicoes na base de validacao
        y_pred = clf_pipeline.predict(X_val)
        
        # Calcular as metricas pedidas pelo professor
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        print(f"Resultados de {name} na Validacao:")
        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        
        # Salvar o modelo treinado em disco para reuso posterior (Inferencia)
        model_filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(clf_pipeline, model_filename)
        print(f"-> Modelo salvo em: {model_filename}")

if __name__ == "__main__":
    train_models()