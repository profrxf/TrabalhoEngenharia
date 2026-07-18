import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Configurar URI de Rastreamento Local do MLflow
# Usar SQLite local evita o erro de permissão no Windows com caracteres especiais (como 'Usuário')
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Predicao_Churn_UFG")

# 2. Ativar o rastreamento automático para scikit-learn e xgboost
mlflow.sklearn.autolog()
mlflow.xgboost.autolog()

print("--- Carregando dados gerados pelo seu pipeline original ---")
# Carrega as bases que o seu data_preparation.py original já deixou salvas
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")

# Separando variáveis preditoras (X) e a variável alvo (y)
X_train, y_train = train.drop(columns=["Churn"]), train["Churn"]
X_val, y_val = val.drop(columns=["Churn"]), val["Churn"]

# 3. Definir o Pré-processamento igual ao do seu src/train.py original
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

# Garantir que a pasta para salvar os modelos originais existe
os.makedirs('models', exist_ok=True)

# --- EXECUÇÃO 1: RANDOM FOREST ---
print("\n[MLflow] Treinando Random Forest...")
with mlflow.start_run(run_name="Random_Forest_Oficial") as run:
    # Encapsular pré-processador e estimador no Pipeline, assim como no train.py original
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
    ])
    
    # Treinar
    rf_pipeline.fit(X_train, y_train)
    
    # Avaliar e Calcular Métricas na validação
    y_pred = rf_pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    # Registrar as métricas no MLflow explicitamente
    mlflow.log_metric("val_accuracy", acc)
    mlflow.log_metric("val_precision", prec)
    mlflow.log_metric("val_recall", rec)
    mlflow.log_metric("val_f1_score", f1)
    
    print(f"Random Forest - Acurácia: {acc:.4f} | F1-Score: {f1:.4f}")

# --- EXECUÇÃO 2: XGBOOST ---
print("\n[MLflow] Treinando XGBoost...")
with mlflow.start_run(run_name="XGBoost_Oficial") as run:
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])
    
    # Treinar
    xgb_pipeline.fit(X_train, y_train)
    
    # Avaliar e Calcular Métricas na validação
    y_pred = xgb_pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    # Registrar as métricas no MLflow explicitamente
    mlflow.log_metric("val_accuracy", acc)
    mlflow.log_metric("val_precision", prec)
    mlflow.log_metric("val_recall", rec)
    mlflow.log_metric("val_f1_score", f1)
    
    print(f"XGBoost - Acurácia: {acc:.4f} | F1-Score: {f1:.4f}")

print("\n--- Experimentos concluídos com sucesso e salvos no MLflow! ---")
print("Para abrir o painel do MLflow, execute o comando: mlflow ui --backend-store-uri sqlite:///mlflow.db")
