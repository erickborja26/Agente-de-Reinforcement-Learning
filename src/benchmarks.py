import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict

# Modelos
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, 
                              AdaBoostRegressor, GradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

def get_10_models():
    """
    Retorna un diccionario con los 10 Pipelines configurados.
    Aplica el concepto de 'Cadenas de Algoritmos' del Cap. 6.
    """
    pipelines = {
        # --- LÍNEA BASE ---
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
        
        # --- NO LINEALES ---
        "SVR_RBF": make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, epsilon=0.1)),
        "DecisionTree": make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=5)),
        
        # --- ENSAMBLES (BAGGING) ---
        "RandomForest": make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42)),
        "ExtraTrees": make_pipeline(StandardScaler(), ExtraTreesRegressor(n_estimators=100, random_state=42)),
        
        # --- ENSAMBLES (BOOSTING) ---
        "AdaBoost": make_pipeline(StandardScaler(), AdaBoostRegressor(n_estimators=100, random_state=42)),
        "GradientBoosting": make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=100, random_state=42)),
        "XGBoost": make_pipeline(StandardScaler(), XGBRegressor(n_estimators=100, n_jobs=-1, random_state=42)),
        
        # --- DEEP LEARNING ---
        "MLP_NeuralNet": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42))
    }
    return pipelines

def run_benchmark_predictions(X, y, n_splits=5):
    
    modelos = get_10_models()
    df_preds = pd.DataFrame(index=y.index)
    
    # TimeSeriesSplit para respetar la temporalidad (Finanzas)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    print(f"--- Iniciando Benchmarking de {len(modelos)} modelos ---")
    
    for nombre, pipeline in modelos.items():
        print(f"   > Entrenando: {nombre}...")
        try:
            # cross_val_predict genera predicciones "out-of-sample"
            preds = cross_val_predict(pipeline, X, y, cv=tscv, n_jobs=-1)
            
            # Ajustar longitud (las primeras n muestras se usan para train y no tienen predicción)
            # Rellenamos el inicio con NaN para alinear
            pad = np.full(len(y) - len(preds), np.nan)
            preds_full = np.concatenate((pad, preds))
            
            df_preds[nombre] = preds_full
            
        except Exception as e:
            print(f"   [ERROR] Falló {nombre}: {e}")
            
    return df_preds