import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# --- IMPORTACIÓN DE LOS 10 MODELOS ---
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, 
                              AdaBoostRegressor, GradientBoostingRegressor)
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

def get_10_models():
    """
    Retorna el diccionario con los 10 Pipelines exactos para la comparativa.
    """
    pipelines = {
        # 1. Línea Base
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
        
        # 2. No Lineales
        "SVR_RBF": make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, epsilon=0.1)),
        "DecisionTree": make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=5)),
        
        # 3. Ensambles (Bagging)
        "RandomForest": make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42)),
        "ExtraTrees": make_pipeline(StandardScaler(), ExtraTreesRegressor(n_estimators=100, random_state=42)),
        
        # 4. Ensambles (Boosting)
        "AdaBoost": make_pipeline(StandardScaler(), AdaBoostRegressor(n_estimators=100, random_state=42)),
        "GradientBoosting": make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=100, random_state=42)),
        "XGBoost": make_pipeline(StandardScaler(), XGBRegressor(n_estimators=100, n_jobs=-1, random_state=42)),
        
        # 5. Deep Learning (Simple)
        "MLP_NeuralNet": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
    }
    return pipelines

def run_benchmark_predictions(X, y, n_splits=5):
    """
    Ejecuta validación cruzada MANUAL para series temporales.
    Evita el error 'cross_val_predict only works for partitions'.
    """
    modelos = get_10_models()
    # DataFrame vacío alineado al índice original
    df_preds = pd.DataFrame(index=y.index)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    print(f"--- Entrenando {len(modelos)} modelos (Bucle Manual TimeSeriesSplit) ---")
    
    for nombre, pipeline in modelos.items():
        print(f" > Modelo: {nombre}...")
        
        # Array lleno de NaNs para guardar las predicciones
        # (Los primeros datos se quedarán como NaN porque solo se usan para train)
        full_preds = np.full(len(y), np.nan)
        
        try:
            # Bucle manual sobre los folds
            for train_index, test_index in tscv.split(X):
                # Separar Train / Test para este fold
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                # Entrenar
                pipeline.fit(X_train, y_train)
                
                # Predecir
                fold_preds = pipeline.predict(X_test)
                
                # Guardar en el array global en las posiciones correctas
                full_preds[test_index] = fold_preds
            
            # Guardar columna en el DataFrame final
            df_preds[nombre] = full_preds
            
        except Exception as e:
            print(f" [ERROR] Falló {nombre}: {e}")
            df_preds[nombre] = np.nan
            
    return df_preds