import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 1. CONFIGURACIÓN DE RUTAS ---
# Añadimos la carpeta raíz al path para poder importar src.models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Intentamos importar la función de predicción de tus modelos
try:
    from src.models.modelos import run_benchmark_predictions
except ImportError as e:
    print(f"ERROR DE IMPORTACIÓN: {e}")
    print("Asegúrate de que existe el archivo: src/models/modelos.py")
    sys.exit(1)

# Ruta al archivo que generó run_pipeline.py
# Ajusta si tu archivo tiene otro nombre, pero según tus logs es master_df.csv
DATA_PATH = "../data/processed/master_df.csv" 
TARGET_COL = "close"

# --- 2. FUNCIONES DE EVALUACIÓN ---
def calculate_financial_metrics(df_comparison):
    """
    Calcula si los modelos ganan dinero.
    Estrategia: Si Predicción(t+1) > Precio(t) -> COMPRA (Signal=1), sino CASH (Signal=0).
    """
    metrics = []
    
    real_next_close = df_comparison['y_real_next'] 
    current_close = df_comparison['current_close']
    
    # Retorno del mercado (Log return)
    market_return = np.log(real_next_close / current_close)
    
    # Identificar columnas que son modelos (excluyendo las columnas de precios reales)
    model_cols = [c for c in df_comparison.columns if c not in ['y_real_next', 'current_close']]
    
    for model_name in model_cols:
        preds = df_comparison[model_name]
        
        # Validar datos (quitar NaNs del inicio)
        mask = ~np.isnan(preds) & ~np.isnan(real_next_close)
        if not mask.any():
            continue

        # --- Métricas de Error ---
        rmse = np.sqrt(mean_squared_error(real_next_close[mask], preds[mask]))
        mae = mean_absolute_error(real_next_close[mask], preds[mask])
        r2 = r2_score(real_next_close[mask], preds[mask])
        
        # --- Métricas Financieras ---
        # Si predice subida > precio de hoy, compramos
        signals = np.where(preds[mask] > current_close[mask], 1, 0)
        strategy_rets = signals * market_return[mask]
        cumulative_return = np.sum(strategy_rets)
        
        metrics.append({
            "Modelo": model_name,
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4),
            "Retorno Acumulado": round(cumulative_return, 4)
        })
        
    return pd.DataFrame(metrics)

def main():
    print("--- INICIANDO COMPARATIVA DE 10 MODELOS ---")
    
    # 1. Cargar Datos
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        # Intentar ruta absoluta si falla la relativa
        abs_path = os.path.join(os.getcwd(), "data", "processed", "master_df.csv")
        if os.path.exists(abs_path):
            df = pd.read_csv(abs_path)
        else:
            print(f"ERROR: No se encuentra el archivo de datos en {DATA_PATH} ni en {abs_path}")
            return

    # Limpieza de nombres de columnas
    df.columns = df.columns.str.strip()
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df = df.dropna()
    
    print(f"Datos cargados: {df.shape}")

    # 2. Preparar X e y para la predicción
    # Objetivo: Predecir el 'close' de MAÑANA (t+1) usando datos de HOY (t)
    
    y_target = df[TARGET_COL].shift(-1) # Lo que queremos predecir (futuro)
    X_features = df.copy()              # Lo que sabemos hoy
    
    # Eliminar la última fila (porque no tiene futuro conocido)
    X_features = X_features.iloc[:-1]
    y_target = y_target.iloc[:-1]
    
    # 3. Ejecutar Predicciones (Usando tu modelos.py)
    print("Ejecutando validación cruzada (esto puede tardar)...")
    df_preds = run_benchmark_predictions(X_features, y_target, n_splits=5)
    
    # 4. Consolidar para evaluación
    df_eval = df_preds.copy()
    df_eval['y_real_next'] = y_target
    df_eval['current_close'] = X_features[TARGET_COL] # Precio base de hoy
    
    # 5. Calcular Resultados
    results = calculate_financial_metrics(df_eval)
    
    # Ordenar por menor error (RMSE)
    results = results.sort_values(by="RMSE")
    
    print("\n=== TABLA DE RESULTADOS ===")
    print(results)
    
    # Guardar Excel
    results.to_excel("resultados_10_modelos.xlsx", index=False)
    print("\nArchivo guardado: resultados_10_modelos.xlsx")

if __name__ == "__main__":
    main()