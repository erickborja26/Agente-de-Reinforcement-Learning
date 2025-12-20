import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 1. CONFIGURACIÓN DE RUTAS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.models.modelos import run_benchmark_predictions
except ImportError as e:
    print(f"ERROR DE IMPORTACIÓN: {e}")
    sys.exit(1)

DATA_PATH = "../data/processed/master_df.csv" 
TARGET_COL = "close"

# --- 2. TUS FUNCIONES DE MÉTRICAS FINANCIERAS ---
def sharpe_ratio(equity: pd.Series, periods: int = 252) -> float:
    """Calcula el ratio de Sharpe (Retorno / Volatilidad)"""
    r = equity.pct_change().dropna()
    if r.std() == 0 or np.isnan(r.std()):
        return 0.0
    return float(np.sqrt(periods) * (r.mean() / r.std()))

def max_drawdown(equity: pd.Series) -> float:
    """Calcula la peor caída desde un pico histórico"""
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())

def cumulative_return(equity: pd.Series) -> float:
    """Retorno total al final del periodo"""
    if len(equity) == 0: return 0.0
    return float(equity.iloc[-1] - 1.0)

# --- 3. LÓGICA DE EVALUACIÓN ---
def evaluate_models_financially(df_comparison):
    metrics = []
    
    # Precios reales
    real_next_close = df_comparison['y_real_next'] 
    current_close = df_comparison['current_close']
    
    # Retorno Porcentual Diario del Mercado (Close mañana - Close hoy) / Close hoy
    market_pct_change = (real_next_close - current_close) / current_close
    
    # Columnas de modelos
    model_cols = [c for c in df_comparison.columns if c not in ['y_real_next', 'current_close']]
    
    print(f"Evaluando métricas financieras (Sharpe, Drawdown) para {len(model_cols)} modelos...")

    for model_name in model_cols:
        preds = df_comparison[model_name]
        
        # Filtrar datos válidos
        mask = ~np.isnan(preds) & ~np.isnan(real_next_close)
        if not mask.any():
            continue

        # --- A. Métricas de Error (Regresión) ---
        rmse = np.sqrt(mean_squared_error(real_next_close[mask], preds[mask]))
        mae = mean_absolute_error(real_next_close[mask], preds[mask])
        r2 = r2_score(real_next_close[mask], preds[mask])
        
        # --- B. Simulación de Trading (Equity Curve) ---
        # Estrategia: Si Predicción > Precio Hoy -> Comprar (1.0), sino Cash (0.0)
        # Nota: Podrías usar venta corta (-1.0) si tu agente lo hace, pero asumimos Long-Only o Cash.
        signals = np.where(preds[mask] > current_close[mask], 1.0, 0.0)
        
        # Retorno de la estrategia diario
        strategy_daily_ret = signals * market_pct_change[mask]
        
        # Construir Curva de Equidad (Empezamos con 1.0)
        # (1 + r1) * (1 + r2) * ...
        equity_curve = (1 + strategy_daily_ret).cumprod()
        
        # --- C. Calcular tus Métricas ---
        cum_ret = cumulative_return(equity_curve)
        sharpe = sharpe_ratio(equity_curve)
        dd = max_drawdown(equity_curve)
        
        metrics.append({
            "Modelo": model_name,
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4),
            "Sharpe Ratio": round(sharpe, 4),
            "Max Drawdown": round(dd, 4),
            "Cumulative Return": round(cum_ret, 4)
        })
        
    return pd.DataFrame(metrics)

def main():
    print("--- INICIANDO COMPARATIVA FINANCIERA DE 10 MODELOS ---")
    
    # 1. Cargar Datos
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        abs_path = os.path.join(os.getcwd(), "data", "processed", "master_df.csv")
        if os.path.exists(abs_path):
            df = pd.read_csv(abs_path)
        else:
            print("ERROR: No se encuentra master_df.csv")
            return

    df.columns = df.columns.str.strip()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df = df.dropna()

    # 2. Preparar Predicción
    y_target = df[TARGET_COL].shift(-1)
    X_features = df.copy()
    X_features = X_features.iloc[:-1]
    y_target = y_target.iloc[:-1]
    
    # 3. Ejecutar Modelos
    print("Ejecutando validación cruzada...")
    df_preds = run_benchmark_predictions(X_features, y_target, n_splits=5)
    
    # 4. Consolidar
    df_eval = df_preds.copy()
    df_eval['y_real_next'] = y_target
    df_eval['current_close'] = X_features[TARGET_COL]
    
    # 5. Calcular Métricas Completas
    results = evaluate_models_financially(df_eval)
    
    # Ordenar por Sharpe Ratio (Mejor relación riesgo-beneficio arriba)
    results = results.sort_values(by="Sharpe Ratio", ascending=False)
    
    print("\n=== TABLA FINAL DE COMPARACIÓN (RIESGO VS RETORNO) ===")
    print(results)
    
    results.to_excel("resultados_10_modelos_financieros.xlsx", index=False)
    print("\n[OK] Guardado: resultados_10_modelos_financieros.xlsx")

if __name__ == "__main__":
    main()