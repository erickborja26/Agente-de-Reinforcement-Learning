import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.build_dataset import build_unified_dataset
from src.benchmarks import run_benchmark_predictions

# Configuración de rutas
RL_RESULTS_PATH = "data/rl_agent_history.csv"  
OUTPUT_IMG_PATH = "artifacts/comparativa_final.png"
OUTPUT_METRICS_PATH = "artifacts/tabla_metricas.csv"

def main():
    # 1. CARGAR DATOS (La misma fuente que usó el RL)
    print("1. Cargando Dataset Maestro...")
    df = build_unified_dataset(ticker="SPY")
    
    if df is None or df.empty:
        print("Error crítico: No hay datos.")
        return

    # Preparar X e y
    df['Target'] = df['Close'].shift(-1)   # Predecir cierre mañana
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1)) # Retorno real
    df.dropna(inplace=True)
    
    X = df.drop(columns=['Target', 'Log_Ret'])
    y = df['Target']

    # 2. EJECUTAR LOS 10 MODELOS
    print("\n2. Ejecutando Modelos Supervisados...")
    df_preds = run_benchmark_predictions(X, y)
    
    # 3. CONVERTIR PREDICCIONES A CURVAS DE EQUIDAD (DINERO)
    print("\n3. Calculando Curvas de Equidad...")
    equity = pd.DataFrame(index=df_preds.index)
    equity['Buy_Hold'] = (1 + df['Log_Ret']).cumprod()
    
    for col in df_preds.columns:
        # Señal: 1 si predice subida, 0 si no
        # Usamos shift(1) en la señal porque la predicción de hoy es para mañana
        signal = np.where(df_preds[col] > X['Close'], 1, 0)
        
        # Retorno Estrategia = Retorno Real * Señal
        equity[col] = (1 + (df['Log_Ret'] * signal)).cumprod()

    # 4. INTEGRAR AGENTE RL
    print("\n4. Buscando resultados del Agente RL...")
    if os.path.exists(RL_RESULTS_PATH):
        try:
            df_rl = pd.read_csv(RL_RESULTS_PATH)
            # Asumiendo que el script RL guardó 'Date' y 'Portfolio_Value'
            df_rl['Date'] = pd.to_datetime(df_rl['Date'])
            df_rl.set_index('Date', inplace=True)
            
            # Normalizar a base 1.0 (Retorno Acumulado)
            start_val = df_rl['Portfolio_Value'].iloc[0]
            rl_curve = df_rl['Portfolio_Value'] / start_val
            
            # Unir (Left join para respetar fechas del benchmark)
            equity = equity.join(rl_curve.rename("Agente_RL"), how='left')
            print("   -> Agente RL integrado exitosamente.")
        except Exception as e:
            print(f"   -> Error leyendo RL: {e}")
    else:
        print(f"   -> [AVISO] No se encontró {RL_RESULTS_PATH}. Se omitirá el RL.")

    # 5. GENERAR REPORTE Y GRÁFICOS
    print("\n5. Guardando resultados...")
    
    # A. Tabla de Métricas
    resumen = []
    for col in equity.columns:
        if equity[col].count() > 0:
            total_ret = (equity[col].iloc[-1] - 1) * 100
            # Sharpe anualizado simple
            daily_ret = equity[col].pct_change()
            sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
            
            resumen.append({
                "Modelo": col,
                "Retorno Total %": round(total_ret, 2),
                "Sharpe Ratio": round(sharpe, 2)
            })
    
    df_metrics = pd.DataFrame(resumen).sort_values("Retorno Total %", ascending=False)
    print(df_metrics)
    df_metrics.to_csv(OUTPUT_METRICS_PATH, index=False)
    
    # B. Gráfico
    plt.figure(figsize=(14, 8))
    
    # Graficar Buy & Hold
    plt.plot(equity['Buy_Hold'], label='Buy & Hold', color='black', ls='--', alpha=0.5)
    
    # Graficar Agente RL (Destacado)
    if 'Agente_RL' in equity.columns:
        plt.plot(equity['Agente_RL'], label='🤖 Agente RL', color='blue', linewidth=2.5)
        
    # Graficar el MEJOR modelo supervisado
    best_ml = df_metrics[~df_metrics['Modelo'].isin(['Buy_Hold', 'Agente_RL'])]['Modelo'].iloc[0]
    plt.plot(equity[best_ml], label=f'Mejor ML ({best_ml})', color='green', linewidth=2)
    
    # Graficar el resto en gris
    for col in equity.columns:
        if col not in ['Buy_Hold', 'Agente_RL', best_ml]:
            plt.plot(equity[col], color='gray', alpha=0.1)

    plt.title("Comparativa Final: RL vs Modelos Supervisados")
    plt.ylabel("Crecimiento ($1 invertido)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Guardar imagen en lugar de mostrarla (más seguro en scripts)
    os.makedirs(os.path.dirname(OUTPUT_IMG_PATH), exist_ok=True)
    plt.savefig(OUTPUT_IMG_PATH)
    print(f"\n Gráfico guardado en: {OUTPUT_IMG_PATH}")
    print(f" Métricas guardadas en: {OUTPUT_METRICS_PATH}")

if __name__ == "__main__":
    main()