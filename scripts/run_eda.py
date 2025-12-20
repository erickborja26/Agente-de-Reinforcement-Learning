import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Configuración de estilo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Rutas
DATA_PATH = "data/processed/master_df.csv"
OUTPUT_DIR = "reports/figures"  # Carpeta donde se guardarán las imágenes

def load_data():
    # Intenta cargar con ruta relativa o absoluta
    if os.path.exists(DATA_PATH):
        path = DATA_PATH
    elif os.path.exists(f"../{DATA_PATH}"):
        path = f"../{DATA_PATH}"
    else:
        print(f"ERROR: No se encuentra {DATA_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    return df

def check_data_quality(df):
    """Auditoría rápida de limpieza y nulos"""
    print("\n--- AUDITORÍA DE CALIDAD DE DATOS ---")
    
    # 1. Chequeo de Nulos
    nulos = df.isnull().sum()
    if nulos.sum() > 0:
        print("[ALERTA] Se encontraron valores nulos (NaN):")
        print(nulos[nulos > 0])
        print("-> RECOMENDACIÓN: Revisar run_pipeline.py o usar df.dropna() antes de entrenar.")
    else:
        print("[OK] No hay valores nulos. El dataset está limpio.")
        
    # 2. Chequeo de Infinitos
    infinitos = df.isin([np.inf, -np.inf]).sum()
    if infinitos.sum() > 0:
        print("[ALERTA] Se encontraron valores Infinitos:")
        print(infinitos[infinitos > 0])
    else:
        print("[OK] No hay valores infinitos.")
        
    # 3. Chequeo de Duplicados
    duplicados = df.index.duplicated().sum()
    if duplicados > 0:
        print(f"[ALERTA] Hay {duplicados} fechas duplicadas en el índice.")
    else:
        print("[OK] Índice temporal único (sin duplicados).")
    print("---------------------------------------\n")

def plot_price_and_returns(df):
    """1. Evolución del Precio y Retornos (Volatilidad)"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Precio
    sns.lineplot(data=df, x=df.index, y='close', ax=axes[0], color='tab:blue', linewidth=1.5)
    axes[0].set_title('Evolución del Precio de Cierre (Close)', fontsize=14)
    axes[0].set_ylabel('Precio ($)')
    
    # Retornos
    sns.lineplot(data=df, x=df.index, y='ret', ax=axes[1], color='tab:orange', linewidth=1)
    axes[1].set_title('Retornos Logarítmicos (Volatilidad diaria)', fontsize=14)
    axes[1].set_ylabel('Retorno Log')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_price_returns_history.png")
    print("Gráfico 1 guardado: Evolución temporal.")

def plot_correlation_heatmap(df):
    """2. Mapa de Calor de Correlaciones"""
    plt.figure(figsize=(12, 10))
    
    # Calculamos correlación solo de columnas numéricas
    corr = df.select_dtypes(include=[np.number]).corr()
    
    # Máscara para ver solo la mitad inferior (más limpio)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=.5)
    
    plt.title('Matriz de Correlación de Variables', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_correlation_heatmap.png")
    print("Gráfico 2 guardado: Correlaciones.")

def plot_distributions(df):
    """3. Distribución de Retornos y VIX"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histograma Retornos
    sns.histplot(df['ret'].dropna(), bins=50, kde=True, ax=axes[0], color='purple')
    axes[0].set_title('Distribución de Retornos (¿Campana de Gauss?)')
    axes[0].set_xlabel('Retorno Logarítmico')
    
    # Histograma VIX
    if 'vix' in df.columns:
        sns.histplot(df['vix'].dropna(), bins=50, kde=True, ax=axes[1], color='green')
        axes[1].set_title('Distribución de Volatilidad (VIX)')
        axes[1].set_xlabel('Índice VIX')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_distributions.png")
    print("Gráfico 3 guardado: Distribuciones.")

def plot_features_vs_target(df):
    """4. Relación Sentimiento vs Precio y Macro vs Precio"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sentimiento vs Retorno
    if 'sentiment' in df.columns:
        sns.scatterplot(data=df, x='sentiment', y='ret', alpha=0.3, ax=axes[0])
        try:
            sns.regplot(data=df, x='sentiment', y='ret', scatter=False, ax=axes[0], color='red')
        except:
            pass # Si hay error en regresión, solo mostrar scatter
        axes[0].set_title('Relación Sentimiento vs Retorno Diario')
    
    # Macro vs Precio
    macro_col = 'FR.INR.RINR' # Ajusta si tu columna se llama distinto
    if macro_col in df.columns:
        sns.scatterplot(data=df, x=macro_col, y='close', alpha=0.5, ax=axes[1], color='orange')
        axes[1].set_title(f'Indicador Macro ({macro_col}) vs Precio')
    else:
        # Fallback si no existe esa columna exacta, toma la 6ta columna si existe
        if len(df.columns) > 5:
            alt_col = df.columns[5]
            sns.scatterplot(data=df, x=alt_col, y='close', alpha=0.5, ax=axes[1], color='orange')
            axes[1].set_title(f'Feature ({alt_col}) vs Precio')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_feature_relations.png")
    print("Gráfico 4 guardado: Relaciones de variables.")

def plot_outliers_boxplots(df):
    """5. Gráfico de Cajas para detectar Outliers"""
    # Columnas clave para buscar outliers
    cols_to_plot = ['ret', 'vix', 'vol_20', 'sentiment']
    valid_cols = [c for c in cols_to_plot if c in df.columns]
    
    if not valid_cols:
        return

    fig, axes = plt.subplots(len(valid_cols), 1, figsize=(12, 4 * len(valid_cols)))
    
    if len(valid_cols) == 1:
        axes = [axes]

    for i, col in enumerate(valid_cols):
        sns.boxplot(x=df[col], ax=axes[i], color='cyan', fliersize=3, linewidth=1.5)
        axes[i].set_title(f'Detección de Outliers: {col}', fontweight='bold')
        axes[i].set_xlabel('')
        
        # IQR Info
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        axes[i].text(0.95, 0.9, f'IQR: {iqr:.4f}', transform=axes[i].transAxes, ha='right', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_outliers_boxplots.png")
    print("Gráfico 5 guardado: Boxplots de Outliers.")

def generate_stats_report(df):
    """Genera un pequeño reporte de texto con estadísticas"""
    desc = df.describe().T
    desc.to_excel(f"{OUTPUT_DIR}/estadisticas_descriptivas.xlsx")
    print(f"Reporte estadístico guardado en {OUTPUT_DIR}/estadisticas_descriptivas.xlsx")

def main():
    print("--- INICIANDO ANÁLISIS EXPLORATORIO DE DATOS (EDA) ---")
    
    # Crear carpeta si no existe
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Carpeta creada: {OUTPUT_DIR}")
        
    df = load_data()
    print(f"Datos cargados: {df.shape}")
    
    # 1. Auditoría
    check_data_quality(df)
    
    # 2. Generar Gráficos
    plot_price_and_returns(df)
    plot_correlation_heatmap(df)
    plot_distributions(df)
    plot_features_vs_target(df)
    plot_outliers_boxplots(df) # ¡Agregado!
    
    # 3. Reporte Excel
    generate_stats_report(df)
    
    print("\n[ÉXITO] Todo el análisis EDA ha sido guardado en 'reports/figures/'")

if __name__ == "__main__":
    main()