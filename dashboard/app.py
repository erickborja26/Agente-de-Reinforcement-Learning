import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
import os
from typing import Optional
import joblib

# Importar utilidades
from utils import (
    load_master_df,
    load_excel_results,
    load_images_from_directory,
    calculate_kpis,
    get_regime_colors
)

# ==============================================================================
# CONFIGURACIÓN DE STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Trading RL + HMM Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS para mejor presentación
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .title-section {
        color: #1f77b4;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# VARIABLES GLOBALES DE RUTAS
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "master_df.csv"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_EXCEL = PROJECT_ROOT / "resultados_10_modelos_financieros.xlsx"
HMM_TIMESERIES = ARTIFACTS_DIR / "hmm" / "hmm_probabilities_timeseries.csv"
RL_MODEL_HMM = ARTIFACTS_DIR / "rl" / "dqn_hmm.zip"
RL_SCALER_PATH = ARTIFACTS_DIR / "scalers" / "rl_scaler.joblib"
ARTIFACT_IMAGE_EXTS = (".png", ".jpg", ".jpeg")

BASE_FEATURES = [
    "ret",
    "vol_20",
    "mom_5",
    "FP.CPI.TOTL.ZG",
    "FR.INR.RINR",
    "PA.NUS.FCRF",
    "vix",
    "sentiment_7d",
    "news_7d"
]
HMM_PROB_COLS = [f"hmm_p{k}" for k in range(3)]
ACTION_LABELS = {
    0: "Hold (mantener)",
    1: "Buy (ir long)",
    2: "Sell (cerrar/neutral)"
}


@st.cache_data
def load_hmm_probabilities(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
        return df
    except Exception as exc:
        st.error(f"No se pudo cargar las probabilidades HMM: {exc}")
        return None


@st.cache_data
def load_rl_dataframe() -> Optional[pd.DataFrame]:
    base_df = load_master_df(DATA_PROCESSED)
    hmm_df = load_hmm_probabilities(HMM_TIMESERIES)
    if base_df is None or base_df.empty:
        return None
    if hmm_df is None or hmm_df.empty:
        return None
    needed_cols = ["hmm_state"] + HMM_PROB_COLS
    for col in needed_cols:
        if col not in hmm_df.columns:
            return None

    merged = base_df.join(hmm_df[needed_cols], how="inner")
    return merged.sort_index()


@st.cache_resource
def load_rl_artifacts():
    try:
        from stable_baselines3 import DQN
    except ImportError:
        st.error("Falta stable-baselines3. Instala las dependencias del proyecto antes de usar el agente RL.")
        st.stop()

    if not RL_MODEL_HMM.exists():
        st.error(f"No se encontrÇü el modelo RL entrenado: {RL_MODEL_HMM}")
        st.stop()
    if not RL_SCALER_PATH.exists():
        st.error(f"No se encontrÇü el scaler del agente RL: {RL_SCALER_PATH}")
        st.stop()

    model = DQN.load(str(RL_MODEL_HMM))
    scaler = joblib.load(RL_SCALER_PATH)
    return model, scaler


@st.cache_data
def list_artifact_files(root: Path):
    root = Path(root)
    files = []
    if not root.exists():
        return files
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root)
            files.append({
                "path": path,
                "rel": str(rel),
                "ext": path.suffix.lower(),
                "size_kb": round(path.stat().st_size / 1024, 1)
            })
    return files


@st.cache_data
def load_artifact_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        # Parse columnas tipo fecha si aplican
        date_cols = [c for c in df.columns if "date" in c.lower()]
        for c in date_cols:
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:
                pass
        return df
    except Exception as exc:
        st.error(f"No se pudo leer el CSV {path}: {exc}")
        return None


@st.cache_data
def load_artifact_image(path: Path) -> Optional[Image.Image]:
    try:
        return Image.open(path)
    except Exception as exc:
        st.error(f"No se pudo cargar la imagen {path}: {exc}")
        return None

# ==============================================================================
# SIDEBAR - NAVEGACIÓN
# ==============================================================================
st.sidebar.markdown("# 📊 Trading Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "🏠 Home",
        "📈 EDA",
        "🏆 Comparativa de Modelos",
        "Agente RL + HMM",
        "📂 Artifacts",
        "🔄 Simulador HMM",
        "📋 Detalles Técnicos"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Agente de Reinforcement Learning con HMM**\n\n"
    "Trading del ETF EPU (iShares MSCI Peru ETF)\n\n"
    "Combina: RL (DQN) + HMM (Regímenes de Mercado)"
)

# ==============================================================================
# PÁGINA 1: HOME / RESUMEN
# ==============================================================================
if page == "🏠 Home":
    st.markdown('<div class="title-section">🏠 Home - Resumen Ejecutivo</div>', unsafe_allow_html=True)
    
    # Descripción del proyecto
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Descripción General
        
        Este proyecto implementa un **agente de Reinforcement Learning (DQN)** que integra 
        un **Hidden Markov Model (HMM)** para identificar regímenes ocultos del mercado y 
        tomar decisiones de trading sobre el **EPU (iShares MSCI Peru ETF)**.
        
        ### Objetivo Principal
        Evaluar si la incorporación de estados latentes (HMM) mejora el desempeño de un 
        agente de RL comparándolo contra un agente sin esta información.
        
        ### Fuentes de Datos Integradas
        - 📊 **Yahoo Finance**: Precios históricos del EPU
        - 🌍 **World Bank**: Indicadores macroeconómicos del Perú
        - 📉 **VIX**: Volatilidad histórica
        - 📰 **Alpha Vantage**: Análisis de sentimiento de noticias
        """)
    
    with col2:
        st.markdown("""
        ### Tecnologías
        - Python 3.11
        - Scikit-learn (HMM)
        - Stable-Baselines3 (DQN)
        - Gymnasium (Entorno RL)
        - Streamlit (Dashboard)
        """)
    
    st.markdown("---")
    
    # Cargar datos y calcular KPIs
    try:
        df = load_master_df(DATA_PROCESSED)
        kpis = calculate_kpis(df)
        
        # Mostrar KPIs en tarjetas
        st.markdown("## 📈 Indicadores Clave (KPIs)")
        
        kpi_cols = st.columns(4)
        
        with kpi_cols[0]:
            st.metric(
                "💰 Último Precio",
                f"${kpis['last_price']:.2f}",
                f"{kpis['price_change_pct']:.2f}%"
            )
        
        with kpi_cols[1]:
            st.metric(
                "📊 Volatilidad (20d)",
                f"{kpis['volatility_20d']:.4f}",
                "Anualizada"
            )
        
        with kpi_cols[2]:
            st.metric(
                "💹 Retorno Acumulado",
                f"{kpis['cumulative_return']:.2f}%",
                "Período completo"
            )
        
        with kpi_cols[3]:
            st.metric(
                "📅 Días de Datos",
                f"{kpis['total_days']}",
                f"Desde {kpis['start_date']} a {kpis['end_date']}"
            )
        
        st.markdown("---")
        
        # Gráfico de precio histórico
        st.markdown("## 📉 Evolución del Precio")
        
        fig = px.line(
            df,
            x=df.index,
            y='close',
            title='Precio de Cierre Histórico (EPU)',
            labels={'close': 'Precio ($)', 'index': 'Fecha'},
            template='plotly_white'
        )
        fig.update_layout(hovermode='x unified', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error al cargar datos: {e}")

# ==============================================================================
# PÁGINA 2: ANÁLISIS EDA (CORREGIDO)
# ==============================================================================
elif page == "📈 EDA":
    st.markdown('<div class="title-section">📈 Análisis Exploratorio de Datos (EDA)</div>', unsafe_allow_html=True)
    
    try:
        images = load_images_from_directory(REPORTS_DIR)
        
        if not images:
            st.warning("⚠️ No se encontraron imágenes en la carpeta de reportes.")
            st.info(f"Ruta esperada: `{REPORTS_DIR}`")
        else:
            st.markdown(f"**Se encontraron {len(images)} gráficos de análisis**")
            
            # Selector para elegir qué gráfico ver
            selected_image = st.selectbox(
                "Selecciona un gráfico para visualizar:",
                list(images.keys()),
                index=0
            )
            
            # Mostrar imagen seleccionada
            st.image(
                images[selected_image],
                caption=selected_image,
                use_container_width=True  # <--- CAMBIO AQUÍ
            )
            
            st.markdown("---")
            
            # Mostrar todas las imágenes en grid (opcional)
            if st.checkbox("📸 Mostrar todos los gráficos en grid"):
                cols = st.columns(2)
                for idx, (name, img) in enumerate(images.items()):
                    with cols[idx % 2]:
                        st.image(img, caption=name, use_container_width=True) # <--- CAMBIO AQUÍ
    
    except Exception as e:
        st.error(f"❌ Error al cargar gráficos: {e}")

# ==============================================================================
# PÁGINA 3: COMPARATIVA DE MODELOS (CON LECTURA AUTOMÁTICA DE METRICS.CSV)
# ==============================================================================
elif page == "🏆 Comparativa de Modelos":
    st.markdown('<div class="title-section">🏆 Benchmark: Supervisados vs RL</div>', unsafe_allow_html=True)
    
    try:
        # 1. Cargar modelos supervisados (Excel)
        df_results = load_excel_results(RESULTS_EXCEL)
        if df_results is not None:
            df_results["Tipo"] = "Supervisado"
        
        # 2. CARGAR MÉTRICAS REALES DE RL (metrics.csv)
        metrics_csv_path = ARTIFACTS_DIR / "metrics.csv"
        df_rl = pd.DataFrame()

        if metrics_csv_path.exists():
            try:
                # Leemos el CSV
                df_rl_raw = pd.read_csv(metrics_csv_path)
                
                # --- LIMPIEZA Y ADAPTACIÓN DE COLUMNAS ---
                # A veces el CSV guarda el nombre del modelo en la primera columna sin nombre o como index
                # Intentamos detectar la columna del nombre
                if 'Unnamed: 0' in df_rl_raw.columns:
                    df_rl_raw = df_rl_raw.rename(columns={'Unnamed: 0': 'Modelo'})
                elif 'Model' in df_rl_raw.columns:
                    df_rl_raw = df_rl_raw.rename(columns={'Model': 'Modelo'})
                
                # Si no hay columna 'Modelo', asumimos que es el índice y lo reseteamos
                if 'Modelo' not in df_rl_raw.columns:
                     # Si la primera columna parece ser texto (nombres de modelos), la usamos
                     if df_rl_raw.iloc[:, 0].dtype == object:
                         df_rl_raw = df_rl_raw.rename(columns={df_rl_raw.columns[0]: 'Modelo'})
                
                # Normalizar nombres de métricas (Mapeo flexible)
                # Tu dashboard espera: 'Sharpe Ratio', 'Cumulative Return', 'Max Drawdown'
                # Tu CSV puede tener: 'Sharpe', 'Cumulative Return', 'Max Drawdown' (según tu log anterior)
                column_mapping = {
                    'Sharpe': 'Sharpe Ratio',
                    'sharpe': 'Sharpe Ratio',
                    'Sharpe Ratio': 'Sharpe Ratio',
                    'Cumulative Return': 'Cumulative Return',
                    'cumulative_return': 'Cumulative Return',
                    'Max Drawdown': 'Max Drawdown',
                    'max_drawdown': 'Max Drawdown'
                }
                df_rl_raw = df_rl_raw.rename(columns=column_mapping)
                
                # Seleccionar solo las columnas necesarias y asignar Tipo
                required_cols = ['Modelo', 'Sharpe Ratio', 'Cumulative Return', 'Max Drawdown']
                available_cols = [c for c in required_cols if c in df_rl_raw.columns]
                
                if 'Modelo' in available_cols:
                    df_rl = df_rl_raw[available_cols].copy()
                    df_rl["Tipo"] = "Reinforcement Learning"
                    
                    # Asegurar que sean numéricos
                    cols_to_numeric = ['Sharpe Ratio', 'Cumulative Return', 'Max Drawdown']
                    for col in cols_to_numeric:
                        if col in df_rl.columns:
                            df_rl[col] = pd.to_numeric(df_rl[col], errors='coerce')
                    
                    st.success(f"✅ Métricas de RL cargadas correctamente desde {metrics_csv_path.name}")
                else:
                    st.error(f"⚠️ El archivo metrics.csv existe pero no se pudo identificar la columna de 'Modelo'. Columnas encontradas: {list(df_rl_raw.columns)}")
            
            except Exception as e:
                st.error(f"❌ Error leyendo metrics.csv: {e}")
        else:
            st.warning(f"⚠️ No se encontró el archivo: {metrics_csv_path}. Verifica que run_pipeline.py lo haya generado en 'artifacts/'.")

        # 3. UNIR DATOS (Supervisado + RL)
        if not df_rl.empty and df_results is not None:
            df_final = pd.concat([df_results, df_rl], ignore_index=True)
        elif not df_rl.empty:
            df_final = df_rl
        elif df_results is not None:
            df_final = df_results
        else:
            df_final = pd.DataFrame()

        # 4. VISUALIZACIÓN
        if not df_final.empty:
            st.markdown("### 📊 Tabla General de Resultados")
            
            # Ordenar
            if 'Sharpe Ratio' in df_final.columns:
                df_sorted = df_final.sort_values(by="Sharpe Ratio", ascending=False)
            else:
                df_sorted = df_final
            
            # Formatear
            st.dataframe(
                df_sorted.style.format({
                    "Sharpe Ratio": "{:.4f}",
                    "Max Drawdown": "{:.2%}",
                    "Cumulative Return": "{:.2%}"
                }, na_rep="-"),
                use_container_width=True,
                height=400
            )
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            # GRÁFICO 1: Retornos
            if 'Cumulative Return' in df_final.columns:
                with col1:
                    st.markdown("### 📈 Retorno Acumulado")
                    fig1 = px.bar(
                        df_sorted,
                        x='Modelo',
                        y='Cumulative Return',
                        color='Tipo',
                        color_discrete_map={"Supervisado": "#1f77b4", "Reinforcement Learning": "#ff7f0e"},
                        title='Comparativa de Retornos',
                        text_auto='.1%'
                    )
                    fig1.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig1, use_container_width=True)
            
            # GRÁFICO 2: Drawdown
            if 'Max Drawdown' in df_final.columns:
                with col2:
                    st.markdown("### 🛡️ Gestión de Riesgo")
                    fig2 = px.bar(
                        df_sorted.sort_values(by="Max Drawdown", ascending=False),
                        x='Modelo',
                        y='Max Drawdown',
                        color='Tipo',
                        color_discrete_map={"Supervisado": "#1f77b4", "Reinforcement Learning": "#ff7f0e"},
                        title='Caída Máxima (Drawdown)',
                        text_auto='.1%'
                    )
                    fig2.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig2, use_container_width=True)
            
            # GRÁFICO 3: Scatter
            if 'Sharpe Ratio' in df_final.columns:
                st.markdown("### ⚖️ Mapa de Riesgo vs Retorno")
                fig3 = px.scatter(
                    df_final,
                    x='Max Drawdown',
                    y='Cumulative Return',
                    color='Sharpe Ratio',
                    symbol='Tipo',
                    hover_name='Modelo',
                    color_continuous_scale='RdYlGn',
                    title='Frontera de Eficiencia',
                )
                fig3.update_traces(marker=dict(size=15, line=dict(width=1, color='DarkSlateGrey')))
                fig3.add_hline(y=0, line_dash="dash", line_color="gray")
                fig3.add_vline(x=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig3, use_container_width=True)

        else:
            st.warning("⚠️ No hay datos válidos para mostrar en la comparativa.")

    except Exception as e:
        st.error(f"❌ Error general en la comparativa: {e}")

# ==============================================================================
# PÁGINA 4: AGENTE RL + HMM (DECISIÓN)
# ==============================================================================
elif page == "Agente RL + HMM":
    st.markdown('<div class="title-section">Agente RL + HMM - Decide Buy / Sell / Hold</div>', unsafe_allow_html=True)
    st.markdown(
        "Prueba rápidamente el agente entrenado (DQN + probabilidades del HMM) "
        "para una fecha específica y obtén la acción sugerida."
    )

    rl_df = load_rl_dataframe()
    if rl_df is None or rl_df.empty:
        st.error(
            "No se pudieron cargar los datos necesarios. Verifica que existan "
            "`data/processed/master_df.csv` y `artifacts/hmm/hmm_probabilities_timeseries.csv`."
        )
    else:
        missing_cols = [c for c in BASE_FEATURES + HMM_PROB_COLS if c not in rl_df.columns]
        if missing_cols:
            st.error(f"Faltan columnas requeridas para el agente RL: {missing_cols}")
        else:
            model, scaler = load_rl_artifacts()

            min_date = rl_df.index.min().date()
            max_date = rl_df.index.max().date()
            chosen_date = st.date_input(
                "Fecha a evaluar",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                help="Selecciona el día (dentro del rango de datos) para obtener la decisión del agente."
            )
            ts_date = pd.Timestamp(chosen_date)

            if ts_date not in rl_df.index:
                st.warning("No hay datos para esa fecha (posible fin de semana o feriado). Prueba otra fecha.")
            else:
                row = rl_df.loc[ts_date]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]

                try:
                    base_scaled = scaler.transform(row[BASE_FEATURES].values.reshape(1, -1))[0]
                except Exception as exc:
                    st.error(f"No se pudieron escalar las features: {exc}")
                    st.stop()

                obs = np.concatenate([base_scaled, row[HMM_PROB_COLS].values.astype(float)])
                action, _ = model.predict(obs, deterministic=True)
                action_int = int(action)
                decision = ACTION_LABELS.get(action_int, f"Acción {action_int}")

                top_state_idx = int(row[HMM_PROB_COLS].values.argmax())
                top_state_prob = float(row[HMM_PROB_COLS[top_state_idx]])

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Decisión del agente", decision)
                with col_b:
                    if "close" in row.index:
                        st.metric("Precio de cierre", f"${row['close']:.2f}")
                with col_c:
                    st.metric("Régimen más probable", f"Estado {top_state_idx}", f"{top_state_prob:.2%}")

                st.markdown("### Probabilidades HMM del día")
                prob_df = pd.DataFrame({
                    "Probabilidad": row[HMM_PROB_COLS].values
                }, index=[f"Estado {i}" for i in range(len(HMM_PROB_COLS))])
                st.bar_chart(prob_df)

                st.markdown("### Features utilizadas por el agente")
                features_view = pd.DataFrame({
                    "Feature": BASE_FEATURES,
                    "Valor (sin escalar)": row[BASE_FEATURES].values,
                    "Valor escalado": base_scaled
                })
                st.dataframe(features_view, use_container_width=True)

# ==============================================================================
# PÁGINA 5: EXPLORADOR DE ARTIFACTS
# ==============================================================================
elif page == "📂 Artifacts":
    st.markdown('<div class="title-section">📂 Artefactos del proyecto</div>', unsafe_allow_html=True)
    st.markdown("Explora los archivos generados en `artifacts/`: CSV, imágenes, modelos y otros.")

    artifacts = list_artifact_files(ARTIFACTS_DIR)
    if not artifacts:
        st.warning("No hay archivos en la carpeta artifacts.")
    else:
        tab_csv, tab_imgs, tab_models, tab_other = st.tabs(["CSV / Series", "Imágenes", "Modelos y scalers", "Otros"])

        # --- CSV ---
        with tab_csv:
            csv_files = [a for a in artifacts if a["ext"] == ".csv"]
            if not csv_files:
                st.info("No se encontraron CSV en artifacts.")
            else:
                names = [f"{c['rel']} ({c['size_kb']} KB)" for c in csv_files]
                choice = st.selectbox("Selecciona un CSV", options=names, index=0)
                selected = csv_files[names.index(choice)]
                df_csv = load_artifact_csv(selected["path"])

                st.caption(f"Ruta: `{selected['path']}`")
                if df_csv is not None:
                    st.write(f"Shape: {df_csv.shape}")
                    st.dataframe(df_csv.head(200), use_container_width=True, height=400)

                    # Intentar gráfico de series si hay columna fecha
                    date_cols = [c for c in df_csv.columns if "date" in c.lower()]
                    num_cols = [c for c in df_csv.columns if pd.api.types.is_numeric_dtype(df_csv[c])]
                    if date_cols and num_cols:
                        date_col = date_cols[0]
                        st.markdown("### Serie de tiempo")
                        y_cols = st.multiselect("Columnas numéricas a graficar", num_cols, default=num_cols[:1])
                        if y_cols:
                            chart_df = df_csv[[date_col] + y_cols].copy()
                            chart_df = chart_df.sort_values(date_col)
                            chart_df = chart_df.set_index(date_col)
                            st.line_chart(chart_df)

        # --- Imágenes ---
        with tab_imgs:
            img_files = [a for a in artifacts if a["ext"] in ARTIFACT_IMAGE_EXTS]
            if not img_files:
                st.info("No se encontraron imágenes (png/jpg) en artifacts.")
            else:
                names = [f"{i['rel']} ({i['size_kb']} KB)" for i in img_files]
                choice = st.selectbox("Selecciona una imagen", options=names, index=0, key="img_select")
                selected = img_files[names.index(choice)]
                img = load_artifact_image(selected["path"])
                st.caption(f"Ruta: `{selected['path']}`")
                if img:
                    st.image(img, use_container_width=True)

        # --- Modelos y scalers ---
        with tab_models:
            model_exts = [".zip", ".joblib", ".pkl"]
            model_files = [a for a in artifacts if a["ext"] in model_exts]
            if not model_files:
                st.info("No se encontraron modelos o scalers (.zip/.joblib/.pkl).")
            else:
                st.dataframe(
                    pd.DataFrame(model_files)[["rel", "ext", "size_kb"]],
                    use_container_width=True
                )

        # --- Otros ---
        with tab_other:
            known_exts = {".csv", ".png", ".jpg", ".jpeg", ".zip", ".joblib", ".pkl"}
            other_files = [a for a in artifacts if a["ext"] not in known_exts]
            if not other_files:
                st.info("No hay otros tipos de archivos.")
            else:
                st.dataframe(
                    pd.DataFrame(other_files)[["rel", "ext", "size_kb"]],
                    use_container_width=True
                )

# ==============================================================================
# PÁGINA 6: SIMULADOR HMM (CORREGIDO)
# ==============================================================================
elif page == "🔄 Simulador HMM":
    st.markdown('<div class="title-section">🔄 Simulador de Regímenes (HMM)</div>', unsafe_allow_html=True)
    
    # Importación local para no romper el resto si falta la librería
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        st.error("⚠️ Falta la librería 'hmmlearn'. Instálala con: `pip install hmmlearn`")
        st.stop()

    try:
        df = load_master_df(DATA_PROCESSED)
        
        if df is not None and not df.empty:
            st.markdown("""
            ### Descripción
            El sistema calcula automáticamente los regímenes de mercado basándose en los retornos y la volatilidad:
            - 🔴 **Bear (Bajista):** Retornos negativos / Alta volatilidad.
            - 🟡 **Sideways (Lateral):** Retornos cercanos a cero.
            - 🟢 **Bull (Alcista):** Retornos positivos / Baja volatilidad relativa.
            """)
            
            # --- LÓGICA DE CÁLCULO DE HMM EN TIEMPO REAL ---
            if 'hmm_state' not in df.columns:
                with st.spinner("Calculando regímenes de mercado (HMM)..."):
                    # 1. Preparar datos para el HMM (Retornos y Volatilidad)
                    # Aseguramos que no haya NaNs
                    hmm_data = df[['ret', 'vol_20']].dropna()
                    X = hmm_data.values

                    # 2. Entrenar modelo
                    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
                    model.fit(X)
                    hidden_states = model.predict(X)

                    # 3. Ordenar estados para que los colores coincidan (Bear=0, Bull=2)
                    # Calculamos el retorno promedio de cada estado predicho
                    means = []
                    for i in range(3):
                        means.append(hmm_data.iloc[hidden_states == i]['ret'].mean())
                    
                    # Ordenamos: Menor retorno -> Bear (0), Mayor retorno -> Bull (2)
                    order = np.argsort(means)
                    mapping = {old: new for new, old in enumerate(order)}
                    
                    # Reasignamos los estados ordenados
                    mapped_states = np.array([mapping[s] for s in hidden_states])
                    
                    # Guardamos en el DF (alineando índices)
                    df.loc[hmm_data.index, 'hmm_state'] = mapped_states
                
                st.success("✅ Regímenes calculados exitosamente.")

            # --- INTERFAZ GRÁFICA ---
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input(
                    "Fecha de inicio",
                    value=df.index[0].date() if isinstance(df.index[0], pd.Timestamp) else df.index[0]
                )
            
            with col2:
                end_date = st.date_input(
                    "Fecha de fin",
                    value=df.index[-1].date() if isinstance(df.index[-1], pd.Timestamp) else df.index[-1]
                )
            
            # Filtrar datos
            mask = (df.index >= str(start_date)) & (df.index <= str(end_date))
            df_filtered = df.loc[mask].copy()
            
            if df_filtered.empty:
                st.warning("⚠️ No hay datos en el rango seleccionado.")
            else:
                # --- GRÁFICO ---
                st.markdown("### 📈 Precio con Regímenes de Mercado")
                
                fig = go.Figure()
                
                # Definir colores explícitos para asegurar consistencia
                # 0: Bear (Rojo), 1: Sideways (Amarillo/Gris), 2: Bull (Verde)
                color_map = {0: 'rgba(255, 0, 0, 0.15)', 1: 'rgba(255, 165, 0, 0.15)', 2: 'rgba(0, 255, 0, 0.15)'}
                label_map = {0: 'Bear (Bajista)', 1: 'Sideways (Lateral)', 2: 'Bull (Alcista)'}
                
                # Dibujar áreas de fondo
                # Truco: Usamos bar charts invisibles o shapes. Aquí usamos shapes para mejor performance.
                shapes = []
                
                # Iterar sobre segmentos contiguos para no crear miles de trazos
                # (Simplificación: dibujamos puntos coloreados detrás de la línea)
                
                # Método alternativo: Scatter con línea de precio encima
                for state in [0, 1, 2]:
                    state_data = df_filtered[df_filtered['hmm_state'] == state]
                    if not state_data.empty:
                        fig.add_trace(go.Scatter(
                            x=state_data.index,
                            y=state_data['close'],
                            mode='markers',
                            marker=dict(size=6, color=color_map[state].replace('0.15', '1')), # Color sólido para los puntos
                            name=label_map[state],
                            showlegend=True
                        ))

                # Línea de precio principal (negra y fina para conectar)
                fig.add_trace(go.Scatter(
                    x=df_filtered.index,
                    y=df_filtered['close'],
                    mode='lines',
                    line=dict(color='black', width=1),
                    name='Precio',
                    opacity=0.5
                ))

                fig.update_layout(
                    title='Clasificación de Regímenes HMM',
                    xaxis_title='Fecha',
                    yaxis_title='Precio ($)',
                    height=500,
                    template='plotly_white',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # --- ESTADÍSTICAS ---
                st.markdown("### 📊 Estadísticas por Régimen Detectado")
                
                regime_stats = []
                for state in [0, 1, 2]:
                    state_data = df_filtered[df_filtered['hmm_state'] == state]
                    if not state_data.empty:
                        regime_stats.append({
                            'Régimen': label_map[state],
                            'Días en Régimen': len(state_data),
                            '% del Tiempo': f"{(len(state_data)/len(df_filtered)*100):.1f}%",
                            'Retorno Promedio Diario': f"{state_data['ret'].mean()*100:.4f}%",
                            'Volatilidad (Std)': f"{state_data['ret'].std():.4f}"
                        })
                
                if regime_stats:
                    st.dataframe(pd.DataFrame(regime_stats), use_container_width=True)

        else:
            st.error("❌ Error al cargar datos maestros.")
    
    except Exception as e:
        st.error(f"❌ Error crítico en el módulo HMM: {e}")

# ==============================================================================
# PÁGINA 6: DETALLES TÉCNICOS
# ==============================================================================
elif page == "📋 Detalles Técnicos":
    st.markdown('<div class="title-section">📋 Detalles Técnicos</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Arquitectura", "Datos", "Configuración"])
    
    with tab1:
        st.markdown("""
        ## 🏗️ Arquitectura del Sistema
        
        ```
        Fuentes de Datos (4)
        ├── Yahoo Finance (Precios EPU)
        ├── World Bank (Macro Perú)
        ├── CSV VIX (Volatilidad)
        └── Alpha Vantage (Sentimiento)
        
        ↓
        
        Data Pipeline
        ├── Limpieza y normalización
        ├── Feature Engineering (ret, vol_20, mom_5)
        └── Unificación de frecuencias
        
        ↓
        
        Hidden Markov Model (3 estados)
        ├── Bear (Bajista)
        ├── Sideways (Lateral)
        └── Bull (Alcista)
        
        ↓
        
        Reinforcement Learning (DQN)
        ├── Agente CON información HMM
        └── Agente SIN información HMM (baseline)
        
        ↓
        
        Evaluación y Comparativa
        └── Sharpe, Retorno Acumulado, Max Drawdown
        ```
        """)
    
    with tab2:
        st.markdown("""
        ## 📊 Información de Datos
        
        | Aspecto | Descripción |
        |---------|-------------|
        | **Ticker** | EPU (iShares MSCI Peru ETF) |
        | **Período** | 2018-01-01 a 2025-12-01 |
        | **Frecuencia** | Diaria |
        | **Fuentes** | 4 (Yahoo, World Bank, VIX, Alpha Vantage) |
        | **Variables** | Precio, Macro, Volatilidad, Sentimiento |
        | **Características** | ret, vol_20, mom_5 + HMM probabilities |
        """)
        
        try:
            df = load_master_df(DATA_PROCESSED)
            if df is not None:
                st.markdown("### Muestra de Datos")
                st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo cargar muestra de datos: {e}")
    
    with tab3:
        st.markdown("""
        ## ⚙️ Parámetros de Configuración
        
        ### HMM
        - **Estados**: 3 (Bear, Sideways, Bull)
        - **Covarianza**: Diagonal
        - **Iteraciones**: 800
        - **Seed**: 7
        
        ### RL (DQN)
        - **Algoritmo**: Deep Q-Network (Stable-Baselines3)
        - **Learning Rate**: 1e-4
        - **Buffer Size**: 50,000
        - **Batch Size**: 64
        - **Gamma**: 0.99
        - **Training Timesteps**: 50,000
        - **Fee Transacción**: 0.05% (0.0005)
        
        ### Train/Test Split
        - **Proporción**: 80% train / 20% test
        
        ### Acciones RL
        - `Hold` (0): Mantener posición
        - `Buy` (1): Ir a posición long (+1)
        - `Sell` (2): Ir a posición neutral (0)
        """)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>📊 Trading RL + HMM Dashboard | Proyecto Académico | 2024</p>
    <p>Datos: EPU (iShares MSCI Peru ETF) | Período: 2018-2025</p>
</div>
""", unsafe_allow_html=True)
