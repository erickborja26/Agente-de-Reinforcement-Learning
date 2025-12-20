import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
import os
from typing import Optional

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
# PÁGINA 2: ANÁLISIS EDA
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
                use_column_width=True
            )
            
            st.markdown("---")
            
            # Mostrar todas las imágenes en grid (opcional)
            if st.checkbox("📸 Mostrar todos los gráficos en grid"):
                cols = st.columns(2)
                for idx, (name, img) in enumerate(images.items()):
                    with cols[idx % 2]:
                        st.image(img, caption=name, use_column_width=True)
    
    except Exception as e:
        st.error(f"❌ Error al cargar gráficos: {e}")

# ==============================================================================
# PÁGINA 3: COMPARATIVA DE MODELOS
# ==============================================================================
elif page == "🏆 Comparativa de Modelos":
    st.markdown('<div class="title-section">🏆 Benchmark de 10 Modelos</div>', unsafe_allow_html=True)
    
    try:
        df_results = load_excel_results(RESULTS_EXCEL)
        
        if df_results is not None and not df_results.empty:
            st.markdown("### 📊 Tabla de Resultados")
            
            # Ordenar por Sharpe Ratio
            df_sorted = df_results.sort_values(
                by="Sharpe Ratio",
                ascending=False,
                na_position='last'
            )
            
            # Mostrar tabla con formato
            st.dataframe(
                df_sorted,
                use_container_width=True,
                height=400
            )
            
            st.markdown("---")
            
            # Gráficos comparativos
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Retorno Acumulado")
                fig1 = px.bar(
                    df_sorted,
                    x='Modelo',
                    y='Cumulative Return',
                    color='Cumulative Return',
                    color_continuous_scale='RdYlGn',
                    title='Retorno Acumulado por Modelo',
                    labels={'Cumulative Return': 'Retorno (%)'}
                )
                fig1.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Sharpe Ratio")
                fig2 = px.bar(
                    df_sorted,
                    x='Modelo',
                    y='Sharpe Ratio',
                    color='Sharpe Ratio',
                    color_continuous_scale='Blues',
                    title='Sharpe Ratio por Modelo (Mejor = Mayor)',
                    labels={'Sharpe Ratio': 'Sharpe Ratio'}
                )
                fig2.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("---")
            
            # Scatter plot: Retorno vs Riesgo
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("### ⚖️ Riesgo vs Retorno")
                fig3 = px.scatter(
                    df_sorted,
                    x='Max Drawdown',
                    y='Cumulative Return',
                    hover_name='Modelo',
                    color='Sharpe Ratio',
                    size='Sharpe Ratio',
                    color_continuous_scale='Viridis',
                    title='Análisis Riesgo-Retorno',
                    labels={
                        'Max Drawdown': 'Max Drawdown (Riesgo)',
                        'Cumulative Return': 'Retorno Acumulado'
                    }
                )
                fig3.update_layout(height=400)
                st.plotly_chart(fig3, use_container_width=True)
            
            with col4:
                st.markdown("### 🥇 Top 3 Modelos")
                top3 = df_sorted.head(3)[['Modelo', 'Sharpe Ratio', 'Cumulative Return']]
                st.dataframe(top3, use_container_width=True, height=200)
                
                st.markdown("### 📝 Interpretación")
                st.info(
                    "**Sharpe Ratio**: Mide el retorno ajustado por riesgo. "
                    "Mayor es mejor.\n\n"
                    "**Cumulative Return**: Ganancia/pérdida total acumulada.\n\n"
                    "**Max Drawdown**: Peor caída desde un pico. Menor es mejor (menos negativo)."
                )
        else:
            st.warning("⚠️ No se encontró archivo de resultados.")
            st.info(f"Ruta esperada: `{RESULTS_EXCEL}`")
    
    except Exception as e:
        st.error(f"❌ Error al cargar resultados: {e}")

# ==============================================================================
# PÁGINA 4: SIMULADOR HMM
# ==============================================================================
elif page == "🔄 Simulador HMM":
    st.markdown('<div class="title-section">🔄 Simulador de Regímenes (HMM)</div>', unsafe_allow_html=True)
    
    try:
        df = load_master_df(DATA_PROCESSED)
        
        if df is not None and not df.empty:
            st.markdown("""
            ### Descripción
            El **Hidden Markov Model (HMM)** identifica regímenes ocultos del mercado:
            - 🔴 **Bear** (Bajista): Tendencia negativa, mayor riesgo
            - 🟡 **Sideways** (Lateral): Consolidación, movimiento lateral
            - 🟢 **Bull** (Alcista): Tendencia positiva, menor riesgo
            """)
            
            # Slider de rango de fechas
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
            
            # Filtrar datos en el rango
            mask = (df.index >= str(start_date)) & (df.index <= str(end_date))
            df_filtered = df.loc[mask].copy()
            
            if df_filtered.empty:
                st.warning("⚠️ No hay datos en el rango seleccionado.")
            else:
                # Gráfico interactivo con regímenes
                st.markdown("### 📈 Precio con Regímenes de Mercado")
                
                # Crear figura con Plotly
                fig = go.Figure()
                
                # Agregar línea de precio
                fig.add_trace(go.Scatter(
                    x=df_filtered.index,
                    y=df_filtered['close'],
                    mode='lines',
                    name='Precio (Close)',
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Precio: $%{y:.2f}<extra></extra>'
                ))
                
                # Si existe columna hmm_state, colorear el fondo
                if 'hmm_state' in df_filtered.columns:
                    regime_colors = get_regime_colors()
                    
                    # Agregar áreas de color según régimen
                    for state in df_filtered['hmm_state'].unique():
                        if pd.isna(state):
                            continue
                        
                        state_mask = df_filtered['hmm_state'] == state
                        state_data = df_filtered[state_mask]
                        
                        # Determinar etiqueta
                        state_map = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
                        label = state_map.get(int(state), f'State {state}')
                        
                        # Agregar scatter invisible para la leyenda
                        fig.add_trace(go.Scatter(
                            x=state_data.index,
                            y=state_data['close'],
                            mode='markers',
                            marker=dict(size=0),
                            name=label,
                            showlegend=True,
                            hoverinfo='skip'
                        ))
                
                fig.update_layout(
                    title='Precio EPU con Regímenes HMM',
                    xaxis_title='Fecha',
                    yaxis_title='Precio ($)',
                    hovermode='x unified',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Estadísticas por régimen
                st.markdown("### 📊 Estadísticas por Régimen")
                
                if 'hmm_state' in df_filtered.columns:
                    regime_stats = []
                    state_map = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
                    
                    for state in sorted(df_filtered['hmm_state'].unique()):
                        if pd.isna(state):
                            continue
                        
                        state_data = df_filtered[df_filtered['hmm_state'] == state]
                        label = state_map.get(int(state), f'State {state}')
                        
                        regime_stats.append({
                            'Régimen': label,
                            'Días': len(state_data),
                            'Precio Medio': f"${state_data['close'].mean():.2f}",
                            'Retorno Medio': f"{state_data['ret'].mean():.4f}",
                            'Volatilidad': f"{state_data['ret'].std():.4f}"
                        })
                    
                    df_stats = pd.DataFrame(regime_stats)
                    st.dataframe(df_stats, use_container_width=True)
                else:
                    st.info("ℹ️ La columna 'hmm_state' no está disponible en los datos.")
        else:
            st.error("❌ Error al cargar datos maestros.")
    
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ==============================================================================
# PÁGINA 5: DETALLES TÉCNICOS
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