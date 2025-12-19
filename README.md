# Agente de Reinforcement Learning con HMM para Trading del ETF EPU (Perú)

## 📌 Descripción general

Este proyecto implementa un **agente de Reinforcement Learning (RL)** que integra un  
**Hidden Markov Model (HMM)** para identificar **regímenes ocultos del mercado** y tomar
decisiones de **compra, venta o mantener (Buy / Sell / Hold)** sobre el activo financiero:

> **EPU – iShares MSCI Peru ETF**

El sistema combina información proveniente de **múltiples fuentes heterogéneas**:
- Precios históricos del ETF
- Indicadores macroeconómicos del Perú
- Volatilidad histórica (archivo Excel)
- Noticias financieras con análisis de sentimiento

El objetivo principal es **evaluar si la incorporación de estados latentes (HMM) mejora
el desempeño de un agente de RL**, comparándolo contra un agente que no utiliza dicha
información.

---

## 🎯 Objetivos

### Objetivo general
Construir y evaluar un agente de trading basado en **Reinforcement Learning** que utilice
un **Hidden Markov Model** para enriquecer la representación del estado del mercado.

### Objetivos específicos
- Integrar al menos **4 fuentes de datos distintas** (APIs y archivos Excel).
- Modelar **estados ocultos del mercado** (alcista, bajista y lateral) mediante HMM.
- Entrenar un agente **Deep Q-Network (DQN)** con y sin información del HMM.
- Comparar el desempeño usando métricas financieras estándar.

---

## 🧠 Arquitectura del sistema

Fuentes de Datos
│
├── Precios (Yahoo Finance - EPU)
├── Macro Perú (BCRPData)
├── Volatilidad (Excel)
└── Noticias y Sentimiento (Alpha Vantage)
↓
Limpieza y Feature Engineering
↓
Hidden Markov Model (Regímenes de mercado)
↓
Estado aumentado (features + probabilidades HMM)
↓
Agente de Reinforcement Learning (DQN)
↓
Decisiones: Buy / Sell / Hold


---

## 📊 Fuentes de datos

| Tipo | Fuente | Uso |
|----|------|----|
| Precios | Yahoo Finance (`EPU`) | Retornos y dinámica del mercado |
| Macro Perú | BCRPData (API) | Contexto macroeconómico |
| Volatilidad | Archivo Excel | Medida de riesgo |
| Noticias | Alpha Vantage – Market News & Sentiment | Análisis de sentimiento |

---

## 🧩 Componentes principales

### 1. Ingesta y limpieza de datos
- Descarga automática de precios y noticias vía API.
- Lectura de archivos Excel.
- Normalización de fechas y escalas.
- Manejo de valores faltantes mediante *forward fill*.

### 2. Hidden Markov Model (HMM)
- Implementado con `GaussianHMM`.
- Identificación de regímenes de mercado.
- Uso de **probabilidades posteriores** como parte del estado del agente.

### 3. Reinforcement Learning
- Agente **Deep Q-Network (DQN)**.
- Espacio de acciones discreto: `Hold`, `Buy`, `Sell`.
- Función de recompensa basada en retorno diario neto de costos de transacción.

### 4. Evaluación
- **Cumulative Return**
- **Sharpe Ratio**
- **Maximum Drawdown**
- Comparación entre:
  - DQN **con HMM**
  - DQN **sin HMM**

---

## 📁 Estructura del proyecto

Agente-de-Reinforcement-Learning/
├─ src/ # Código fuente
│ ├─ data/ # Ingesta de datos
│ ├─ features/ # Feature engineering
│ ├─ hmm/ # Modelos HMM
│ ├─ rl/ # Entorno y entrenamiento RL
│ └─ utils/ # Cache y métricas
├─ scripts/
│ └─ run_pipeline.py # Ejecución completa del flujo
├─ notebooks/ # Análisis exploratorio
├─ data/ # Datos (NO versionados)
├─ artifacts/ # Modelos entrenados (NO versionados)
├─ requirements.txt
├─ .gitignore
└─ README.md

## ⚙️ Instalación y entorno

### 1. Crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .\.venv\Scripts\activate     # Windows
```
### 2. Instalar dependencias

pip install -r requirements.txt

### 3. Configurar API Key

Crear un archivo .env en la raíz del proyecto:

ALPHAVANTAGE_API_KEY=TU_API_KEY_AQUI

La API Key se obtiene gratuitamente en:
https://www.alphavantage.co/support/#api-key

## ▶️ Ejecución del pipeline

```bash
python scripts/run_pipeline.py
```

Este script realiza las siguientes etapas:

1. Descarga o reutiliza datos cacheados
2. Entrena el modelo **Hidden Markov Model (HMM)**
3. Entrena el agente de **Reinforcement Learning (DQN)**
4. Evalúa y compara los resultados obtenidos

---

## 🗃️ Cache de noticias (optimización)

Para evitar exceder los límites de la API de **Alpha Vantage**, el sistema implementa un mecanismo de cache local:

- Las respuestas de la API se **almacenan en disco**.
- Si un rango de fechas ya fue consultado, **no se realiza un nuevo request**.
- El cache está indexado por:
  - Ticker
  - Rango temporal
  - Topics
  - Parámetros de la API

Este enfoque mejora la **eficiencia**, **reproducibilidad** y **estabilidad** del experimento.

---

## ⚠️ Consideraciones importantes

- Este proyecto tiene **fines estrictamente académicos**.
- No constituye una recomendación de inversión.
- No se consideran fricciones reales del mercado como:
  - *Slippage*
  - Liquidez
  - Restricciones regulatorias

---

## 📚 Tecnologías utilizadas

- Python 3.11
- Pandas / NumPy
- Scikit-learn
- hmmlearn
- Gymnasium
- Stable-Baselines3 (DQN)
- Alpha Vantage API
- Yahoo Finance

---

## 📈 Posibles extensiones

- Uso de PPO o SAC en lugar de DQN
- Incorporar costos de transacción dinámicos
- Validación *walk-forward*
- Uso de LSTM o Transformers en el agente RL
- Trading multi-activo

---

## 👤 Autor

Proyecto desarrollado con fines académicos para el estudio de:

**Reinforcement Learning, Hidden Markov Models y Finanzas Computacionales**  
aplicados al análisis del mercado peruano.
