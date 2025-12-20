import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Optional

def load_master_df(path: str | Path) -> Optional[pd.DataFrame]:
    """
    Carga el archivo master_df.csv con manejo de errores.
    """
    try:
        path = Path(path)
        if not path.exists():
            return None
        
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df.sort_index()
    
    except Exception as e:
        print(f"Error cargando {path}: {e}")
        return None

def load_excel_results(path: str | Path) -> Optional[pd.DataFrame]:
    """
    Carga el archivo Excel de resultados de comparativa de modelos.
    """
    try:
        path = Path(path)
        if not path.exists():
            return None
        
        df = pd.read_excel(path)
        return df
    
    except Exception as e:
        print(f"Error cargando {path}: {e}")
        return None

def load_images_from_directory(directory: str | Path) -> Dict[str, Image.Image]:
    """
    Carga todas las imágenes PNG de un directorio.
    Retorna un diccionario {nombre_archivo: Image}
    """
    images = {}
    directory = Path(directory)
    
    if not directory.exists():
        return images
    
    try:
        for img_path in sorted(directory.glob("*.png")):
            try:
                img = Image.open(img_path)
                images[img_path.stem] = img
            except Exception as e:
                print(f"Error cargando {img_path}: {e}")
    
    except Exception as e:
        print(f"Error al explorar directorio: {e}")
    
    return images

def calculate_kpis(df: Optional[pd.DataFrame]) -> Dict:
    """
    Calcula KPIs principales del DataFrame.
    """
    kpis = {
        'last_price': 0.0,
        'price_change_pct': 0.0,
        'volatility_20d': 0.0,
        'cumulative_return': 0.0,
        'total_days': 0,
        'start_date': 'N/A',
        'end_date': 'N/A'
    }
    
    if df is None or df.empty:
        return kpis
    
    try:
        # Precio último
        if 'close' in df.columns:
            last_price = df['close'].iloc[-1]
            first_price = df['close'].iloc[0]
            kpis['last_price'] = float(last_price)
            kpis['price_change_pct'] = float(((last_price / first_price) - 1.0) * 100)
            kpis['cumulative_return'] = kpis['price_change_pct']
        
        # Volatilidad
        if 'ret' in df.columns:
            vol_20 = df['ret'].rolling(20).std().iloc[-1]
            kpis['volatility_20d'] = float(vol_20 * np.sqrt(252))  # Anualizada
        
        # Días
        kpis['total_days'] = len(df)
        kpis['start_date'] = str(df.index[0].date())
        kpis['end_date'] = str(df.index[-1].date())
    
    except Exception as e:
        print(f"Error calculando KPIs: {e}")
    
    return kpis

def get_regime_colors() -> Dict[int, str]:
    """
    Retorna mapeo de colores para regímenes HMM.
    """
    return {
        0: '#FF6B6B',  # Bear - Rojo
        1: '#FFD93D',  # Sideways - Amarillo
        2: '#6BCB77'   # Bull - Verde
    }