import pandas as pd
import numpy as np
from typing import Any
import pickle
import os
import random

def puntuaciones_pisa_pais(df, competencia, col_pais='CNT', col_peso='W_FSTUWT'):
    """
    Calcula las medias ponderadas de los 10 Valores Plausibles (PV) por país,
    la puntuación media final y el error de imputación (varianza).
    
    Argumentos:
        df (pd.DataFrame): Dataset crudo de PISA.
        competencia (str): Sufijo de la competencia (ej. 'READ', 'MATH', 'SCIE').
    """
    medias_pv = []
    
    # Calculamos una media ponderada por cada valor de PV presente
    for i in range(1, 11):
        col_actual = f'PV{i}{competencia}'

        #Se crean 10 series de Pandas con indice del pais y valor la media ponderada de PV(i)
        media_ponderada = df.groupby(col_pais).apply(
            lambda x: (x[col_actual] * x[col_peso]).sum() / x[col_peso].sum(),
            include_groups=False
        )

        #Se registra cada Series en un array
        medias_pv.append(media_ponderada)
    
    # Se genera un dataframe unico cuyo indice son los CNT y las columnas son las medias ponderadas PV
    df_pv_consolidado = pd.concat(medias_pv, axis=1,)
    
    # Calcular métricas finales aggregadas
    puntuacion_final = df_pv_consolidado.mean(axis='columns')   #Media del conjunto PV
    error_imputacion = df_pv_consolidado.var(axis='columns')    #Varianza de las medias
    
    # Se construye el dataframe final con la media PV y error de imputacion por pais
    df_resultados = pd.DataFrame({
        f'media_{competencia.lower()}_pisa': puntuacion_final,
        'varianza_imputacion': error_imputacion
    })
    
    return df_resultados.reset_index()

def guardar_pickle(ruta_archivo: str, *objetos: Any) -> None:
    """
    Guarda uno o múltiples objetos de Python en un único archivo binario (.pkl).
    
    Uso:
        guardar_pickle('datos.pkl', mi_set, mi_dict, mi_lista)
    """
    try:
        # Asegurar que la carpeta contenedora exista
        carpeta = os.path.dirname(ruta_archivo)
        if carpeta and not os.path.exists(carpeta):
            os.makedirs(carpeta)
            
        with open(ruta_archivo, 'wb') as f:
            # Si es un solo objeto, lo guarda directo; si son varios, los guarda como tupla
            pickle.dump(objetos[0] if len(objetos) == 1 else objetos, f)
            
        print(f"✅ {len(objetos)} objeto(s) guardado(s) con éxito en: {ruta_archivo}")
    except Exception as e:
        print(f"❌ Error al guardar en Pickle: {e}")


def cargar_pickle(ruta_archivo: str) -> Any:
    """
    Carga objeto(s) desde un archivo binario (.pkl).
    
    Uso:
        mi_set, mi_dict = cargar_pickle('datos.pkl')
    """
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"❌ El archivo especificado no existe: {ruta_archivo}")
        
    try:
        with open(ruta_archivo, 'rb') as f:
            datos = pickle.load(f)
        print(f" Objetos cargados correctamente desde: {ruta_archivo}")
        return datos
    except Exception as e:
        print(f"❌ Error al cargar desde Pickle: {e}")
        return None
    

def establecer_semilla(seed: int = 42):
    """
    Fija las semillas de aleatoriedad para asegurar la reproducibilidad del proyecto.
    """
    # 1. Python puro
    random.seed(seed)
    
    # 2. Variables de entorno (evita aleatoriedad en hashes de diccionarios)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. NumPy
    np.random.seed(seed)
    
    print(f"✅ Semillas del proyecto establecidas con éxito (seed={seed}).")