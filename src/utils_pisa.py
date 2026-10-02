import pandas as pd
import numpy as np
import re
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

def puntuaciones_pisa_alumnos(df, competencia):
    """
    Calcula la media individual de los 10 Valores Plausibles (PV) por alumno
    y conserva los identificadores para cruces posteriores.
    
    Argumentos:
        df (pd.DataFrame): Dataset 
        competencia (str): Sufijo de la competencia (ej. 'READ', 'MATH', 'SCIE').
    """
    
    # Columnas a conservar
    cols_identificacion = ['CNT', 'CNTSCHID', 'CNTSTUID']
    
    # Generamos los nombres de las 10 columnas PV (PV1MATH, PV2MATH...)
    cols_pv = [f'PV{i}{competencia}' for i in range(1, 11)]
    
    # Creamos un dataframe con los IDs, pesos y los 10 PVs
    df_resultados = df[cols_identificacion + cols_pv].copy()
    
    # Calculamos la media de los 10 PVs para cada alumno (un valor por 'alumno')
    df_resultados[f'media_{competencia.lower()}_pisa'] = df_resultados[cols_pv].mean(axis='columns') 
    
    # Limpiamos los 10 PV sueltos
    columnas_finales = cols_identificacion + [f'media_{competencia.lower()}_pisa']
    
    # Devolvemos el dataframe
    return df_resultados[columnas_finales]

def puntuaciones_pisa_estratos(df, competencia,col_peso='W_FSTUWT'):
    """
    Calcula la media ponderada de los valores plausibles por país y estrato.
    """
    # Identificar las columnas de valores plausibles (Las queue empiecen por PV y terminen en el nombre de la competencia a evaluar)
    cols_pv = [col for col in df.columns if col.startswith('PV') and col.endswith(competencia)]
    
    if not cols_pv:
        raise ValueError(f"No se encontraron valores plausibles para {competencia}.")
        
    # Calcular la media de los PVs para cada alumno directamente
    media_alumno = df[cols_pv].mean(axis=1)
    
    # Calcular la nota ponderada de cada alumno
    nota_ponderada = media_alumno * df[col_peso]
    
    # Agrupar por país y estrato y calculamos la media 
    # Agrupamos el DataFrame original usando series calculadas externamente
    grouped = df.groupby(['CNT', 'STRATUM'])
    
    suma_notas = grouped.apply(lambda x: (media_alumno.loc[x.index] * x[col_peso]).sum())
    suma_pesos = grouped[col_peso].sum()
    
    # Generamos el dataframe de salida
    df_estratos = (suma_notas / suma_pesos).reset_index(name=f'media_{competencia.lower()}_estrato')
    
    return df_estratos

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