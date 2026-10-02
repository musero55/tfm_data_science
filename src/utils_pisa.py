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
    Calcula la media ponderada de los 10 Valores Plausibles (PV) a nivel de ESTRATO.
    
    Argumentos:
        df (pd.DataFrame): Dataset (ej. datos de 2022)
        competencia (str): Sufijo de la competencia (ej. 'READ', 'MATH', 'SCIE').
    """
    
    # Columnas que conservamos
    cols_identificacion = ['CNT', 'STRATUM', col_peso]
    
    # Generamos los nombres de las 10 columnas PV (PV1MATH, PV2MATH...)
    cols_pv = [f'PV{i}{competencia}' for i in range(1, 11)]
    
    # Creamos un dataframe temporal solo con las columnas necesarias
    df_temp = df[cols_identificacion + cols_pv].copy()
    
    # Calculamos la media de los 10 PVs para cada alumno 
    df_temp['media_alumno'] = df_temp[cols_pv].mean(axis='columns') 
    
    # Ponderamos la nota del alumno por su peso
    df_temp['nota_ponderada'] = df_temp['media_alumno'] * df_temp[col_peso]
    
    # Agrupamos por país y estrato sumando notas ponderadas y pesos
    df_estratos = df_temp.groupby(['CNT', 'STRATUM']).agg(
        suma_notas=('nota_ponderada', 'sum'),
        suma_pesos=(col_peso, 'sum')
    ).reset_index()
    
    # Calculamos la media real del estrato ( la media ponderada  basada en los pesos )
    nombre_col_final = f'media_{competencia.lower()}_estrato'
    df_estratos[nombre_col_final] = df_estratos['suma_notas'] / df_estratos['suma_pesos']
    
    # Limpiamos las columnas intermedias y dejamos solo el resultado
    columnas_finales = ['CNT', 'STRATUM', nombre_col_final]
    
    return df_estratos[columnas_finales]

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