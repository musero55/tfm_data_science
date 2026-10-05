import pandas as pd
import numpy as np
import re
from typing import Any
import pickle
import os
import random

def puntuaciones_pisa_pais(df, competencia, col_peso='W_FSTUWT'):
    """
    Calcula la media ponderada de los valores plausibles por país (CNT).
    Usa la misma lógica que puntuaciones_pisa_estratos pero agrupando solo por CNT.
    
    Argumentos:
        df (pd.DataFrame): Dataset crudo de PISA.
        competencia (str): Sufijo de la competencia (ej. 'READ', 'MATH', 'SCIE').
        col_peso (str): Columna de pesos para la ponderación.
    """
    # Identificar las columnas de valores plausibles
    cols_pv = [col for col in df.columns if col.startswith('PV') and col.endswith(competencia)]
    
    if not cols_pv:
        raise ValueError(f"No se encontraron valores plausibles para {competencia}.")
        
    # Calcular la media de los PVs para cada alumno
    media_alumno = df[cols_pv].mean(axis=1)
    
    # Agrupar por país y calcular la media ponderada
    grouped = df.groupby('CNT')
    
    suma_notas = grouped.apply(lambda x: (media_alumno.loc[x.index] * x[col_peso]).sum())
    suma_pesos = grouped[col_peso].sum()
    
    # Generar el dataframe de salida
    df_paises = (suma_notas / suma_pesos).reset_index(name=f'media_{competencia.lower()}_pais')
    
    return df_paises

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

def promedio_ponderado_indicadores_agrupado(datos, variables, columna_peso, claves_agrupacion=None,):
    """
    Agrega indicadores ponderados por una o varias claves.
    datos: datos no separados por estratos

    """
    #Crea un dataframe vacio que contiene las claves de agrupación 
    resultado = datos[claves_agrupacion].drop_duplicates().reset_index(drop=True)

    
# Bucle para calcular el promedio ponderado de cada variable
    for variable in variables:
        
        # Evitamos multiplicar la columna de peso por si misma 
        if variable == columna_peso:
            continue
 

        #Localiza los datos válidos (no nulos) para la variable especifica 
        datos_validos = datos.loc[datos[variable].notna(),claves_agrupacion + [variable, columna_peso]].copy()
        #Multiplica la variable por el peso para obtener el indice ponderado
        datos_validos['_suma_ponderada'] = datos_validos[variable] * datos_validos[columna_peso]

        #Agrupa los datos por las claves de agrupación y calcula la suma ponderada y la suma de pesos
        resumen = datos_validos.groupby(claves_agrupacion, as_index=False).agg(
            suma_ponderada=('_suma_ponderada', 'sum'),
            suma_pesos=(columna_peso, 'sum'),
        )

        # se divide el total acumulado entre el peso total del estrato
        resumen[variable] = resumen['suma_ponderada'] / resumen['suma_pesos']

        # Se genera el dataframe final uniendo el df vacio de resultado con el de resumen
        resultado = resultado.merge(
            resumen[claves_agrupacion + [variable]],
            on=claves_agrupacion,
            how='left',
            validate='one_to_one',
        )

    return resultado

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