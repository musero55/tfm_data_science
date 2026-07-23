import pandas as pd

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
