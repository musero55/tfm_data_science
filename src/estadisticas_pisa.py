import pandas as pd

def consolidar_puntuaciones_pisa(df, competencia, col_pais='CNT', col_peso='W_FSTUWT'):
    """
    Calcula las medias ponderadas de los 10 Valores Plausibles (PV) por país,
    la puntuación media final y el error de imputación (varianza).
    
    Argumentos:
        df (pd.DataFrame): Dataset crudo de PISA.
        competencia (str): Sufijo de la competencia (ej. 'READ', 'MATH', 'SCIE').
    """
    medias_pv = []
    
    # Calculamos las 10 medias ponderadas por pais
    for i in range(1, 11):
        col_actual = f'PV{i}{competencia}'
        
        media_ponderada = df.groupby(col_pais).apply(
            lambda x: (x[col_actual] * x[col_peso]).sum() / x[col_peso].sum(),
            include_groups=False
        )
        medias_pv.append(media_ponderada)
    
    # Registramos las 10 medias calculadas en un dataFrame
    df_pv_consolidado = pd.concat(medias_pv, axis=1)
    
    # Nombrar las columnas individuales de los PVs para la tabla final
    df_pv_consolidado.columns = [f'pv{i}_{competencia.lower()}' for i in range(1, 11)]
    
    # Calcular métricas finales aggregadas
    puntuacion_final = df_pv_consolidado.mean(axis='columns')   #Media del conjunto PV
    error_imputacion = df_pv_consolidado.var(axis='columns')    #Varianza de las medias
    
    # Construir la matriz limpia final
    df_resultados = pd.DataFrame({
        'media_pisa': puntuacion_final,
        'varianza_imputacion': error_imputacion
    })
    
    # Unir los PVs individuales con las medias y resetear el índice del país
    df_final = pd.concat([df_resultados, df_pv_consolidado], axis=1).reset_index()
    
    # Estandarizar nombres de columnas según vuestro estándar snake_case
    df_final = df_final.rename(columns={
        col_pais: 'codigo_pais',
        'media_pisa': f'media_{competencia.lower()}_pisa',
        'varianza_imputacion': f'varianza_imputacion_{competencia.lower()}'
    })
    
    return df_final
