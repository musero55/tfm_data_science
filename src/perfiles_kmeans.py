"""
perfiles_kmeans.py
==================

Agrupación en perfiles con K-means: estandariza las variables, prueba varios números de grupos y
elige el de mayor silueta entre los que no dejan grupos demasiado pequeños. La usan los perfiles de
gasto de `01_EDA_presupuestos.ipynb`, los perfiles macroeconómicos de
`01_EDA_factores_macroeconomicos.ipynb` y el perfil de país de `02_data_panel_modelos.ipynb`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

INTENTOS_KMEANS = 20   # n_init: K-means se repite desde 20 puntos de partida y se queda con la mejor solución


def agrupar_perfiles(
    datos: pd.DataFrame,
    numeros_de_grupos=range(2, 9),
    min_por_grupo: int = 3,
    semilla: int = 42,
) -> dict:
    """K-means sobre las columnas de `datos` (sin nulos) estandarizadas con `StandardScaler`,
    con cada nº de grupos de `numeros_de_grupos`.

    Elige el nº de grupos con mayor silueta entre los que no dejan ningún grupo con menos de
    `min_por_grupo` filas (si ninguno lo cumple, entre todos). Devuelve un diccionario con:

    * `resumen`: inercia, silueta y tamaño del grupo más pequeño de cada nº de grupos.
    * `k`: el nº de grupos elegido; `k_codo`: el del método del codo (el más alejado de la recta
      que une el primer y el último punto de la curva de inercia), como referencia.
    * `etiquetas`: grupo de cada fila con el `k` elegido (indexado como `datos`).
    * `modelos`: el KMeans ajustado para cada nº de grupos; `escalador`: el StandardScaler.
    """
    escalador = StandardScaler().fit(datos)
    datos_estandarizados = escalador.transform(datos)
    modelos, filas = {}, []
    for numero_grupos in numeros_de_grupos:
        modelo = KMeans(n_clusters=numero_grupos, n_init=INTENTOS_KMEANS, random_state=semilla).fit(datos_estandarizados)
        modelos[numero_grupos] = modelo
        filas.append({'k': numero_grupos, 'inercia': modelo.inertia_,
                      'silueta': silhouette_score(datos_estandarizados, modelo.labels_),
                      'grupo más pequeño': np.bincount(modelo.labels_).min()})
    resumen = pd.DataFrame(filas).set_index('k')

    validos = resumen[resumen['grupo más pequeño'] >= min_por_grupo]
    k_elegido = int((validos if len(validos) else resumen)['silueta'].idxmax())
    numeros, inercias = resumen.index.to_numpy(dtype=float), resumen['inercia'].to_numpy()
    recta = inercias[0] + (inercias[-1] - inercias[0]) * (numeros - numeros[0]) / (numeros[-1] - numeros[0])
    k_codo = int(numeros[np.argmax(recta - inercias)])

    return {'resumen': resumen, 'k': k_elegido, 'k_codo': k_codo,
            'etiquetas': pd.Series(modelos[k_elegido].labels_, index=datos.index, name='perfil'),
            'modelos': modelos, 'escalador': escalador}
