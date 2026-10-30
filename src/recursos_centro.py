"""
recursos_centro.py
==================

Recursos del centro comparables entre las ediciones de PISA 2012, 2015, 2018 y 2022, a partir del
cuestionario de centro (`leer_recursos_centros` de `pisa_pipeline.py`, guardado por `01_EDA_equidad`):

    * `VARIABLES_DOTACION`: dotación del centro que existe en las cuatro ediciones, con el signo que hace que
      un valor más alto sea un centro mejor dotado.
    * `VARIABLES_ESCASEZ`: escasez de material y de personal según el director (`EDUSHORT` y `STAFFSHORT` desde
      2015; en 2012, sus equivalentes de `ESCASEZ_2012`).
    * `leer_centros`: recursos de cada centro y edición de los países del análisis, con la escasez de 2012.
    * `transformar_recursos`: recorte de valores extremos y logaritmo de las razones.
    * `estandarizar_conjunto` / `cargas_primera_componente`: estandarización y componente principal con todas
      las ediciones juntas, con cada país-edición pesando igual.
    * `construir_indices`: índices de dotación (componente principal y pesos iguales) y de escasez.

Lo usa `01_EDA_recursos_centro.ipynb`.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.pisa_pipeline import rutas_datos_anuales

GRUPO_PAIS_EDICION = ['CNT', 'year']

# {variable: signo}: +1 si más es mejor dotado, -1 si más es peor dotado
VARIABLES_DOTACION = {'alumnos_por_profesor': -1, 'tamaño_clase': -1, 'ordenadores_por_alumno': 1}
VARIABLES_ESCASEZ = ['escasez_material_educativo', 'escasez_personal']
# 2012 no tiene `EDUSHORT` ni `STAFFSHORT`. Sus equivalentes del cuestionario de centro de 2012, con el signo que hace
# que más sea más escasez: {variable: (columna de 2012, signo)}. La calidad de los recursos educativos (`SCMATEDU`)
# mide lo contrario que la escasez de material; la escasez de profesores (`TCSHORT`), lo mismo que la de personal.
# La calidad de las instalaciones (`SCMATBUI`), que `EDUSHORT` incluye desde 2015, no se usa: falta en más centros.
ESCASEZ_2012 = {'escasez_material_educativo': ('SCMATEDU', -1), 'escasez_personal': ('TCSHORT', 1)}
# Razones muy asimétricas, con una cola larga a la derecha: se usan en logaritmo
VARIABLES_LOGARITMO = {'alumnos_por_profesor': 0.0, 'ordenadores_por_alumno': 0.01}   # {variable: lo que se suma antes del logaritmo}
PERCENTILES_RECORTE = (0.01, 0.99)


def pesos_iguales_por_grupo(datos: pd.DataFrame, claves: list[str] = GRUPO_PAIS_EDICION) -> pd.Series:
    """Peso de cada fila para que cada grupo de `claves` sume lo mismo (1 / filas del grupo)."""
    return 1 / datos.groupby(claves)[claves[0]].transform('size')


def leer_centros(ruta_raw_base: str, ruta_intermedios: str, paises: list[str]) -> pd.DataFrame:
    """Una fila por centro y edición de `paises` con sus recursos (`pisa_recursos_centro_anual.parquet`, que guarda
    `01_EDA_equidad`), con la escasez de 2012 tomada de las columnas de `ESCASEZ_2012`."""
    centros = pd.read_parquet(os.path.join(ruta_intermedios, 'recursos_anual', 'pisa_recursos_centro_anual.parquet'))
    centros[['CNT', 'CNTSCHID']] = centros[['CNT', 'CNTSCHID']].astype(str)
    centros = centros[centros['CNT'].isin(paises)]
    columnas_2012 = [columna for columna, _ in ESCASEZ_2012.values()]
    centros_2012 = pd.read_parquet(rutas_datos_anuales(ruta_raw_base, 2012)[1], columns=['CNT', 'CNTSCHID'] + columnas_2012)
    centros_2012[['CNT', 'CNTSCHID']] = centros_2012[['CNT', 'CNTSCHID']].astype(str)
    for variable, (columna, signo) in ESCASEZ_2012.items():
        centros_2012[f'{variable}_2012'] = signo * centros_2012[columna]
    centros = centros.merge(centros_2012.drop(columns=columnas_2012).assign(year=2012), on=['CNT', 'CNTSCHID', 'year'], how='left', validate='one_to_one')
    for variable in ESCASEZ_2012:
        centros[variable] = centros[variable].fillna(centros.pop(f'{variable}_2012'))
    return centros.reset_index(drop=True)


def transformar_recursos(centros: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Copia de `centros` con cada variable de `variables` recortada a sus percentiles `PERCENTILES_RECORTE`
    (calculados con todas las ediciones juntas) y, si está en `VARIABLES_LOGARITMO`, pasada a logaritmo."""
    centros = centros.copy()
    for variable in variables:
        inferior, superior = centros[variable].quantile(list(PERCENTILES_RECORTE))
        centros[variable] = centros[variable].clip(inferior, superior)
        if variable in VARIABLES_LOGARITMO:
            centros[variable] = np.log(centros[variable] + VARIABLES_LOGARITMO[variable])
    return centros


def estandarizar_conjunto(centros: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Cada variable menos su media y dividida por su desviación típica, ambas con todas las ediciones juntas y
    cada país-edición pesando igual: conserva las diferencias de nivel entre países y entre ediciones."""
    pesos = pesos_iguales_por_grupo(centros)
    estandarizadas = {}
    for variable in variables:
        con_dato = centros[variable].notna()
        media = np.average(centros.loc[con_dato, variable], weights=pesos[con_dato])
        desviacion = np.sqrt(np.average((centros.loc[con_dato, variable] - media) ** 2, weights=pesos[con_dato]))
        estandarizadas[variable] = (centros[variable] - media) / desviacion
    return pd.DataFrame(estandarizadas, index=centros.index)


def cargas_primera_componente(estandarizadas: pd.DataFrame, grupos: pd.DataFrame) -> tuple[pd.Series, float]:
    """Cargas de la primera componente principal de las columnas de `estandarizadas` (filas completas) y % de la
    varianza que explica. Se calcula con la variación dentro de cada grupo de `grupos` (país-edición), que es
    la que usa un modelo con efecto fijo de país, y con cada grupo pesando igual. El signo se elige para que
    la suma de las cargas sea positiva."""
    completas = estandarizadas.dropna()
    grupos = grupos.loc[completas.index]
    pesos = pesos_iguales_por_grupo(grupos, list(grupos.columns)).to_numpy()
    centradas = completas - completas.groupby([grupos[clave] for clave in grupos.columns]).transform('mean')
    covarianza = np.cov(centradas.to_numpy(), rowvar=False, aweights=pesos)
    valores, vectores = np.linalg.eigh(covarianza)
    cargas = vectores[:, -1] * np.sign(vectores[:, -1].sum())
    return pd.Series(cargas, index=completas.columns), 100 * valores[-1] / valores.sum()


def construir_indices(centros: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, float]:
    """Añade a `centros` (una fila por centro y edición, con `CNT` y `year`) tres índices, todos con media 0 y
    desviación típica 1 con las ediciones juntas y cada país-edición pesando igual:

    * `dotacion_centro`: primera componente principal de `VARIABLES_DOTACION` transformadas y con su signo
      (más = mejor dotado). Solo para centros con las tres variables.
    * `dotacion_centro_equiponderado`: media de las mismas tres variables estandarizadas (pesos iguales).
    * `escasez_centro`: media de `VARIABLES_ESCASEZ` (más = más escasez). Solo con las dos. Cada edición la mide con
      su propia escala, así que solo compara centros de una misma edición.

    Devuelve también las cargas de la componente principal y el % de varianza que explica."""
    variables = list(VARIABLES_DOTACION)
    transformadas = transformar_recursos(centros, variables)
    estandarizadas = estandarizar_conjunto(transformadas, variables) * pd.Series(VARIABLES_DOTACION)
    cargas, varianza = cargas_primera_componente(estandarizadas, centros[GRUPO_PAIS_EDICION])
    centros = centros.copy()
    centros['dotacion_centro'] = estandarizadas[variables] @ cargas
    centros['dotacion_centro_equiponderado'] = estandarizadas[variables].mean(axis=1, skipna=False)
    centros['escasez_centro'] = centros[VARIABLES_ESCASEZ].mean(axis=1, skipna=False)
    indices = ['dotacion_centro', 'dotacion_centro_equiponderado', 'escasez_centro']
    centros[indices] = estandarizar_conjunto(centros, indices)
    return centros, cargas, varianza
