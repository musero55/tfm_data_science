"""
pisa_estadisticas.py
====================

Estadísticas sobre las tablas maestras de PISA (`data/raw/datos_anuales/`, ver `pisa_pipeline.py`):

    * `porcentaje_nulos` / `porcentaje_nulos_por_pais`: nulos por columna y por país.
    * `agregar_media_ponderada` / `agregar_desviacion_ponderada` / `agregar_pv`: medias y desviaciones
      típicas ponderadas por país, estrato o centro (ver `CLAVES_NIVEL`); `agregar_pv` da la nota media
      de un dominio.
    * `reparto_varianza`: % de la varianza de una variable entre niveles anidados (por ejemplo,
      entre países, entre centros del país y entre alumnos del centro).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pisa_pipeline import columnas_pv


# 1. Nulos

def porcentaje_nulos(datos: pd.DataFrame) -> pd.Series:
    """% de nulos por columna, de mayor a menor."""
    return (datos.isnull().sum() / len(datos) * 100).sort_values(ascending=False)


def porcentaje_nulos_por_pais(datos: pd.DataFrame, columna: str, pais_col: str = 'CNT') -> pd.DataFrame:
    """% de nulos de `columna` en cada país, de mayor a menor."""
    resultado = datos[columna].isna().groupby(datos[pais_col]).mean() * 100
    return (
        resultado.reset_index(name=f'pct_nulos_{columna}')
        .sort_values(f'pct_nulos_{columna}', ascending=False)
    )


# 2. Medias y desviaciones ponderadas por país, estrato o centro

# Claves de agrupación de cada nivel. En 2012 `CNTSCHID` se repite entre países: el centro
# se identifica siempre con `CNT` + `CNTSCHID`. Por alumno no se agrega: cada alumno es una
# fila de las tablas maestras, con sus valores plausibles
CLAVES_NIVEL = {
    'pais': ['CNT'],
    'estrato': ['CNT', 'STRATUM'],
    'centro': ['CNT', 'CNTSCHID'],
}


def agregar_media_ponderada(
    datos: pd.DataFrame,
    columnas: list[str],
    claves: list[str] | str = 'pais',
    peso_col: str = 'W_FSTUWT',
) -> pd.DataFrame:
    """Media ponderada por `peso_col` de cada columna de `columnas`, agrupando por `claves`.

    `claves` es un nivel de `CLAVES_NIVEL` ('pais', 'estrato' o 'centro') o una
    lista de columnas. En cada columna solo cuentan los alumnos con valor: su peso no entra
    en el denominador si no tienen dato. Devuelve una fila por grupo, con las claves como columnas.
    """
    claves = CLAVES_NIVEL[claves] if isinstance(claves, str) else list(claves)
    valores = datos[columnas]
    pesos_con_dato = valores.notna().mul(datos[peso_col], axis=0)
    suma_ponderada = valores.mul(datos[peso_col], axis=0).groupby([datos[clave] for clave in claves]).sum(min_count=1)
    suma_pesos = pesos_con_dato.groupby([datos[clave] for clave in claves]).sum()
    return (suma_ponderada / suma_pesos.where(suma_pesos > 0)).reset_index()


def agregar_desviacion_ponderada(
    datos: pd.DataFrame,
    columnas: list[str],
    claves: list[str] | str = 'pais',
    peso_col: str = 'W_FSTUWT',
) -> pd.DataFrame:
    """Desviación típica ponderada por `peso_col` de cada columna de `columnas`, agrupando por `claves`:
    la raíz de la media ponderada de los cuadrados de las desviaciones respecto a la media ponderada
    del grupo (ver `agregar_media_ponderada`, con las mismas reglas para claves y nulos)."""
    claves = CLAVES_NIVEL[claves] if isinstance(claves, str) else list(claves)
    medias_por_fila = datos[claves].merge(agregar_media_ponderada(datos, columnas, claves, peso_col), on=claves, how='left')
    cuadrados = (datos[columnas].to_numpy(dtype=float) - medias_por_fila[columnas].to_numpy(dtype=float)) ** 2
    datos_cuadrados = datos[claves + [peso_col]].assign(**{columna: cuadrados[:, posicion] for posicion, columna in enumerate(columnas)})
    varianzas = agregar_media_ponderada(datos_cuadrados, columnas, claves, peso_col)
    return varianzas.assign(**{columna: np.sqrt(varianzas[columna]) for columna in columnas})


def agregar_pv(
    datos: pd.DataFrame,
    dominio: str,
    claves: list[str] | str = 'pais',
    peso_col: str = 'W_FSTUWT',
    nombre: str | None = None,
) -> pd.DataFrame:
    """Nota media ponderada de un dominio (MATH, READ o SCIE) por grupo.

    Promedia los valores plausibles de cada alumno y calcula su media ponderada por `peso_col`
    en cada grupo de `claves` (ver `agregar_media_ponderada`). La columna resultante se llama
    `nombre` (por defecto `media_<dominio>`).
    """
    columnas_pv_dominio = [columna for columna in columnas_pv(dominio) if columna in datos.columns]
    if not columnas_pv_dominio:
        raise ValueError(f'Los datos no tienen valores plausibles de {dominio}')
    nombre = nombre or f'media_{dominio.lower()}'
    datos_alumno = datos.assign(**{nombre: datos[columnas_pv_dominio].mean(axis=1)})
    return agregar_media_ponderada(datos_alumno, [nombre], claves=claves, peso_col=peso_col)


# 3. Reparto de la varianza entre niveles anidados

def reparto_varianza(
    datos: pd.DataFrame,
    columna: str,
    niveles: dict[str, list[str]],
    nombre_residuo: str,
    peso_col: str | None = None,
) -> pd.Series:
    """% de la varianza de `columna` repartida entre niveles anidados (sumas de cuadrados).

    `niveles` es {nombre del componente: claves de agrupación}, del nivel más alto al más bajo y
    cada uno anidado en el anterior; por ejemplo {'entre países': ['CNT'], 'entre centros (dentro
    del país)': ['CNT', 'CNTSCHID']}. Cada componente es la variación de las medias de su nivel
    respecto a las del nivel anterior, y `nombre_residuo`, la variación dentro del nivel más bajo.
    Los componentes suman 100. Con `peso_col` las medias y las sumas de cuadrados se ponderan; sin
    él, cada fila pesa igual. Solo cuentan las filas con todas las columnas necesarias.
    """
    claves = list(dict.fromkeys(clave for claves_nivel in niveles.values() for clave in claves_nivel))
    datos = datos[claves + [columna] + ([peso_col] if peso_col else [])].dropna()
    valores = datos[columna]
    pesos = datos[peso_col] if peso_col else pd.Series(1.0, index=datos.index)
    media_nivel_anterior = pd.Series(np.average(valores, weights=pesos), index=datos.index)
    suma_total = (pesos * (valores - media_nivel_anterior) ** 2).sum()
    reparto = {}
    for nombre, claves_nivel in niveles.items():
        grupos = [datos[clave] for clave in claves_nivel]
        media_nivel = (valores * pesos).groupby(grupos).transform('sum') / pesos.groupby(grupos).transform('sum')
        reparto[nombre] = 100 * (pesos * (media_nivel - media_nivel_anterior) ** 2).sum() / suma_total
        media_nivel_anterior = media_nivel
    reparto[nombre_residuo] = 100 * (pesos * (valores - media_nivel_anterior) ** 2).sum() / suma_total
    return pd.Series(reparto)
