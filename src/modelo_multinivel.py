"""
modelo_multinivel.py
====================

Modelo multinivel lineal por alumno (alumno dentro de centro, dentro de país-edición, dentro de país)
para comparar recursos y esfuerzo. Se ajusta con `gpboost` y el optimizador `newton`: el que trae
por defecto (`lbfgs`) se para antes del óptimo y sobreestima la varianza entre países.

    * `tabla_por_alumno`: valores plausibles y `ESCS` de las tablas maestras, más las tablas por
      centro y por país que guardan los EDA (`ESCS`, recursos, presupuesto).
    * `ajustar_multinivel`: un modelo con un valor plausible (coeficientes, errores estándar y
      varianza de cada nivel).
    * `ajustar_con_valores_plausibles`: el mismo modelo con cada valor plausible, combinado con las
      reglas de Rubin.
    * `reduccion_varianza`: cuánto reduce un modelo la varianza de cada nivel respecto a otro.
"""

from __future__ import annotations

import os

import gpboost as gpb
import numpy as np
import pandas as pd

from src.pisa_pipeline import columnas_pv, rutas_datos_anuales
from src.presupuesto_pais_edicion import rellenar_con_serie_anual

# Niveles de agrupación, del más alto al más bajo; cada identificador es único en toda la tabla
NIVELES = ['pais', 'pais_edicion', 'centro']


def tabla_por_alumno(ruta_raw_base: str, ruta_intermedios: str, ediciones: list[int], dominio: str,
                     recursos_centro: list[str], variables_pais: list[str], rellenar_pais_con_serie: bool = False) -> pd.DataFrame:
    """Una fila por alumno de las `ediciones` con los valores plausibles de `dominio` (`PV1..PVn`),
    `ESCS_TREND`, `W_FSTUWT`, el `ESCS` medio de su centro, los `recursos_centro` de su centro y las
    `variables_pais` de su país-edición (de `presupuesto_pais_edicion.parquet`). Añade los
    identificadores de cada nivel (`NIVELES`): el centro incluye la edición, porque las muestras de
    centros cambian de una edición a otra. Con `rellenar_pais_con_serie`, los huecos de las variables de
    país que estén en la serie anual se rellenan con ella (`rellenar_con_serie_anual`)."""
    escs_centro = pd.read_parquet(os.path.join(ruta_intermedios, 'escs_anual', 'pisa_escs_centro_anual.parquet'))
    recursos = pd.read_parquet(os.path.join(ruta_intermedios, 'recursos_anual', 'pisa_recursos_centro_anual.parquet'),
                               columns=['CNT', 'CNTSCHID', 'year'] + recursos_centro)
    presupuesto = pd.read_parquet(os.path.join(ruta_intermedios, 'presupuesto_pais_edicion.parquet'), columns=['pais', 'edicion'] + variables_pais)
    if rellenar_pais_con_serie:
        serie_anual = pd.read_parquet(os.path.join(ruta_intermedios, 'presupuesto_serie_anual.parquet'))
        presupuesto, _ = rellenar_con_serie_anual(presupuesto, serie_anual, [variable for variable in variables_pais if variable in serie_anual.columns])
    presupuesto = presupuesto.rename(columns={'pais': 'CNT', 'edicion': 'year'})
    tablas = []
    for edicion in ediciones:
        ruta_estudiantes, _ = rutas_datos_anuales(ruta_raw_base, edicion)
        numero_valores_plausibles = 5 if edicion == 2012 else 10
        alumnos = pd.read_parquet(ruta_estudiantes, columns=['CNT', 'CNTSCHID', 'W_FSTUWT', 'ESCS_TREND'] + columnas_pv(dominio, numero_valores_plausibles))
        tablas.append(alumnos.assign(year=edicion))
    datos = pd.concat(tablas, ignore_index=True)
    datos = (datos.merge(escs_centro, on=['CNT', 'CNTSCHID', 'year'], how='left', validate='many_to_one')
                  .merge(recursos, on=['CNT', 'CNTSCHID', 'year'], how='left', validate='many_to_one')
                  .merge(presupuesto, on=['CNT', 'year'], how='left', validate='many_to_one'))
    datos['pais'] = datos['CNT'].astype(str)
    datos['pais_edicion'] = datos['pais'] + '_' + datos['year'].astype(str)
    datos['centro'] = datos['pais_edicion'] + '_' + datos['CNTSCHID'].astype(str)
    return datos


def ajustar_multinivel(datos: pd.DataFrame, columna_nota: str, variables: list[str],
                       niveles: list[str] = NIVELES, inicio_varianzas: np.ndarray | None = None) -> dict:
    """Modelo lineal con constante, `variables` como efectos fijos y una ordenada aleatoria por cada
    nivel de `niveles`, sobre las filas sin nulos. Devuelve coeficientes, errores estándar, varianza
    de cada nivel (y la residual, entre alumnos del mismo centro) y el número de filas."""
    datos = datos.dropna(subset=[columna_nota] + variables)
    matriz = np.column_stack([np.ones(len(datos))] + [datos[variable].to_numpy(dtype=float) for variable in variables])
    modelo = gpb.GPModel(group_data=datos[niveles].astype(str).to_numpy(), likelihood='gaussian')
    parametros = {'optimizer_cov': 'newton'}
    if inicio_varianzas is not None:
        parametros['init_cov_pars'] = inicio_varianzas
    modelo.fit(y=datos[columna_nota].to_numpy(dtype=float), X=matriz, params=parametros)
    coeficientes = modelo.get_coef(std_err=True)
    nombres = ['constante'] + variables
    varianzas = modelo.get_cov_pars().to_numpy().ravel()
    return {'coeficientes': pd.Series(coeficientes.iloc[0].to_numpy(), index=nombres),
            'errores': pd.Series(coeficientes.iloc[1].to_numpy(), index=nombres),
            'varianzas': pd.Series(varianzas, index=['alumno'] + niveles),
            'filas': len(datos)}


def combinar_rubin(ajustes: list[dict]) -> pd.DataFrame:
    """Combina los ajustes de cada valor plausible con las reglas de Rubin: el coeficiente es la media,
    y su varianza, la media de las varianzas de cada ajuste más (1 + 1/m) veces la varianza entre ajustes."""
    coeficientes = pd.DataFrame([ajuste['coeficientes'] for ajuste in ajustes])
    varianzas_muestrales = pd.DataFrame([ajuste['errores'] ** 2 for ajuste in ajustes])
    numero = len(ajustes)
    error = np.sqrt(varianzas_muestrales.mean() + (1 + 1 / numero) * coeficientes.var(ddof=1))
    return pd.DataFrame({'coeficiente': coeficientes.mean(), 'error_estandar': error,
                         'z': coeficientes.mean() / error, 'variacion_entre_valores_plausibles': coeficientes.std(ddof=1)})


def ajustar_con_valores_plausibles(datos: pd.DataFrame, columnas_nota: list[str], variables: list[str],
                                   niveles: list[str] = NIVELES) -> dict:
    """`ajustar_multinivel` con cada valor plausible de `columnas_nota`; coeficientes combinados con
    `combinar_rubin` y varianza de cada nivel promediada entre valores plausibles."""
    ajustes = [ajustar_multinivel(datos, columna, variables, niveles) for columna in columnas_nota]
    return {'coeficientes': combinar_rubin(ajustes),
            'varianzas': pd.DataFrame([ajuste['varianzas'] for ajuste in ajustes]).mean(),
            'filas': ajustes[0]['filas']}


def reduccion_varianza(varianzas_modelo: pd.Series, varianzas_referencia: pd.Series) -> pd.Series:
    """% de la varianza de cada nivel del modelo de referencia que explica el modelo (puede ser
    negativo: añadir variables de un nivel puede aumentar la varianza estimada de otro)."""
    return 100 * (1 - varianzas_modelo / varianzas_referencia)
