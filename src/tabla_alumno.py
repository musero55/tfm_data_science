"""
tabla_alumno.py
===============

Tabla por alumno con las variables candidatas del modelo, unidas desde las tablas maestras y las tablas
que guardan los EDA:

    * `VARIABLES_MODELO`: variables del modelo agrupadas por su papel (compromiso, recursos, controles, comprobación).
    * `construir_skipping` / `construir_tardysd`: absentismo e impuntualidad con la regla de la OCDE, a partir
      de los ítems de asistencia (`ITEMS_ASISTENCIA`), para tenerlos en todas las ediciones.
    * `composicion_social_centro`: media en el centro del origen social de sus alumnos (padres).
    * `PANELES` / `paises_panel`: paneles de países del análisis (el común, de 12 países con las cuatro
      ediciones, y el ampliado, de 25 países sin 2018) y los países de cada uno.
    * `tabla_por_alumno`: una fila por alumno con perfil de centro de cada edición de los países de un panel,
      con identificadores, pesos, perfil, valores plausibles de matemáticas, lectura y ciencias (`DOMINIOS`),
      las variables de `VARIABLES_MODELO` y los componentes del compromiso (`COMPONENTES_COMPROMISO`).
    * `paises_edicion_sin_variable`: países-edición en los que una variable falta en todas las filas
      (pregunta no hecha); el resto de huecos de esa variable son sueltos.

Orígenes: tablas maestras (`data/raw/datos_anuales/`), `01_EDA_recursos_alumno` (subíndices del hogar),
`01_EDA_indice_de_compromiso_competencia` (índice de compromiso y competencia y horas de deberes, en
`data/intermediate/compromiso_competencia/` y `compromiso_competencia_panel_ampliado/`), `01_EDA_equidad`
(`ESCS` medio del centro y recursos del centro, de los que `src/recursos_centro.py` calcula los índices) y
`00_Pipeline_generacion_estratos` (perfil de centro, `data/intermediate/estratos_fijos/`).
"""

from __future__ import annotations

import os

import pandas as pd

from src.pisa_pipeline import DOMINIOS, columnas_pv, rutas_datos_anuales
from src.recursos_centro import construir_indices, leer_centros

CLAVES_ALUMNO = ['CNT', 'CNTSCHID', 'CNTSTUID']
COLUMNA_SEXO = {2012: 'ST04Q01', 2015: 'ST004D01T', 2018: 'ST004D01T', 2022: 'ST004D01T'}   # 1 = femenino, 2 = masculino
VALORES_PLAUSIBLES = {2012: 5, 2015: 10, 2018: 10, 2022: 10}   # todos los de cada edición; cada una se combina con los suyos

# Ítems de asistencia en las dos últimas semanas de clase: faltar un día entero, saltarse alguna clase y llegar
# tarde (1 = nunca, 2 = una o dos veces, 3 = tres o cuatro, 4 = cinco o más). En 2012 tienen otro código.
ITEMS_ASISTENCIA = {2012: ['ST09Q01', 'ST115Q01', 'ST08Q01'],
                    **{edicion: ['ST062Q01TA', 'ST062Q02TA', 'ST062Q03TA'] for edicion in (2015, 2018, 2022)}}

# Paneles de países: carpeta del índice de compromiso y competencia (`01_EDA_indice_de_compromiso_competencia`)
# y ediciones. El común reúne los 12 países con horas de deberes en las cuatro ediciones (en 2018 solo las
# preguntan ellos); el ampliado, los 25 con datos en 2012, 2015 y 2022, para la comprobación de robustez. Cada
# panel tiene su propio índice, con los terciles y la escala calculados con sus alumnos.
PANELES = {'común': ('compromiso_competencia', [2012, 2015, 2018, 2022]),
           'ampliado': ('compromiso_competencia_panel_ampliado', [2012, 2015, 2022])}

# Variables del modelo por su papel. Las de interés son el compromiso del alumno (`Indice_Compromiso_Competencia`)
# y los recursos de estudio del hogar; el perfil de centro (`perfil`) no está aquí porque define los grupos que
# se comparan. Las comprobaciones sustituyen a una variable del modelo: los tres componentes del índice
# (estar en el tercil central de deberes, `deberes_en_u`, el absentismo y la impuntualidad) al índice, el
# `ESCS` medio del centro a la composición social y la escasez según el director se añade a la dotación del
# centro (solo compara centros de una misma edición, porque cada edición la mide con su propia escala).
VARIABLES_MODELO = {
    'compromiso': ['Indice_Compromiso_Competencia'],
    'recursos del alumno': ['recursos_estudio'],
    'control: origen social': ['HISEI_TREND', 'PAREDINT_TREND', 'riqueza_hogar', 'libros'],
    'control: alumno': ['sexo_femenino', 'inmigrante'],
    'control: centro': ['composicion_social_centro', 'dotacion_centro'],
    'comprobación': ['deberes_en_u', 'SKIPPING', 'TARDYSD', 'escs_centro', 'escasez_centro'],
}
# Componentes del índice de compromiso, que se describen por edición
COMPONENTES_COMPROMISO = ['tiempo_deberes_totales_diarios', 'deberes_en_u', 'SKIPPING', 'TARDYSD']


def construir_skipping(falta_dias: pd.Series, salta_clases: pd.Series) -> pd.Series:
    """Absentismo (`SKIPPING` de la OCDE): 1 si faltó un día entero o se saltó alguna clase al menos una vez
    en las dos últimas semanas, 0 si ninguna de las dos. Vacío solo si faltan las dos respuestas."""
    absentismo = ((falta_dias > 1) | (salta_clases > 1)).astype(float)
    return absentismo.where(falta_dias.notna() | salta_clases.notna())


def construir_tardysd(llega_tarde: pd.Series) -> pd.Series:
    """Impuntualidad (`TARDYSD` de la OCDE): 0 nunca, 1 una o dos veces, 2 tres o más veces."""
    return llega_tarde.map({1: 0.0, 2: 1.0, 3: 2.0, 4: 2.0})


def composicion_social_centro(datos: pd.DataFrame) -> pd.Series:
    """Composición social del centro sin posesiones del hogar: media en el centro (`centro`), ponderada con
    `W_FSTUWT`, del origen social de sus alumnos. El origen social de cada alumno es la media de su ocupación
    (`HISEI_TREND`) y su educación de los padres (`PAREDINT_TREND`), estandarizadas con todos los alumnos de
    `datos` (con una sola de las dos, se usa esa). Centros sin ningún alumno con dato: vacío."""
    componentes = datos[['HISEI_TREND', 'PAREDINT_TREND']]
    origen_social = ((componentes - componentes.mean()) / componentes.std()).mean(axis=1)
    peso = datos['W_FSTUWT'].where(origen_social.notna(), 0)
    suma_pesos = peso.groupby(datos['centro']).transform('sum')
    return ((origen_social.fillna(0) * peso).groupby(datos['centro']).transform('sum') / suma_pesos).where(suma_pesos > 0)


def leer_compromiso(ruta_intermedios: str, panel: str = 'común') -> pd.DataFrame:
    """Índice de compromiso y competencia de un panel de `PANELES`, su tercil de deberes y las horas de deberes,
    una fila por alumno que lo tiene (los que tienen los tres componentes). `deberes_en_u` vale 1 en el tercil
    central de deberes y 0 en los otros dos, como en el índice."""
    carpeta, ediciones = PANELES[panel]
    compromiso = pd.concat([pd.read_parquet(os.path.join(ruta_intermedios, carpeta, f'pisa{edicion}_compromiso_competencia.parquet'),
                                            columns=CLAVES_ALUMNO + ['year', 'Indice_Compromiso_Competencia', 'tercil_deberes', 'tiempo_deberes_totales_diarios'])
                            for edicion in ediciones], ignore_index=True)
    compromiso[CLAVES_ALUMNO] = compromiso[CLAVES_ALUMNO].astype(str)
    compromiso['deberes_en_u'] = (compromiso['tercil_deberes'] == 2).astype(float)
    return compromiso


def paises_panel(ruta_intermedios: str, panel: str = 'común') -> list[str]:
    """Países de un panel de `PANELES`: los que tienen el índice de compromiso en todas sus ediciones."""
    ediciones = PANELES[panel][1]
    ediciones_por_pais = leer_compromiso(ruta_intermedios, panel).groupby('CNT')['year'].nunique()
    return sorted(ediciones_por_pais[ediciones_por_pais == len(ediciones)].index)


def tabla_por_alumno(ruta_raw_base: str, ruta_intermedios: str, panel: str = 'común') -> pd.DataFrame:
    """Una fila por alumno de las ediciones de un panel de `PANELES` cuyo centro tiene perfil (`perfil`, de
    `00_Pipeline_generacion_estratos`), solo de los países del panel (`paises_panel`). Columnas: `CLAVES_ALUMNO`,
    `year`, los pesos `W_FSTUWT` y `SENWT`, los valores plausibles de cada asignatura de `DOMINIOS`
    (`VALORES_PLAUSIBLES` de la edición: PV1MATH ... PV10SCIE; en 2012, PV6 a PV10 quedan vacíos), las variables
    de `VARIABLES_MODELO` y `COMPONENTES_COMPROMISO`, más el identificador del país-edición (`pais_edicion`) y del
    centro-edición (`centro`). `sexo_femenino` e `inmigrante` (primera o segunda generación, `IMMIG` 2 o 3) valen
    1 o 0. El índice de compromiso, su tercil y las horas de deberes son los del panel (vacíos en los alumnos sin
    alguno de los tres componentes). Los índices del centro (`construir_indices`) se calculan con los centros de
    los países del panel."""
    ediciones = PANELES[panel][1]
    paises = paises_panel(ruta_intermedios, panel)
    tablas = []
    for edicion in ediciones:
        columnas = (CLAVES_ALUMNO + ['W_FSTUWT', 'SENWT', 'HISEI_TREND', 'PAREDINT_TREND', 'IMMIG', COLUMNA_SEXO[edicion]]
                    + ITEMS_ASISTENCIA[edicion] + [columna for dominio in DOMINIOS for columna in columnas_pv(dominio, VALORES_PLAUSIBLES[edicion])])
        alumnos = pd.read_parquet(rutas_datos_anuales(ruta_raw_base, edicion)[0], columns=columnas)
        falta_dias, salta_clases, llega_tarde = (alumnos[item] for item in ITEMS_ASISTENCIA[edicion])
        alumnos['SKIPPING'] = construir_skipping(falta_dias, salta_clases)
        alumnos['TARDYSD'] = construir_tardysd(llega_tarde)
        alumnos['sexo_femenino'] = alumnos[COLUMNA_SEXO[edicion]].map({1: 1.0, 2: 0.0})
        alumnos['inmigrante'] = alumnos['IMMIG'].map({1: 0.0, 2: 1.0, 3: 1.0})
        tablas.append(alumnos.drop(columns=[COLUMNA_SEXO[edicion], 'IMMIG'] + ITEMS_ASISTENCIA[edicion]).assign(year=edicion))
    datos = pd.concat(tablas, ignore_index=True).astype({clave: str for clave in CLAVES_ALUMNO})
    datos = datos[datos['CNT'].isin(paises)]

    perfiles = pd.concat([pd.read_parquet(os.path.join(ruta_intermedios, 'estratos_fijos', f'pisa{edicion}_estratos_fijos.parquet'),
                                          columns=CLAVES_ALUMNO + ['PERFIL_CODE']).rename(columns={'PERFIL_CODE': 'perfil'}).assign(year=edicion)
                          for edicion in ediciones], ignore_index=True)
    recursos_alumno = pd.read_parquet(os.path.join(ruta_intermedios, 'recursos_alumno_anual', 'pisa_recursos_alumno_anual.parquet'),
                                      columns=CLAVES_ALUMNO + ['year', 'recursos_estudio', 'riqueza_hogar', 'libros'])
    escs_centro = pd.read_parquet(os.path.join(ruta_intermedios, 'escs_anual', 'pisa_escs_centro_anual.parquet'))
    for tabla in (perfiles, recursos_alumno):
        tabla[CLAVES_ALUMNO] = tabla[CLAVES_ALUMNO].astype(str)
    escs_centro[['CNT', 'CNTSCHID']] = escs_centro[['CNT', 'CNTSCHID']].astype(str)
    centros = construir_indices(leer_centros(ruta_raw_base, ruta_intermedios, paises))[0][['CNT', 'CNTSCHID', 'year', 'dotacion_centro', 'escasez_centro']]

    datos = (datos.merge(perfiles, on=CLAVES_ALUMNO + ['year'], how='inner', validate='one_to_one')
                  .merge(recursos_alumno, on=CLAVES_ALUMNO + ['year'], how='left', validate='one_to_one')
                  .merge(leer_compromiso(ruta_intermedios, panel), on=CLAVES_ALUMNO + ['year'], how='left', validate='one_to_one')
                  .merge(escs_centro, on=['CNT', 'CNTSCHID', 'year'], how='left', validate='many_to_one')
                  .merge(centros, on=['CNT', 'CNTSCHID', 'year'], how='left', validate='many_to_one')
                  .reset_index(drop=True))
    datos['pais_edicion'] = datos['CNT'] + '_' + datos['year'].astype(str)
    datos['centro'] = datos['pais_edicion'] + '_' + datos['CNTSCHID']
    datos['composicion_social_centro'] = composicion_social_centro(datos)
    return datos


def paises_edicion_sin_variable(datos: pd.DataFrame, variables: list[str]) -> dict[str, set]:
    """Para cada variable, los países-edición (`CNT`, `year`) en los que falta en todas las filas de `datos`:
    la pregunta no se hizo (o no se publicó) en ese país y edición. Esos huecos no se imputan; los demás
    huecos de la variable son sueltos."""
    paises_edicion = [datos['CNT'].astype(str), datos['year']]
    return {variable: set(datos[variable].isna().groupby(paises_edicion).all().loc[lambda falta_toda: falta_toda].index) for variable in variables}
