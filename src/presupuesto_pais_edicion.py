"""
presupuesto_pais_edicion.py
===========================

Construye la tabla país-edición para el análisis de presupuesto: una fila por país
(33 OCDE) y edición PISA (2012, 2015, 2018, 2022), con resultados y contexto de PISA
y las variables externas de gasto, renta y personal de `data/raw/externos/`.

La tabla se divide en dos (`dividir_tabla`): la **económica** (gasto, financiación,
composición, salarios, renta y dotación de personal, más los resultados de PISA y el
`ESCS` medio como control) y el **contexto educativo** (variables no económicas:
segregación, autonomía, tiempo de aprendizaje, equidad de recursos, perfil del
profesorado, escolarización...), que se guarda aparte.

Las fuentes están descritas en `01_EDA_presupuestos.ipynb`. Los huecos se dejan como
NaN: nada se imputa (los ceros que en realidad son datos no disponibles se pasan a NaN). Solo se interpolan linealmente los huecos *interiores* de una
serie anual para calcular las medias de años previos y el gasto acumulado.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.datos_externos import cargar_oecd
from src.pisa_pipeline import DOMINIOS, PAISES_OCDE_TODAS_LAS_EDICIONES, columnas_pv, rutas_datos_anuales

EDICIONES = [2012, 2015, 2018, 2022]
AÑOS_SERIE = range(1995, 2025)
AÑO_BASE = 2020   # año de los precios constantes (el mismo que usa la OCDE en el gasto por alumno)

# Financiación del centro (% por origen): 2012 usa otros nombres
COLUMNAS_FINANCIACION = {
    'pisa_financiacion_centro_gobierno_pct': ('SC02Q01', 'SC016Q01TA'),
    'pisa_financiacion_centro_cuotas_pct': ('SC02Q02', 'SC016Q02TA'),
    'pisa_financiacion_centro_donaciones_pct': ('SC02Q03', 'SC016Q03TA'),
    'pisa_financiacion_centro_otras_pct': ('SC02Q04', 'SC016Q04TA'),
}

# Recursos de centro cuya distribución entre centros desfavorecidos y favorecidos se mide
RECURSOS_BRECHA = ['alumnos_por_profesor', 'tamaño_clase', 'profesorado_certificado', 'escasez_material', 'escasez_personal']

# Nombre de cada dominio y de cada nivel de la OCDE en las columnas de la tabla
NOMBRE_DOMINIO = {'MATH': 'matematicas', 'READ': 'lectura', 'SCIE': 'ciencias'}
NOMBRE_NIVEL_GASTO = {'1': 'primaria', '2': 'secundaria_primera_etapa', '3': 'secundaria_segunda_etapa', '2_3': 'secundaria'}


# ---------------------------------------------------------------- clasificación de columnas

CLAVE = ['pais', 'edicion']
# Resultados de PISA y control social (necesarios para cualquier análisis del gasto)
RESULTADOS = ['pisa_matematicas', 'pisa_lectura', 'pisa_ciencias', 'pisa_numero_alumnos', 'pisa_escs_trend_media']
# Variables NO económicas: contexto educativo (controles o mediadores posibles). Para mover una
# variable a la tabla económica basta con quitarla de esta lista.
CONTEXTO_EDUCATIVO = [
    'poblacion', 'pisa_numero_centros', 'pisa_escs_trend_desviacion_tipica', 'pisa_segregacion_social_entre_centros_pct',
    'pisa_minutos_semanales_matematicas', 'pisa_minutos_semanales_lengua', 'pisa_minutos_semanales_ciencias',
    'pisa_alumnos_por_profesor', 'pisa_tamaño_clase', 'pisa_tamaño_centro', 'pisa_centros_publicos_pct',
    'pisa_profesores_certificados_pct', 'pisa_profesores_master_pct', 'pisa_ordenadores_por_alumno',
    'pisa_autonomia_recursos', 'pisa_escasez_material_educativo', 'pisa_escasez_personal',
    'pisa_brecha_alumnos_por_profesor', 'pisa_brecha_tamaño_clase', 'pisa_brecha_profesorado_certificado', 'pisa_brecha_escasez_material', 'pisa_brecha_escasez_personal',
    'escolarizacion_3_años_pct', 'escolarizacion_4_años_pct', 'escolarizacion_15_años_pct',
    'repeticion_pct_secundaria_primera_etapa', 'docentes_50_o_mas_años_pct', 'docentes_menos_de_30_años_pct', 'tamaño_clase_secundaria_primera_etapa', 'horas_docencia_anuales_secundaria_primera_etapa',
    'gini', 'gini_fuente', 'gini_año',
]


# Selección que se guarda en el parquet (solo fuentes externas). Son variables **básicas** para el
# feature engineering posterior: no se incluyen razones ni sumas de otras seleccionadas (se calculan
# después), y todas están en unidades comparables entre países y años (% o USD PPP constantes de 2020).
SELECCION = [
    # Gasto por alumno (USD PPP, precios constantes de 2020)
    'gasto_alumno_preprimaria_usd_ppp', 'gasto_alumno_primaria_usd_ppp', 'gasto_alumno_secundaria_primera_etapa_usd_ppp', 'gasto_alumno_secundaria_segunda_etapa_usd_ppp', 'gasto_alumno_secundaria_usd_ppp',
    'gasto_alumno_secundaria_servicios_auxiliares_usd_ppp',
    # Gasto total de secundaria 1.ª+2.ª (USD PPP constantes de 2020)
    'gasto_total_corriente_secundaria_usd_ppp', 'gasto_total_capital_secundaria_usd_ppp', 'gasto_total_personal_secundaria_usd_ppp', 'gasto_total_docentes_secundaria_usd_ppp',
    # Esfuerzo y origen de la financiación (%)
    'gasto_pct_pib_primaria_a_postsecundaria', 'pct_gasto_publico_en_primaria_a_postsecundaria', 'gasto_publico_total_pct_pib',
    'financiacion_privada_pct_gasto_secundaria', 'financiacion_exterior_pct_gasto_secundaria',
    # Docentes (salario en USD PPP constantes de 2020)
    'salario_legal_docente_inicial_usd_ppp', 'salario_legal_docente_15_años_experiencia_usd_ppp', 'alumnos_por_docente_secundaria_primera_etapa',
    # Renta (USD PPP constantes de 2020) y tamaño
    'pib_per_capita_usd_ppp', 'poblacion',
    # Desigualdad de renta (0-1; OCDE y, si falta, Banco Mundial; ver `gini_edicion`)
    'gini',
]


def dividir_tabla(tabla: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa la tabla en (económica, contexto educativo). Ambas llevan `pais` y `edicion`;
    la económica incluye además los resultados de PISA y el `ESCS` medio. Todo lo que no
    esté en `CONTEXTO_EDUCATIVO` ni en `RESULTADOS` se considera económico."""
    desconocidas = [columna for columna in CONTEXTO_EDUCATIVO + RESULTADOS if columna not in tabla.columns]
    assert not desconocidas, f'columnas clasificadas que no existen: {desconocidas}'
    economicas = [columna for columna in tabla.columns if columna not in CLAVE + RESULTADOS + CONTEXTO_EDUCATIVO]
    return (tabla[CLAVE + RESULTADOS + economicas].copy(), tabla[CLAVE + CONTEXTO_EDUCATIVO].copy())


def media_ponderada(valores: pd.Series, pesos: pd.Series) -> float:
    """Media de `valores` ponderada por `pesos`, ignorando nulos. NaN si no hay datos."""
    filas_validas = valores.notna() & pesos.notna()
    return float(np.average(valores[filas_validas], weights=pesos[filas_validas])) if filas_validas.any() else np.nan


def desviacion_ponderada(valores: pd.Series, pesos: pd.Series) -> float:
    filas_validas = valores.notna() & pesos.notna()
    if not filas_validas.any():
        return np.nan
    media = np.average(valores[filas_validas], weights=pesos[filas_validas])
    return float(np.sqrt(np.average((valores[filas_validas] - media) ** 2, weights=pesos[filas_validas])))


def primera(nombres, *candidatas: str) -> str | None:
    """Primera de `candidatas` que exista en `nombres` (las ediciones nombran distinto)."""
    return next((candidata for candidata in candidatas if candidata in nombres), None)


# ---------------------------------------------------------------- PISA

def resultados_pisa(ruta_raw_base: str, año: int) -> pd.DataFrame:
    """Por país: media de cada dominio (PV promediados por alumno, ponderados por
    `W_FSTUWT`), nº de alumnos, media y desviación de `ESCS_TREND`, minutos semanales
    de aprendizaje por asignatura (2012-2018; los ceros se tratan como dato ausente)
    y segregación socioeconómica (% de la varianza de `ESCS_TREND` entre centros)."""
    ruta, _ = rutas_datos_anuales(ruta_raw_base, año)
    nombres = pq.ParquetFile(ruta).schema.names
    columnas_valores_plausibles = [columna for dominio in DOMINIOS for columna in columnas_pv(dominio) if columna in nombres]
    minutos = {'pisa_minutos_semanales_matematicas': 'MMINS', 'pisa_minutos_semanales_lengua': 'LMINS', 'pisa_minutos_semanales_ciencias': 'SMINS'}
    minutos = {k: v for k, v in minutos.items() if v in nombres}
    estudiantes = pd.read_parquet(ruta, columns=['CNT', 'CNTSCHID', 'W_FSTUWT', 'ESCS_TREND'] + columnas_valores_plausibles + list(minutos.values()))
    filas = []
    for pais, estudiantes_pais in estudiantes.groupby('CNT'):
        fila = {'pais': pais, 'pisa_numero_alumnos': len(estudiantes_pais)}
        for dominio in DOMINIOS:
            columnas_dominio = [columna for columna in columnas_pv(dominio) if columna in estudiantes_pais.columns]
            fila[f'pisa_{NOMBRE_DOMINIO[dominio]}'] = media_ponderada(estudiantes_pais[columnas_dominio].mean(axis=1), estudiantes_pais['W_FSTUWT'])
        fila['pisa_escs_trend_media'] = media_ponderada(estudiantes_pais['ESCS_TREND'], estudiantes_pais['W_FSTUWT'])
        fila['pisa_escs_trend_desviacion_tipica'] = desviacion_ponderada(estudiantes_pais['ESCS_TREND'], estudiantes_pais['W_FSTUWT'])
        for nuevo, origen in minutos.items():
            fila[nuevo] = media_ponderada(estudiantes_pais[origen].where(estudiantes_pais[origen] > 0), estudiantes_pais['W_FSTUWT'])
        fila['pisa_segregacion_social_entre_centros_pct'] = segregacion(estudiantes_pais)
        filas.append(fila)
    return pd.DataFrame(filas)


def segregacion(estudiantes_pais: pd.DataFrame) -> float:
    """% de la varianza de `ESCS_TREND` que es entre centros (0 = todos los centros
    igual de mixtos; cuanto más alto, más separados por nivel socioeconómico)."""
    estudiantes_pais = estudiantes_pais.dropna(subset=['ESCS_TREND', 'CNTSCHID', 'W_FSTUWT'])
    if estudiantes_pais.empty:
        return np.nan
    pesos, escs = estudiantes_pais['W_FSTUWT'], estudiantes_pais['ESCS_TREND']
    media = np.average(escs, weights=pesos)
    total = np.average((escs - media) ** 2, weights=pesos)
    por_centro = estudiantes_pais.assign(_peso_por_escs=pesos * escs).groupby('CNTSCHID').agg(_peso_total=('W_FSTUWT', 'sum'), _peso_por_escs=('_peso_por_escs', 'sum'))
    por_centro['_escs_medio_centro'] = por_centro['_peso_por_escs'] / por_centro['_peso_total']
    entre = np.average((por_centro['_escs_medio_centro'] - media) ** 2, weights=por_centro['_peso_total'])
    return float(100 * entre / total) if total > 0 else np.nan


def escs_de_centros(ruta_raw_base: str, año: int) -> pd.DataFrame:
    """Media de `ESCS_TREND` de los alumnos de cada centro (ponderada por `W_FSTUWT`)."""
    ruta, _ = rutas_datos_anuales(ruta_raw_base, año)
    estudiantes = pd.read_parquet(ruta, columns=['CNT', 'CNTSCHID', 'W_FSTUWT', 'ESCS_TREND']).dropna(subset=['ESCS_TREND'])
    estudiantes = estudiantes.assign(_peso_por_escs=estudiantes['W_FSTUWT'] * estudiantes['ESCS_TREND'])
    escs_por_centro = estudiantes.groupby(['CNT', 'CNTSCHID']).agg(_peso_total=('W_FSTUWT', 'sum'), _peso_por_escs=('_peso_por_escs', 'sum'))
    return (escs_por_centro['_peso_por_escs'] / escs_por_centro['_peso_total']).rename('escs_centro').reset_index()


def centros_pisa(ruta_raw_base: str, año: int) -> pd.DataFrame:
    """Por país, con el peso del centro (`W_SCHGRNRABWT`): medias de recursos y
    financiación del centro, % de centros públicos (`SCHLTYPE` = 3), autonomía sobre
    recursos y **brecha de recursos** entre centros desfavorecidos y favorecidos
    (cuartil inferior menos superior de `ESCS_TREND` medio del centro). Un valor
    negativo en ratio, tamaño de clase o escasez significa que los desfavorecidos
    están mejor dotados; positivo en profesorado certificado, al revés."""
    _, ruta = rutas_datos_anuales(ruta_raw_base, año)
    nombres = pq.ParquetFile(ruta).schema.names
    financiacion = {k: primera(nombres, *v) for k, v in COLUMNAS_FINANCIACION.items()}
    renombrar = {  # nombre canónico -> columna de cada edición
        'CERT': primera(nombres, 'PROPCERT', 'PROATCE'),
        'ORDENADORES': primera(nombres, 'RATCMP1', 'RATCMP15'),
        'AUTONOMIA': primera(nombres, 'RESPRES', 'SRESPRES'),
        'MASTER': primera(nombres, 'PROPAT7'),
    }
    columnas_a_leer = ['CNT', 'CNTSCHID', 'W_SCHGRNRABWT', 'STRATIO', 'CLSIZE', 'SCHSIZE', 'SCHLTYPE', 'EDUSHORT', 'STAFFSHORT']
    columnas_a_leer = [columna for columna in columnas_a_leer if columna in nombres] + [columna for columna in {**financiacion, **renombrar}.values() if columna]
    colegios = pd.read_parquet(ruta, columns=list(dict.fromkeys(columnas_a_leer)))
    colegios = colegios.rename(columns={v: k for k, v in renombrar.items() if v})
    for columna in ('CERT', 'MASTER'):   # proporciones (0-1) -> %
        if columna in colegios:
            colegios[columna] = 100 * colegios[columna]
    colegios = colegios.merge(escs_de_centros(ruta_raw_base, año), on=['CNT', 'CNTSCHID'], how='left')

    filas = []
    for pais, centros_pais in colegios.groupby('CNT'):
        pesos_centros = centros_pais['W_SCHGRNRABWT']
        fila = {'pais': pais, 'pisa_numero_centros': len(centros_pais),
                'pisa_alumnos_por_profesor': media_ponderada(centros_pais['STRATIO'], pesos_centros),
                'pisa_tamaño_clase': media_ponderada(centros_pais['CLSIZE'], pesos_centros),
                'pisa_tamaño_centro': media_ponderada(centros_pais['SCHSIZE'], pesos_centros),
                'pisa_centros_publicos_pct': 100 * media_ponderada((centros_pais['SCHLTYPE'] == 3).where(centros_pais['SCHLTYPE'].notna()).astype(float), pesos_centros)}
        for nuevo, origen in financiacion.items():
            fila[nuevo] = media_ponderada(centros_pais[origen], pesos_centros) if origen else np.nan
        for nuevo, origen in (('pisa_profesores_certificados_pct', 'CERT'), ('pisa_profesores_master_pct', 'MASTER'),
                              ('pisa_ordenadores_por_alumno', 'ORDENADORES'), ('pisa_autonomia_recursos', 'AUTONOMIA'),
                              ('pisa_escasez_material_educativo', 'EDUSHORT'), ('pisa_escasez_personal', 'STAFFSHORT')):
            fila[nuevo] = media_ponderada(centros_pais[origen], pesos_centros) if origen in centros_pais else np.nan
        # Brecha de recursos entre el cuartil de centros más desfavorecido y el más favorecido
        cuartil_inferior, cuartil_superior = centros_pais['escs_centro'].quantile([0.25, 0.75])
        centros_desfavorecidos, centros_favorecidos = centros_pais[centros_pais['escs_centro'] <= cuartil_inferior], centros_pais[centros_pais['escs_centro'] >= cuartil_superior]
        for nombre, origen in (('alumnos_por_profesor', 'STRATIO'), ('tamaño_clase', 'CLSIZE'), ('profesorado_certificado', 'CERT'),
                               ('escasez_material', 'EDUSHORT'), ('escasez_personal', 'STAFFSHORT')):
            if origen in centros_pais:
                fila[f'pisa_brecha_{nombre}'] = (media_ponderada(centros_desfavorecidos[origen], centros_desfavorecidos['W_SCHGRNRABWT'])
                                                 - media_ponderada(centros_favorecidos[origen], centros_favorecidos['W_SCHGRNRABWT']))
            else:
                fila[f'pisa_brecha_{nombre}'] = np.nan
        filas.append(fila)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- series externas

def serie(datos: pd.DataFrame, **filtros) -> pd.DataFrame:
    """Serie anual (filas: años; columnas: países) con los filtros dados. Falla si
    quedan duplicados país-año, señal de que falta un filtro."""
    for k, v in filtros.items():
        datos = datos[datos[k] == v]
    assert not datos.duplicated(['pais', 'año']).any(), f'duplicados con {filtros}'
    return datos.pivot(index='año', columns='pais', values='valor').reindex(AÑOS_SERIE)


def a_usd_ppp(xdc_corrientes: pd.DataFrame, deflactor: pd.DataFrame, factor_ppp: pd.DataFrame, base: int = AÑO_BASE) -> pd.DataFrame:
    """Series anuales en moneda nacional a precios corrientes -> USD PPP a precios constantes de `base`
    (el método de la OCDE): se deflactan con el deflactor del PIB del país y se convierten con su
    factor PPP del año `base`. Así son comparables entre países y años, y con el gasto por alumno."""
    return xdc_corrientes * (deflactor.loc[base] / deflactor) / factor_ppp.loc[base]


def valor_en_año(serie_anual: pd.DataFrame, año: int) -> pd.Series:
    """Valor de cada país en un año (NaN si no hay)."""
    return serie_anual.loc[año]


def interpolada(serie_anual: pd.DataFrame) -> pd.DataFrame:
    """Rellena linealmente los huecos *interiores* de cada país (nunca extrapola)."""
    return serie_anual.interpolate(limit_area='inside')


def media_previa(serie_anual: pd.DataFrame, año: int, ventana: int = 4, minimo: int = 3) -> pd.Series:
    """Media de los `ventana` años hasta `año` (incluido), sobre la serie interpolada.
    NaN si hay menos de `minimo` años con dato."""
    ventana_interpolada = interpolada(serie_anual).loc[año - ventana + 1:año]
    return ventana_interpolada.mean().where(ventana_interpolada.notna().sum() >= minimo)


def acumulado_6_15(primaria: pd.DataFrame, secundaria1: pd.DataFrame, año: int) -> pd.Series:
    """Gasto por alumno acumulado entre los 6 y los 15 años de la cohorte que hace PISA
    en `año`: 6 años de primaria (t-9 a t-4) y 4 de secundaria 1.ª etapa (t-3 a t),
    sobre series interpoladas. Aproximado (no usa la duración teórica de cada país).
    NaN si falta algún año."""
    gasto_primaria = interpolada(primaria).loc[año - 9:año - 4]
    gasto_secundaria = interpolada(secundaria1).loc[año - 3:año]
    completo = (gasto_primaria.notna().sum() == 6) & (gasto_secundaria.notna().sum() == 4)
    return (gasto_primaria.sum() + gasto_secundaria.sum()).where(completo)


def gini_edicion(gini_ocde: pd.DataFrame, gini_banco_mundial: pd.DataFrame, año: int) -> pd.DataFrame:
    """Gini (0-1) de la renta disponible de cada país en la edición `año`. Se toma la observación más cercana al
    año (como mucho ±1); a igual distancia, la de la OCDE antes que la del Banco Mundial (que va de 0 a 100) y,
    si aún empatan, el año anterior. Devuelve `gini`, `gini_fuente` y `gini_año`, indexado por país."""
    gini_ocde_filas = gini_ocde[['pais', 'año', 'valor']].assign(fuente='OCDE', prioridad=0)
    gini_banco_mundial_filas = gini_banco_mundial.dropna(subset=['valor']).assign(valor=lambda datos: datos['valor'] / 100, fuente='Banco Mundial', prioridad=1)
    observaciones = pd.concat([gini_ocde_filas, gini_banco_mundial_filas[['pais', 'año', 'valor', 'fuente', 'prioridad']]])
    observaciones['año'] = observaciones['año'].astype(int)
    observaciones = observaciones[observaciones['año'].between(año - 1, año + 1)].assign(distancia=lambda datos: (datos['año'] - año).abs(), posterior=lambda datos: datos['año'] > año)
    mejor = observaciones.sort_values(['pais', 'distancia', 'prioridad', 'posterior']).groupby('pais').head(1).set_index('pais')
    return mejor[['valor', 'fuente', 'año']].rename(columns={'valor': 'gini', 'fuente': 'gini_fuente', 'año': 'gini_año'})


def variables_externas(datos: dict, año: int) -> pd.DataFrame:
    """Variables de presupuesto, renta y personal de cada país en la edición `año`.
    `datos`: dict con los DataFrames cargados con `cargar_oecd` (ver `construir_tabla`)."""
    gasto_por_alumno = datos['gasto']
    base = dict(precios='Q', unidad='USD_PPP_ST', fuente='_T', institucion='INST_EDU', tipo_gasto='DIR_EXP')
    por_nivel = {nivel: serie(gasto_por_alumno, nivel=f'ISCED11_{nivel}', **base) for nivel in ('1', '2', '3', '2_3')}

    variables = {}
    # --- Cuánto: gasto por alumno (USD PPP, precios constantes), en el año de la prueba
    for nivel, serie_nivel in por_nivel.items():
        variables[f'gasto_alumno_{NOMBRE_NIVEL_GASTO[nivel]}_usd_ppp'] = valor_en_año(serie_nivel, año)
    variables['gasto_alumno_secundaria_media_4_años_usd_ppp'] = media_previa(por_nivel['2_3'], año)
    variables['gasto_acumulado_alumno_6_a_15_años_usd_ppp'] = acumulado_6_15(por_nivel['1'], por_nivel['2'], año)
    variables['gasto_alumno_secundaria_pct_pib_per_capita'] = valor_en_año(serie(
        gasto_por_alumno, nivel='ISCED11_2_3', unidad='PT_B1GQ_POP', fuente='_T', institucion='INST_EDU', tipo_gasto='DIR_EXP'), año)

    # --- Esfuerzo y origen de la financiación
    gasto_pib_y_fuentes = datos['gasto_pib']
    variables['gasto_pct_pib_primaria_a_postsecundaria'] = valor_en_año(serie(gasto_pib_y_fuentes, nivel='ISCED11_1T4', unidad='PT_B1GQ', fuente='_T', precios='_Z'), año)
    for nombre, fuente in (('publica', 'S13'), ('privada', 'S1D_NON_EDU'), ('exterior', 'S2')):
        variables[f'financiacion_{nombre}_pct_gasto_secundaria'] = valor_en_año(serie(gasto_pib_y_fuentes, nivel='ISCED11_2_3', unidad='PT_EXP', fuente=fuente, precios='_Z'), año)
    variables['pct_gasto_publico_en_primaria_a_postsecundaria'] = valor_en_año(serie(datos['gasto_pct_publico'], nivel='ISCED11_1T4'), año)

    # --- Cómo: composición del gasto total de secundaria 1.ª+2.ª (corriente + capital)
    comun = dict(nivel='ISCED11_2_3', unidad='USD_PPP', precios='Q', fuente='_T', institucion='INST_EDU')
    corriente = serie(datos['gasto_corriente_capital'], tipo_gasto='CUR', **comun)
    capital = serie(datos['gasto_corriente_capital'], tipo_gasto='CAP', **comun)
    total = corriente + capital
    personal = serie(datos['gasto_personal'], tipo_gasto='CUR_COMP', **comun)
    docentes = serie(datos['gasto_personal'], tipo_gasto='CUR_COMPT', **comun)
    no_docentes = serie(datos['gasto_personal'], tipo_gasto='CUR_COMPO', **comun)
    variables['pct_gasto_secundaria_en_personal'] = 100 * valor_en_año(personal / total, año)
    variables['pct_gasto_secundaria_en_docentes'] = 100 * valor_en_año(docentes / total, año)
    variables['pct_gasto_secundaria_en_personal_no_docente'] = (100 * valor_en_año(no_docentes / total, año)).where(lambda valores: valores != 0)   # 0 = incluido en docentes
    variables['pct_gasto_secundaria_en_capital'] = 100 * valor_en_año(capital / total, año)
    # Gasto total (no por alumno) de secundaria 1.ª+2.ª en USD PPP constantes (la OCDE lo da en millones; 0 = no disponible)
    for nombre, serie_nivel in (('corriente', corriente), ('capital', capital), ('personal', personal),
                      ('docentes', docentes), ('personal_no_docente', no_docentes)):
        variables[f'gasto_total_{nombre}_secundaria_usd_ppp'] = (1e6 * valor_en_año(serie_nivel, año)).where(lambda valores: valores != 0)
    variables['pct_gasto_secundaria_en_otros_corrientes'] = 100 - variables['pct_gasto_secundaria_en_personal'] - variables['pct_gasto_secundaria_en_capital']

    # Servicios básicos frente a auxiliares (transporte, comidas, alojamiento)
    tipo = datos['gasto_tipo']
    base_t = dict(nivel='ISCED11_2_3', unidad='USD_PPP_ST')
    basicos, auxiliares = serie(tipo, tipo_gasto='CORE', **base_t), serie(tipo, tipo_gasto='ASERV', **base_t)
    variables['servicios_auxiliares_pct_gasto_alumno_secundaria'] = 100 * valor_en_año(auxiliares / (basicos + auxiliares), año)
    variables['gasto_alumno_secundaria_servicios_basicos_usd_ppp'] = valor_en_año(basicos, año)
    variables['gasto_alumno_secundaria_servicios_auxiliares_usd_ppp'] = valor_en_año(auxiliares, año)

    # --- Cómo: reparto por nivel y tipo de enseñanza (gasto por alumno, relativo a secundaria)
    def por_alumno(nivel):
        return valor_en_año(serie(gasto_por_alumno, nivel=nivel, **base), año)
    variables['gasto_alumno_preprimaria_usd_ppp'] = por_alumno('ISCED11_02').where(lambda valores: valores != 0)   # 0 = dato no disponible
    variables['ratio_gasto_preprimaria_sobre_secundaria'] = variables['gasto_alumno_preprimaria_usd_ppp'] / variables['gasto_alumno_secundaria_usd_ppp']
    variables['ratio_gasto_fp_sobre_general_secundaria_segunda_etapa'] = por_alumno('ISCED11_35') / por_alumno('ISCED11_34')

    # --- Cuándo: escolarización temprana y repetición
    escolarizacion = datos['escolarizacion']
    for edad, nombre in (('Y3', '3'), ('Y4', '4'), ('Y15', '15')):
        variables[f'escolarizacion_{nombre}_años_pct'] = valor_en_año(serie(escolarizacion, nivel='_T', edad=edad), año)
    variables['repeticion_pct_secundaria_primera_etapa'] = valor_en_año(serie(datos['repeticion'], nivel='ISCED11_24'), año)
    for nombre_columna in ('escolarizacion_3_años_pct', 'escolarizacion_4_años_pct', 'escolarizacion_15_años_pct'):
        variables[nombre_columna] = variables[nombre_columna].where(lambda valores: valores != 0)   # 0% de escolarización a esas edades = dato no disponible (p. ej. Canadá)

    # --- Renta y tamaño
    estadisticas_referencia = datos['referencia']
    pib_per_capita_moneda_local = serie(estadisticas_referencia, medida='GDP_CAPITA', unidad='XDC_PS', precios='V')
    deflactor, factor_ppp = serie(estadisticas_referencia, medida='GDP_DEFLATOR'), serie(estadisticas_referencia, medida='PPPGDP')
    variables['pib_per_capita_usd_ppp'] = valor_en_año(a_usd_ppp(pib_per_capita_moneda_local, deflactor, factor_ppp), año)
    variables['poblacion'] = 1000 * valor_en_año(serie(estadisticas_referencia, medida='POP'), año)   # la OCDE la da en miles
    variables['gasto_publico_total_pct_pib'] = valor_en_año(serie(estadisticas_referencia, medida='T_PUB_EXP', unidad='PT_B1GQ'), año)
    variables['matricula_secundaria_equivalente_tiempo_completo'] = valor_en_año(serie(datos['matricula'], nivel='ISCED11_2_3'), año)

    # --- Docentes: pago, dotación y perfil
    salarios_legales = datos['salarios_legales']
    for etiqueta, experiencia in (('inicial', 'EXP0'), ('15_años_experiencia', 'EXP15')):
        salario_xdc = serie(salarios_legales, nivel='ISCED11_24', unidad='XDC', PERS_EXP_LEV=experiencia)
        variables[f'salario_legal_docente_{etiqueta}_pct_pib_per_capita'] = 100 * valor_en_año(salario_xdc / pib_per_capita_moneda_local, año)
        variables[f'salario_legal_docente_{etiqueta}_usd_ppp'] = valor_en_año(a_usd_ppp(salario_xdc, deflactor, factor_ppp), año)
    variables['salario_real_docente_indice_base_2015'] = valor_en_año(serie(datos['salarios_reales'], nivel='ISCED11_24', unidad='IX'), año)
    variables['horas_docencia_anuales_secundaria_primera_etapa'] = valor_en_año(serie(datos['horas_docencia'], nivel='ISCED11_24'), año)
    variables['alumnos_por_docente_secundaria_primera_etapa'] = valor_en_año(serie(datos['alumnos_docente'], nivel='ISCED11_2', institucion='INST_EDU'), año)
    variables['tamaño_clase_secundaria_primera_etapa'] = valor_en_año(serie(datos['tamaño_clase'], nivel='ISCED11_2', institucion='INST_EDU'), año)
    edad = datos['docentes_edad']
    variables['docentes_50_o_mas_años_pct'] = valor_en_año(serie(edad, edad='Y_GE50'), año)
    variables['docentes_menos_de_30_años_pct'] = valor_en_año(serie(edad, edad='Y_LT30'), año)

    # --- Palancas del gasto por alumno: pago por docente x docentes por alumno
    variables['docentes_por_100_alumnos'] = 100 / variables['alumnos_por_docente_secundaria_primera_etapa']
    variables['coste_docente_por_alumno_pct_pib_per_capita'] = variables['salario_legal_docente_15_años_experiencia_pct_pib_per_capita'] / variables['alumnos_por_docente_secundaria_primera_etapa']
    variables['salario_legal_docente_15_años_pct_pib_per_capita_por_1000_horas'] = 1000 * variables['salario_legal_docente_15_años_experiencia_pct_pib_per_capita'] / variables['horas_docencia_anuales_secundaria_primera_etapa']

    # --- Desigualdad: Gini de la renta disponible
    gini_por_pais = gini_edicion(datos['gini_ocde'], datos['gini_banco_mundial'], año)
    variables['gini'], variables['gini_fuente'], variables['gini_año'] = gini_por_pais['gini'], gini_por_pais['gini_fuente'], gini_por_pais['gini_año']

    return pd.DataFrame(variables).reindex(PAISES_OCDE_TODAS_LAS_EDICIONES).rename_axis('pais').reset_index()


# Niveles de la serie anual de gasto por alumno
NIVELES_SERIE_ANUAL = {'preprimaria': 'ISCED11_02', 'primaria': 'ISCED11_1', 'secundaria_primera_etapa': 'ISCED11_2',
                       'secundaria_segunda_etapa': 'ISCED11_3', 'secundaria': 'ISCED11_2_3'}


def serie_anual_presupuesto(ruta_externos: str, inicio: int = 2000, fin: int = 2024) -> pd.DataFrame:
    """Gasto por alumno y PIB per cápita (USD PPP constantes de 2020) por país y año, para construir
    después medias o acumulados de años anteriores. Una fila por país y año; mismos nombres de
    columna que la tabla por edición. Sin interpolar: los huecos son NaN (en preprimaria, 0 = no disponible)."""
    leer_csv_externo = lambda nombre: cargar_oecd(os.path.join(ruta_externos, f'{nombre}.csv'))
    gasto_por_alumno, estadisticas_referencia = leer_csv_externo('gasto_por_alumno'), leer_csv_externo('estadisticas_referencia')
    base = dict(precios='Q', unidad='USD_PPP_ST', fuente='_T', institucion='INST_EDU', tipo_gasto='DIR_EXP')
    series = {}
    for nombre, nivel in NIVELES_SERIE_ANUAL.items():
        serie_nivel = serie(gasto_por_alumno, nivel=nivel, **base)
        series[f'gasto_alumno_{nombre}_usd_ppp'] = serie_nivel.where(serie_nivel != 0) if nombre == 'preprimaria' else serie_nivel
    series['pib_per_capita_usd_ppp'] = a_usd_ppp(serie(estadisticas_referencia, medida='GDP_CAPITA', unidad='XDC_PS', precios='V'),
                                         serie(estadisticas_referencia, medida='GDP_DEFLATOR'), serie(estadisticas_referencia, medida='PPPGDP'))
    serie_larga = pd.DataFrame({nombre_columna: serie_nivel.loc[inicio:fin].stack(future_stack=True) for nombre_columna, serie_nivel in series.items()})
    serie_larga = serie_larga.rename_axis(['año', 'pais']).reset_index()
    return serie_larga.sort_values(['pais', 'año']).reset_index(drop=True)[['pais', 'año'] + list(series)]


def construir_tabla(ruta_raw_base: str, ruta_externos: str, ediciones: list[int] | None = None) -> pd.DataFrame:
    """Tabla país-edición (una fila por país y edición): PISA + variables externas."""
    ediciones = ediciones or EDICIONES
    leer_csv_externo = lambda nombre: cargar_oecd(os.path.join(ruta_externos, f'{nombre}.csv'))
    pib_per_capita_banco_mundial = pd.read_csv(os.path.join(ruta_externos, 'pib_per_capita_ppp.csv')).dropna(subset=['valor'])
    datos = {
        'gasto': leer_csv_externo('gasto_por_alumno'), 'gasto_pib': leer_csv_externo('gasto_pib_y_fuentes'),
        'gasto_pct_publico': leer_csv_externo('gasto_pct_gasto_publico'), 'gasto_personal': leer_csv_externo('gasto_personal'),
        'gasto_corriente_capital': leer_csv_externo('gasto_corriente_capital'), 'gasto_tipo': leer_csv_externo('gasto_alumno_tipo'),
        'matricula': leer_csv_externo('matricula_equivalente_tc'), 'referencia': leer_csv_externo('estadisticas_referencia'),
        'salarios_reales': leer_csv_externo('salarios_reales_docentes'), 'salarios_legales': leer_csv_externo('salarios_legales_docentes'),
        'horas_docencia': leer_csv_externo('horas_docencia_legales'), 'alumnos_docente': leer_csv_externo('alumnos_por_docente'),
        'tamaño_clase': leer_csv_externo('tamaño_de_clase'), 'docentes_edad': leer_csv_externo('docentes_edad'), 'repeticion': leer_csv_externo('repeticion'),
        'escolarizacion': leer_csv_externo('escolarizacion_por_edad'),
        'gini_ocde': leer_csv_externo('gini_ocde'), 'gini_banco_mundial': pd.read_csv(os.path.join(ruta_externos, 'gini_banco_mundial.csv')),
    }
    pib_per_capita_serie = serie(pib_per_capita_banco_mundial)
    tablas = []
    for año in ediciones:
        tabla_edicion = resultados_pisa(ruta_raw_base, año).merge(centros_pisa(ruta_raw_base, año), on='pais', how='outer')
        tabla_edicion = tabla_edicion.merge(variables_externas(datos, año), on='pais', how='left')
        tabla_edicion['pib_per_capita_banco_mundial_usd_ppp_2021'] = tabla_edicion['pais'].map(valor_en_año(pib_per_capita_serie, año))
        tabla_edicion.insert(0, 'edicion', año)
        tablas.append(tabla_edicion)
    return pd.concat(tablas, ignore_index=True).sort_values(['pais', 'edicion']).reset_index(drop=True)


# ---------------------------------------------------------------- análisis

def residuo_ols(datos: pd.DataFrame, variable_objetivo: str, variables_explicativas: list[str]) -> tuple[pd.Series, pd.Series]:
    """Regresión lineal de `variable_objetivo` sobre `variables_explicativas` (con constante) en las filas completas.
    Devuelve (residuos indexados como `datos`, coeficientes). Un residuo positivo es un
    rendimiento superior al que predicen las `variables_explicativas`."""
    datos_completos = datos.dropna(subset=[variable_objetivo] + variables_explicativas)
    matriz_diseño = np.column_stack([np.ones(len(datos_completos))] + [datos_completos[variable].to_numpy(float) for variable in variables_explicativas])
    coeficientes, *_ = np.linalg.lstsq(matriz_diseño, datos_completos[variable_objetivo].to_numpy(float), rcond=None)
    residuo = pd.Series(datos_completos[variable_objetivo].to_numpy(float) - matriz_diseño @ coeficientes, index=datos_completos.index, name=f'{variable_objetivo}_residuo')
    return residuo, pd.Series(coeficientes, index=['const'] + variables_explicativas)


def agrupar_perfiles(datos: pd.DataFrame, columnas: list[str], numeros_de_grupos=range(2, 7), semilla: int = 42, min_filas_por_grupo: int = 5):
    """K-means sobre `columnas` estandarizadas, en las filas completas. Elige el nº de
    grupos con mayor silueta entre los que no dejan ningún grupo con menos de `min_filas_por_grupo`
    filas. Devuelve (etiquetas indexadas como `datos`, tabla de siluetas)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    datos_completos = datos.dropna(subset=columnas)
    datos_estandarizados = (datos_completos[columnas] - datos_completos[columnas].mean()) / datos_completos[columnas].std(ddof=0)
    siluetas, modelos = {}, {}
    for numero_grupos in numeros_de_grupos:
        modelo = KMeans(n_clusters=numero_grupos, n_init=20, random_state=semilla).fit(datos_estandarizados)
        siluetas[numero_grupos], modelos[numero_grupos] = silhouette_score(datos_estandarizados, modelo.labels_), modelo
    validos = [numero_grupos for numero_grupos in numeros_de_grupos if np.bincount(modelos[numero_grupos].labels_).min() >= min_filas_por_grupo] or list(numeros_de_grupos)
    mejor = max(validos, key=siluetas.get)
    etiquetas = pd.Series(modelos[mejor].labels_, index=datos_completos.index, name='perfil')
    return etiquetas, pd.Series(siluetas, name='silueta')


# ---------------------------------------------------------------- inventario de variables

# variable -> (series de origen, cálculo). Series: claves de `SERIES_OCDE` (= nombre del CSV en
# data/raw/externos), 'pib_per_capita_ppp' (Banco Mundial), 'gasto_publico_por_alumno_eurostat'
# o 'PISA' (tablas maestras: el cálculo indica la variable original).
DESCRIPCION_NIVEL_GASTO = {'1': 'primaria', '2': 'secundaria 1.ª etapa', '3': 'secundaria 2.ª etapa', '2_3': 'secundaria 1.ª+2.ª etapa'}
ORIGEN_VARIABLES = {
    **{f'gasto_alumno_{NOMBRE_NIVEL_GASTO[n]}_usd_ppp': (('gasto_por_alumno',), f'USD PPP constantes por alumno, {d}') for n, d in DESCRIPCION_NIVEL_GASTO.items()},
    'gasto_alumno_secundaria_media_4_años_usd_ppp': (('gasto_por_alumno',), 'Media de los 4 años hasta la edición (interpolación interior; mín. 3 años)'),
    'gasto_acumulado_alumno_6_a_15_años_usd_ppp': (('gasto_por_alumno',), 'Primaria t-9 a t-4 + secundaria 1.ª etapa t-3 a t (interpolación interior)'),
    'gasto_alumno_secundaria_pct_pib_per_capita': (('gasto_por_alumno',), 'Mismo gasto como % del PIB per cápita (`PT_B1GQ_POP`)'),
    'gasto_pct_pib_primaria_a_postsecundaria': (('gasto_pib_y_fuentes',), '% del PIB, primaria a postsecundaria no terciaria (`PT_B1GQ`)'),
    'financiacion_publica_pct_gasto_secundaria': (('gasto_pib_y_fuentes',), '% del gasto de secundaria 1.ª+2.ª que financia el gobierno (`S13`)'),
    'financiacion_privada_pct_gasto_secundaria': (('gasto_pib_y_fuentes',), '% financiado por el sector privado (`S1D_NON_EDU`)'),
    'financiacion_exterior_pct_gasto_secundaria': (('gasto_pib_y_fuentes',), '% financiado desde el exterior (`S2`)'),
    'pct_gasto_publico_en_primaria_a_postsecundaria': (('gasto_pct_gasto_publico',), '% del gasto público total, primaria a postsecundaria no terciaria'),
    'pct_gasto_secundaria_en_personal': (('gasto_personal', 'gasto_corriente_capital'), '100 × remuneración total / (corriente + capital)'),
    'pct_gasto_secundaria_en_docentes': (('gasto_personal', 'gasto_corriente_capital'), '100 × remuneración de docentes / (corriente + capital)'),
    'pct_gasto_secundaria_en_personal_no_docente': (('gasto_personal', 'gasto_corriente_capital'), '100 × remuneración de otro personal / (corriente + capital); 0 = NaN'),
    'pct_gasto_secundaria_en_capital': (('gasto_corriente_capital',), '100 × capital / (corriente + capital)'),
    'pct_gasto_secundaria_en_otros_corrientes': (('gasto_personal', 'gasto_corriente_capital'), '100 − personal − capital'),
    'servicios_auxiliares_pct_gasto_alumno_secundaria': (('gasto_alumno_tipo',), '100 × auxiliares / (básicos + auxiliares), secundaria 1.ª+2.ª'),
    'gasto_alumno_preprimaria_usd_ppp': (('gasto_por_alumno',), 'USD PPP por alumno, preprimaria (`ISCED11_02`); 0 = NaN'),
    'ratio_gasto_preprimaria_sobre_secundaria': (('gasto_por_alumno',), 'Gasto por alumno en preprimaria / secundaria 1.ª+2.ª'),
    'ratio_gasto_fp_sobre_general_secundaria_segunda_etapa': (('gasto_por_alumno',), 'Gasto por alumno en FP (`ISCED11_35`) / general (`ISCED11_34`), secundaria 2.ª etapa'),
    'gasto_publico_total_pct_pib': (('estadisticas_referencia',), 'Gasto público total como % del PIB (`T_PUB_EXP`)'),
    'matricula_secundaria_equivalente_tiempo_completo': (('matricula_equivalente_tc',), 'Matrícula en equivalente a tiempo completo, secundaria 1.ª+2.ª'),
    'salario_legal_docente_inicial_pct_pib_per_capita': (('salarios_legales_docentes', 'estadisticas_referencia'), '100 × salario legal inicial (moneda nacional) / PIB per cápita'),
    'salario_legal_docente_15_años_experiencia_pct_pib_per_capita': (('salarios_legales_docentes', 'estadisticas_referencia'), '100 × salario legal con 15 años de experiencia / PIB per cápita'),
    'salario_real_docente_indice_base_2015': (('salarios_reales_docentes',), 'Índice de salario real medio (base 2015 = 100)'),
    'alumnos_por_docente_secundaria_primera_etapa': (('alumnos_por_docente',), 'Alumnos por docente, secundaria 1.ª etapa (desde 2013)'),
    'docentes_por_100_alumnos': (('alumnos_por_docente',), '100 / alumnos por docente'),
    'coste_docente_por_alumno_pct_pib_per_capita': (('salarios_legales_docentes', 'estadisticas_referencia', 'alumnos_por_docente'),
                                             'Salario legal 15 años (% del PIB per cápita) / alumnos por docente'),
    'salario_legal_docente_15_años_pct_pib_per_capita_por_1000_horas': (('salarios_legales_docentes', 'estadisticas_referencia', 'horas_docencia_legales'),
                                    '1000 × salario legal 15 años (% del PIB per cápita) / horas legales de docencia'),
    'pib_per_capita_banco_mundial_usd_ppp_2021': (('pib_per_capita_ppp',), 'PIB per cápita PPP, USD constantes de 2021 (Banco Mundial; se usa para contrastar `pib_per_capita_usd_ppp`)'),
    'pib_per_capita_usd_ppp': (('estadisticas_referencia',), 'PIB per cápita (moneda nacional) deflactado a 2020 y convertido con la PPP de 2020'),
    'gasto_total_corriente_secundaria_usd_ppp': (('gasto_corriente_capital',), 'Gasto corriente total, secundaria 1.ª+2.ª, en USD PPP constantes (no millones); 0 = NaN'),
    'gasto_total_capital_secundaria_usd_ppp': (('gasto_corriente_capital',), 'Gasto de capital total, secundaria 1.ª+2.ª, en USD PPP constantes (no millones); 0 = NaN'),
    'gasto_total_personal_secundaria_usd_ppp': (('gasto_personal',), 'Remuneración de todo el personal, secundaria 1.ª+2.ª, en USD PPP constantes (no millones); 0 = NaN'),
    'gasto_total_docentes_secundaria_usd_ppp': (('gasto_personal',), 'Remuneración de docentes, secundaria 1.ª+2.ª, en USD PPP constantes (no millones); 0 = NaN'),
    'gasto_total_personal_no_docente_secundaria_usd_ppp': (('gasto_personal',), 'Remuneración de otro personal, secundaria 1.ª+2.ª, en USD PPP constantes (no millones); 0 = NaN'),
    'gasto_alumno_secundaria_servicios_basicos_usd_ppp': (('gasto_alumno_tipo',), 'USD PPP por alumno en servicios básicos de enseñanza (`CORE`), secundaria 1.ª+2.ª'),
    'gasto_alumno_secundaria_servicios_auxiliares_usd_ppp': (('gasto_alumno_tipo',), 'USD PPP por alumno en servicios auxiliares (`ASERV`: transporte, comidas, alojamiento)'),
    'salario_legal_docente_inicial_usd_ppp': (('salarios_legales_docentes', 'estadisticas_referencia'), 'Salario legal inicial (moneda nacional) deflactado a 2020 y convertido con la PPP de 2020'),
    'salario_legal_docente_15_años_experiencia_usd_ppp': (('salarios_legales_docentes', 'estadisticas_referencia'), 'Salario legal con 15 años de experiencia, igual conversión'),
    # contexto educativo, fuentes externas
    'gini': (('gini_ocde', 'gini_banco_mundial'), 'Gini de la renta disponible (0-1): observación más cercana a la edición (±1 año); OCDE antes que Banco Mundial (÷100)'),
    'gini_fuente': (('gini_ocde', 'gini_banco_mundial'), 'Fuente de la observación usada para `gini`'),
    'gini_año': (('gini_ocde', 'gini_banco_mundial'), 'Año de la observación usada para `gini` (edición, ±1)'),
    'poblacion': (('estadisticas_referencia',), 'Población en personas (`POP`; la OCDE la da en miles)'),
    'escolarizacion_3_años_pct': (('escolarizacion_por_edad',), '% de la población de 3 años matriculada; 0 = NaN'),
    'escolarizacion_4_años_pct': (('escolarizacion_por_edad',), '% de la población de 4 años matriculada; 0 = NaN'),
    'escolarizacion_15_años_pct': (('escolarizacion_por_edad',), '% de la población de 15 años matriculada; 0 = NaN'),
    'repeticion_pct_secundaria_primera_etapa': (('repeticion',), '% de repetidores en secundaria 1.ª etapa'),
    'docentes_50_o_mas_años_pct': (('docentes_edad',), '% del profesorado de secundaria 1.ª etapa con 50 o más años'),
    'docentes_menos_de_30_años_pct': (('docentes_edad',), '% del profesorado de secundaria 1.ª etapa con menos de 30 años'),
    'tamaño_clase_secundaria_primera_etapa': (('tamaño_de_clase',), 'Alumnos por clase en secundaria 1.ª etapa (desde 2013)'),
    'horas_docencia_anuales_secundaria_primera_etapa': (('horas_docencia_legales',), 'Horas anuales legales de docencia, secundaria 1.ª etapa (no hay 2022)'),
    # PISA (tablas maestras; media del país ponderada: W_FSTUWT en alumnos, W_SCHGRNRABWT en centros)
    'pisa_matematicas': (('PISA',), 'Media de `PV*MATH` por alumno, ponderada con `W_FSTUWT`'),
    'pisa_lectura': (('PISA',), 'Media de `PV*READ` por alumno, ponderada con `W_FSTUWT`'),
    'pisa_ciencias': (('PISA',), 'Media de `PV*SCIE` por alumno, ponderada con `W_FSTUWT`'),
    'pisa_numero_alumnos': (('PISA',), 'Nº de alumnos de la muestra'),
    'pisa_numero_centros': (('PISA',), 'Nº de centros de la muestra'),
    'pisa_escs_trend_media': (('PISA',), 'Media de `ESCS_TREND`'),
    'pisa_escs_trend_desviacion_tipica': (('PISA',), 'Desviación típica de `ESCS_TREND`'),
    'pisa_segregacion_social_entre_centros_pct': (('PISA',), '% de la varianza de `ESCS_TREND` entre centros'),
    'pisa_minutos_semanales_matematicas': (('PISA',), '`MMINS` (solo 2012-2018; 0 = NaN)'),
    'pisa_minutos_semanales_lengua': (('PISA',), '`LMINS` (solo 2012-2018; 0 = NaN)'),
    'pisa_minutos_semanales_ciencias': (('PISA',), '`SMINS` (solo 2012-2018; 0 = NaN)'),
    'pisa_alumnos_por_profesor': (('PISA',), '`STRATIO`, media de centros'),
    'pisa_tamaño_clase': (('PISA',), '`CLSIZE`, media de centros'),
    'pisa_tamaño_centro': (('PISA',), '`SCHSIZE`, media de centros'),
    'pisa_centros_publicos_pct': (('PISA',), '% de centros con `SCHLTYPE` = 3'),
    'pisa_financiacion_centro_gobierno_pct': (('PISA',), '`SC02Q01` (2012) / `SC016Q01TA`, media de centros'),
    'pisa_financiacion_centro_cuotas_pct': (('PISA',), '`SC02Q02` (2012) / `SC016Q02TA`, media de centros'),
    'pisa_financiacion_centro_donaciones_pct': (('PISA',), '`SC02Q03` (2012) / `SC016Q03TA`, media de centros'),
    'pisa_financiacion_centro_otras_pct': (('PISA',), '`SC02Q04` (2012) / `SC016Q04TA`, media de centros'),
    'pisa_profesores_certificados_pct': (('PISA',), '100 × `PROPCERT` / `PROATCE`, media de centros'),
    'pisa_profesores_master_pct': (('PISA',), '100 × `PROPAT7` (solo 2022)'),
    'pisa_ordenadores_por_alumno': (('PISA',), '`RATCMP1` / `RATCMP15`, media de centros'),
    'pisa_autonomia_recursos': (('PISA',), '`RESPRES` (2012, 2015) / `SRESPRES` (2022); no en 2018; no comparable'),
    'pisa_escasez_material_educativo': (('PISA',), '`EDUSHORT`, media de centros (desde 2015)'),
    'pisa_escasez_personal': (('PISA',), '`STAFFSHORT`, media de centros (desde 2015)'),
    **{f'pisa_brecha_{n}': (('PISA',), f'Media del cuartil de centros más desfavorecido menos la del más favorecido ({v})')
       for n, v in (('alumnos_por_profesor', '`STRATIO`'), ('tamaño_clase', '`CLSIZE`'), ('profesorado_certificado', 'profesorado certificado'),
                    ('escasez_material', '`EDUSHORT`'), ('escasez_personal', '`STAFFSHORT`'))},
}


def _proveedor(series: tuple) -> str:
    if series == ('PISA',):
        return 'PISA (tablas maestras)'
    if series[0] == 'gini_ocde':
        return 'OCDE y Banco Mundial'
    return {'pib_per_capita_ppp': 'Banco Mundial', 'gasto_publico_por_alumno_eurostat': 'Eurostat'}.get(series[0], 'OCDE')


def inventario_variables(tablas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Una fila por variable: tablas donde está, proveedor, series de origen (nombre del CSV en
    `data/raw/externos`), cálculo y cobertura (% de filas y nº de países con dato).
    `tablas`: {nombre: DataFrame} con `pais` y `edicion`."""
    filas, vistas = [], set()
    for nombre, tabla in tablas.items():
        for variable in tabla.columns:
            if variable in CLAVE or variable in vistas:
                continue
            vistas.add(variable)
            series, calculo = ORIGEN_VARIABLES[variable]
            donde = [nombre_tabla for nombre_tabla, otra_tabla in tablas.items() if variable in otra_tabla]
            filas.append({'variable': variable, 'tabla': ' + '.join(donde), 'proveedor': _proveedor(series),
                          'series (CSV)': ', '.join(series), 'cálculo': calculo,
                          '% filas con dato': round(100 * tabla[variable].notna().mean()),
                          'países con dato': tabla.loc[tabla[variable].notna(), 'pais'].nunique()})
    return pd.DataFrame(filas)
