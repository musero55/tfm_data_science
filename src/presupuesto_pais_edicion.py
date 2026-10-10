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

# Financiación del centro (% por origen): 2012 usa otros nombres
COLUMNAS_FINANCIACION = {
    'pisa_fin_gobierno_pct': ('SC02Q01', 'SC016Q01TA'),
    'pisa_fin_cuotas_pct': ('SC02Q02', 'SC016Q02TA'),
    'pisa_fin_donaciones_pct': ('SC02Q03', 'SC016Q03TA'),
    'pisa_fin_otras_pct': ('SC02Q04', 'SC016Q04TA'),
}

# Recursos de centro cuya distribución entre centros desfavorecidos y favorecidos se mide
RECURSOS_BRECHA = ['stratio', 'clsize', 'certificados', 'edushort', 'staffshort']


# ---------------------------------------------------------------- clasificación de columnas

CLAVE = ['pais', 'edicion']
# Resultados de PISA y control social (necesarios para cualquier análisis del gasto)
RESULTADOS = ['pisa_math', 'pisa_read', 'pisa_scie', 'pisa_n_alumnos', 'pisa_escs_trend_media']
# Variables NO económicas: contexto educativo (controles o mediadores posibles). Para mover una
# variable a la tabla económica basta con quitarla de esta lista.
CONTEXTO_EDUCATIVO = [
    'poblacion', 'pisa_n_centros', 'pisa_escs_trend_sd', 'pisa_segregacion_escs_pct',
    'pisa_min_mates', 'pisa_min_lengua', 'pisa_min_ciencias',
    'pisa_stratio', 'pisa_clsize', 'pisa_schsize', 'pisa_pct_centros_publicos',
    'pisa_profesores_certificados_pct', 'pisa_profesores_master_pct', 'pisa_ordenadores_por_alumno',
    'pisa_autonomia_recursos', 'pisa_edushort', 'pisa_staffshort',
    'pisa_brecha_stratio', 'pisa_brecha_clsize', 'pisa_brecha_certificados', 'pisa_brecha_edushort', 'pisa_brecha_staffshort',
    'escolarizacion_3_anios_pct', 'escolarizacion_4_anios_pct', 'escolarizacion_15_anios_pct',
    'repeticion_pct_sec1', 'docentes_50mas_pct', 'docentes_menos_30_pct', 'tamano_clase_sec1', 'horas_docencia_sec1',
]


def dividir_tabla(tabla: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa la tabla en (económica, contexto educativo). Ambas llevan `pais` y `edicion`;
    la económica incluye además los resultados de PISA y el `ESCS` medio. Todo lo que no
    esté en `CONTEXTO_EDUCATIVO` ni en `RESULTADOS` se considera económico."""
    desconocidas = [c for c in CONTEXTO_EDUCATIVO + RESULTADOS if c not in tabla.columns]
    assert not desconocidas, f'columnas clasificadas que no existen: {desconocidas}'
    economicas = [c for c in tabla.columns if c not in CLAVE + RESULTADOS + CONTEXTO_EDUCATIVO]
    return (tabla[CLAVE + RESULTADOS + economicas].copy(), tabla[CLAVE + CONTEXTO_EDUCATIVO].copy())


def media_ponderada(x: pd.Series, w: pd.Series) -> float:
    """Media de `x` ponderada por `w`, ignorando nulos. NaN si no hay datos."""
    m = x.notna() & w.notna()
    return float(np.average(x[m], weights=w[m])) if m.any() else np.nan


def desviacion_ponderada(x: pd.Series, w: pd.Series) -> float:
    m = x.notna() & w.notna()
    if not m.any():
        return np.nan
    media = np.average(x[m], weights=w[m])
    return float(np.sqrt(np.average((x[m] - media) ** 2, weights=w[m])))


def primera(nombres, *candidatas: str) -> str | None:
    """Primera de `candidatas` que exista en `nombres` (las ediciones nombran distinto)."""
    return next((c for c in candidatas if c in nombres), None)


# ---------------------------------------------------------------- PISA

def resultados_pisa(ruta_intermedios: str, año: int) -> pd.DataFrame:
    """Por país: media de cada dominio (PV promediados por alumno, ponderados por
    `W_FSTUWT`), nº de alumnos, media y desviación de `ESCS_TREND`, minutos semanales
    de aprendizaje por asignatura (2012-2018; los ceros se tratan como dato ausente)
    y segregación socioeconómica (% de la varianza de `ESCS_TREND` entre centros)."""
    ruta, _ = rutas_datos_anuales(ruta_intermedios, año)
    nombres = pq.ParquetFile(ruta).schema.names
    pv = [c for d in DOMINIOS for c in columnas_pv(d) if c in nombres]
    minutos = {'pisa_min_mates': 'MMINS', 'pisa_min_lengua': 'LMINS', 'pisa_min_ciencias': 'SMINS'}
    minutos = {k: v for k, v in minutos.items() if v in nombres}
    est = pd.read_parquet(ruta, columns=['CNT', 'CNTSCHID', 'W_FSTUWT', 'ESCS_TREND'] + pv + list(minutos.values()))
    filas = []
    for pais, g in est.groupby('CNT'):
        fila = {'pais': pais, 'pisa_n_alumnos': len(g)}
        for d in DOMINIOS:
            cols = [c for c in columnas_pv(d) if c in g.columns]
            fila[f'pisa_{d.lower()}'] = media_ponderada(g[cols].mean(axis=1), g['W_FSTUWT'])
        fila['pisa_escs_trend_media'] = media_ponderada(g['ESCS_TREND'], g['W_FSTUWT'])
        fila['pisa_escs_trend_sd'] = desviacion_ponderada(g['ESCS_TREND'], g['W_FSTUWT'])
        for nuevo, origen in minutos.items():
            fila[nuevo] = media_ponderada(g[origen].where(g[origen] > 0), g['W_FSTUWT'])
        fila['pisa_segregacion_escs_pct'] = segregacion(g)
        filas.append(fila)
    return pd.DataFrame(filas)


def segregacion(g: pd.DataFrame) -> float:
    """% de la varianza de `ESCS_TREND` que es entre centros (0 = todos los centros
    igual de mixtos; cuanto más alto, más separados por nivel socioeconómico)."""
    g = g.dropna(subset=['ESCS_TREND', 'CNTSCHID', 'W_FSTUWT'])
    if g.empty:
        return np.nan
    w, x = g['W_FSTUWT'], g['ESCS_TREND']
    media = np.average(x, weights=w)
    total = np.average((x - media) ** 2, weights=w)
    por_centro = g.assign(_wx=w * x).groupby('CNTSCHID').agg(_w=('W_FSTUWT', 'sum'), _wx=('_wx', 'sum'))
    por_centro['_m'] = por_centro['_wx'] / por_centro['_w']
    entre = np.average((por_centro['_m'] - media) ** 2, weights=por_centro['_w'])
    return float(100 * entre / total) if total > 0 else np.nan


def escs_de_centros(ruta_intermedios: str, año: int) -> pd.DataFrame:
    """Media de `ESCS_TREND` de los alumnos de cada centro (ponderada por `W_FSTUWT`)."""
    ruta, _ = rutas_datos_anuales(ruta_intermedios, año)
    est = pd.read_parquet(ruta, columns=['CNT', 'CNTSCHID', 'W_FSTUWT', 'ESCS_TREND']).dropna(subset=['ESCS_TREND'])
    est = est.assign(_wx=est['W_FSTUWT'] * est['ESCS_TREND'])
    g = est.groupby(['CNT', 'CNTSCHID']).agg(_w=('W_FSTUWT', 'sum'), _wx=('_wx', 'sum'))
    return (g['_wx'] / g['_w']).rename('escs_centro').reset_index()


def centros_pisa(ruta_intermedios: str, año: int) -> pd.DataFrame:
    """Por país, con el peso del centro (`W_SCHGRNRABWT`): medias de recursos y
    financiación del centro, % de centros públicos (`SCHLTYPE` = 3), autonomía sobre
    recursos y **brecha de recursos** entre centros desfavorecidos y favorecidos
    (cuartil inferior menos superior de `ESCS_TREND` medio del centro). Un valor
    negativo en ratio, tamaño de clase o escasez significa que los desfavorecidos
    están mejor dotados; positivo en profesorado certificado, al revés."""
    _, ruta = rutas_datos_anuales(ruta_intermedios, año)
    nombres = pq.ParquetFile(ruta).schema.names
    financiacion = {k: primera(nombres, *v) for k, v in COLUMNAS_FINANCIACION.items()}
    renombrar = {  # nombre canónico -> columna de cada edición
        'CERT': primera(nombres, 'PROPCERT', 'PROATCE'),
        'ORDENADORES': primera(nombres, 'RATCMP1', 'RATCMP15'),
        'AUTONOMIA': primera(nombres, 'RESPRES', 'SRESPRES'),
        'MASTER': primera(nombres, 'PROPAT7'),
    }
    cols = ['CNT', 'CNTSCHID', 'W_SCHGRNRABWT', 'STRATIO', 'CLSIZE', 'SCHSIZE', 'SCHLTYPE', 'EDUSHORT', 'STAFFSHORT']
    cols = [c for c in cols if c in nombres] + [c for c in {**financiacion, **renombrar}.values() if c]
    col = pd.read_parquet(ruta, columns=list(dict.fromkeys(cols)))
    col = col.rename(columns={v: k for k, v in renombrar.items() if v})
    for c in ('CERT', 'MASTER'):   # proporciones (0-1) -> %
        if c in col:
            col[c] = 100 * col[c]
    col = col.merge(escs_de_centros(ruta_intermedios, año), on=['CNT', 'CNTSCHID'], how='left')

    filas = []
    for pais, g in col.groupby('CNT'):
        w = g['W_SCHGRNRABWT']
        fila = {'pais': pais, 'pisa_n_centros': len(g),
                'pisa_stratio': media_ponderada(g['STRATIO'], w),
                'pisa_clsize': media_ponderada(g['CLSIZE'], w),
                'pisa_schsize': media_ponderada(g['SCHSIZE'], w),
                'pisa_pct_centros_publicos': 100 * media_ponderada((g['SCHLTYPE'] == 3).where(g['SCHLTYPE'].notna()).astype(float), w)}
        for nuevo, origen in financiacion.items():
            fila[nuevo] = media_ponderada(g[origen], w) if origen else np.nan
        for nuevo, origen in (('pisa_profesores_certificados_pct', 'CERT'), ('pisa_profesores_master_pct', 'MASTER'),
                              ('pisa_ordenadores_por_alumno', 'ORDENADORES'), ('pisa_autonomia_recursos', 'AUTONOMIA'),
                              ('pisa_edushort', 'EDUSHORT'), ('pisa_staffshort', 'STAFFSHORT')):
            fila[nuevo] = media_ponderada(g[origen], w) if origen in g else np.nan
        # Brecha de recursos entre el cuartil de centros más desfavorecido y el más favorecido
        q1, q3 = g['escs_centro'].quantile([0.25, 0.75])
        desfav, fav = g[g['escs_centro'] <= q1], g[g['escs_centro'] >= q3]
        for nombre, origen in (('stratio', 'STRATIO'), ('clsize', 'CLSIZE'), ('certificados', 'CERT'),
                               ('edushort', 'EDUSHORT'), ('staffshort', 'STAFFSHORT')):
            if origen in g:
                fila[f'pisa_brecha_{nombre}'] = (media_ponderada(desfav[origen], desfav['W_SCHGRNRABWT'])
                                                 - media_ponderada(fav[origen], fav['W_SCHGRNRABWT']))
            else:
                fila[f'pisa_brecha_{nombre}'] = np.nan
        filas.append(fila)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- series externas

def serie(df: pd.DataFrame, **filtros) -> pd.DataFrame:
    """Serie anual (filas: años; columnas: países) con los filtros dados. Falla si
    quedan duplicados país-año, señal de que falta un filtro."""
    for k, v in filtros.items():
        df = df[df[k] == v]
    assert not df.duplicated(['pais', 'año']).any(), f'duplicados con {filtros}'
    return df.pivot(index='año', columns='pais', values='valor').reindex(AÑOS_SERIE)


def en(serie_anual: pd.DataFrame, año: int) -> pd.Series:
    """Valor de cada país en un año (NaN si no hay)."""
    return serie_anual.loc[año]


def interpolada(serie_anual: pd.DataFrame) -> pd.DataFrame:
    """Rellena linealmente los huecos *interiores* de cada país (nunca extrapola)."""
    return serie_anual.interpolate(limit_area='inside')


def media_previa(serie_anual: pd.DataFrame, año: int, ventana: int = 4, minimo: int = 3) -> pd.Series:
    """Media de los `ventana` años hasta `año` (incluido), sobre la serie interpolada.
    NaN si hay menos de `minimo` años con dato."""
    w = interpolada(serie_anual).loc[año - ventana + 1:año]
    return w.mean().where(w.notna().sum() >= minimo)


def acumulado_6_15(primaria: pd.DataFrame, secundaria1: pd.DataFrame, año: int) -> pd.Series:
    """Gasto por alumno acumulado entre los 6 y los 15 años de la cohorte que hace PISA
    en `año`: 6 años de primaria (t-9 a t-4) y 4 de secundaria 1.ª etapa (t-3 a t),
    sobre series interpoladas. Aproximado (no usa la duración teórica de cada país).
    NaN si falta algún año."""
    p = interpolada(primaria).loc[año - 9:año - 4]
    s = interpolada(secundaria1).loc[año - 3:año]
    completo = (p.notna().sum() == 6) & (s.notna().sum() == 4)
    return (p.sum() + s.sum()).where(completo)


def variables_externas(datos: dict, año: int) -> pd.DataFrame:
    """Variables de presupuesto, renta y personal de cada país en la edición `año`.
    `datos`: dict con los DataFrames cargados con `cargar_oecd` (ver `construir_tabla`)."""
    g = datos['gasto']
    base = dict(precios='Q', unidad='USD_PPP_ST', fuente='_T', institucion='INST_EDU', tipo_gasto='DIR_EXP')
    por_nivel = {n: serie(g, nivel=f'ISCED11_{n}', **base) for n in ('1', '2', '3', '2_3')}

    v = {}
    # --- Cuánto: gasto por alumno (USD PPP, precios constantes), en el año de la prueba
    for n, s in por_nivel.items():
        v[f'gasto_alumno_{n}'] = en(s, año)
    v['gasto_alumno_2_3_media_4a'] = media_previa(por_nivel['2_3'], año)
    v['gasto_acumulado_6_15'] = acumulado_6_15(por_nivel['1'], por_nivel['2'], año)
    v['gasto_alumno_2_3_pct_pib_pc'] = en(serie(
        g, nivel='ISCED11_2_3', unidad='PT_B1GQ_POP', fuente='_T', institucion='INST_EDU', tipo_gasto='DIR_EXP'), año)

    # --- Esfuerzo y origen de la financiación
    f = datos['gasto_pib']
    v['gasto_pib_pct_1T4'] = en(serie(f, nivel='ISCED11_1T4', unidad='PT_B1GQ', fuente='_T', precios='_Z'), año)
    for nombre, fuente in (('publica', 'S13'), ('privada', 'S1D_NON_EDU'), ('exterior', 'S2')):
        v[f'financiacion_{nombre}_pct'] = en(serie(f, nivel='ISCED11_2_3', unidad='PT_EXP', fuente=fuente, precios='_Z'), año)
    v['gasto_pct_gasto_publico_1T4'] = en(serie(datos['gasto_pct_publico'], nivel='ISCED11_1T4'), año)

    # --- Cómo: composición del gasto total de sec. 1.ª+2.ª (corriente + capital)
    comun = dict(nivel='ISCED11_2_3', unidad='USD_PPP', precios='Q', fuente='_T', institucion='INST_EDU')
    corriente = serie(datos['gasto_corriente_capital'], tipo_gasto='CUR', **comun)
    capital = serie(datos['gasto_corriente_capital'], tipo_gasto='CAP', **comun)
    total = corriente + capital
    personal = serie(datos['gasto_personal'], tipo_gasto='CUR_COMP', **comun)
    docentes = serie(datos['gasto_personal'], tipo_gasto='CUR_COMPT', **comun)
    no_docentes = serie(datos['gasto_personal'], tipo_gasto='CUR_COMPO', **comun)
    v['gasto_personal_pct'] = 100 * en(personal / total, año)
    v['gasto_docentes_pct'] = 100 * en(docentes / total, año)
    v['gasto_nodocentes_pct'] = (100 * en(no_docentes / total, año)).where(lambda x: x != 0)   # 0 = incluido en docentes
    v['gasto_capital_pct'] = 100 * en(capital / total, año)
    v['gasto_otros_corrientes_pct'] = 100 - v['gasto_personal_pct'] - v['gasto_capital_pct']

    # Servicios básicos frente a auxiliares (transporte, comidas, alojamiento)
    tipo = datos['gasto_tipo']
    base_t = dict(nivel='ISCED11_2_3', unidad='USD_PPP_ST')
    basicos, auxiliares = serie(tipo, tipo_gasto='CORE', **base_t), serie(tipo, tipo_gasto='ASERV', **base_t)
    v['servicios_auxiliares_pct_2_3'] = 100 * en(auxiliares / (basicos + auxiliares), año)

    # --- Cómo: reparto por nivel y tipo de enseñanza (gasto por alumno, relativo a secundaria)
    def por_alumno(nivel):
        return en(serie(g, nivel=nivel, **base), año)
    v['gasto_alumno_preprimaria'] = por_alumno('ISCED11_02').where(lambda x: x != 0)   # 0 = dato no disponible
    v['gasto_alumno_terciaria'] = por_alumno('ISCED11_5T8')
    v['ratio_preprimaria_sec'] = v['gasto_alumno_preprimaria'] / v['gasto_alumno_2_3']
    v['ratio_terciaria_sec'] = v['gasto_alumno_terciaria'] / v['gasto_alumno_2_3']
    v['ratio_fp_general_sec2'] = por_alumno('ISCED11_35') / por_alumno('ISCED11_34')

    # --- Cuándo: escolarización temprana y repetición
    esc = datos['escolarizacion']
    for edad, nombre in (('Y3', '3'), ('Y4', '4'), ('Y15', '15')):
        v[f'escolarizacion_{nombre}_anios_pct'] = en(serie(esc, nivel='_T', edad=edad), año)
    v['repeticion_pct_sec1'] = en(serie(datos['repeticion'], nivel='ISCED11_24'), año)
    for k in ('escolarizacion_3_anios_pct', 'escolarizacion_4_anios_pct', 'escolarizacion_15_anios_pct'):
        v[k] = v[k].where(lambda x: x != 0)   # 0% de escolarización a esas edades = dato no disponible (p. ej. Canadá)

    # --- Renta y tamaño
    ref = datos['referencia']
    pib_pc_xdc = serie(ref, medida='GDP_CAPITA', unidad='XDC_PS', precios='V')
    v['poblacion'] = en(serie(ref, medida='POP'), año)
    v['gasto_publico_total_pct_pib'] = en(serie(ref, medida='T_PUB_EXP', unidad='PT_B1GQ'), año)
    v['matricula_fte_2_3'] = en(serie(datos['matricula'], nivel='ISCED11_2_3'), año)

    # --- Docentes: pago, dotación y perfil
    sal = datos['salarios_legales']
    for etiqueta, exp in (('inicial', 'EXP0'), ('15a', 'EXP15')):
        v[f'salario_legal_{etiqueta}_pct_pib_pc'] = 100 * en(
            serie(sal, nivel='ISCED11_24', unidad='XDC', PERS_EXP_LEV=exp) / pib_pc_xdc, año)
    v['salario_real_indice_2015'] = en(serie(datos['salarios_reales'], nivel='ISCED11_24', unidad='IX'), año)
    v['horas_docencia_sec1'] = en(serie(datos['horas_docencia'], nivel='ISCED11_24'), año)
    v['alumnos_por_docente_sec1'] = en(serie(datos['alumnos_docente'], nivel='ISCED11_2', institucion='INST_EDU'), año)
    v['tamano_clase_sec1'] = en(serie(datos['tamano_clase'], nivel='ISCED11_2', institucion='INST_EDU'), año)
    edad = datos['docentes_edad']
    v['docentes_50mas_pct'] = en(serie(edad, edad='Y_GE50'), año)
    v['docentes_menos_30_pct'] = en(serie(edad, edad='Y_LT30'), año)

    # --- Palancas del gasto por alumno: pago por docente x docentes por alumno
    v['docentes_por_100_alumnos'] = 100 / v['alumnos_por_docente_sec1']
    v['coste_docentes_por_alumno_pct_pib_pc'] = v['salario_legal_15a_pct_pib_pc'] / v['alumnos_por_docente_sec1']
    v['salario_legal_15a_por_1000h'] = 1000 * v['salario_legal_15a_pct_pib_pc'] / v['horas_docencia_sec1']

    # --- Eurostat (solo UE, gasto público, PPS): proxy para rellenar, no se mezcla con el de la OCDE
    eu = datos['eurostat']
    for n, nivel in (('sec1', 'ED2'), ('sec2', 'ED3')):
        v[f'eurostat_gasto_publico_alumno_{n}_pps'] = en(serie(eu, nivel=nivel, unidad='PPS'), año)

    return pd.DataFrame(v).reindex(PAISES_OCDE_TODAS_LAS_EDICIONES).rename_axis('pais').reset_index()


def construir_tabla(ruta_intermedios: str, ruta_externos: str, ediciones: list[int] | None = None) -> pd.DataFrame:
    """Tabla país-edición (una fila por país y edición): PISA + variables externas."""
    ediciones = ediciones or EDICIONES
    c = lambda nombre: cargar_oecd(os.path.join(ruta_externos, f'{nombre}.csv'))
    pib = pd.read_csv(os.path.join(ruta_externos, 'pib_per_capita_ppp.csv')).dropna(subset=['valor'])
    datos = {
        'gasto': c('gasto_por_alumno'), 'gasto_pib': c('gasto_pib_y_fuentes'),
        'gasto_pct_publico': c('gasto_pct_gasto_publico'), 'gasto_personal': c('gasto_personal'),
        'gasto_corriente_capital': c('gasto_corriente_capital'), 'gasto_tipo': c('gasto_alumno_tipo'),
        'matricula': c('matricula_equivalente_tc'), 'referencia': c('estadisticas_referencia'),
        'salarios_reales': c('salarios_reales_docentes'), 'salarios_legales': c('salarios_legales_docentes'),
        'horas_docencia': c('horas_docencia_legales'), 'alumnos_docente': c('alumnos_por_docente'),
        'tamano_clase': c('tamano_de_clase'), 'docentes_edad': c('docentes_edad'), 'repeticion': c('repeticion'),
        'escolarizacion': c('escolarizacion_por_edad'),
        'eurostat': pd.read_csv(os.path.join(ruta_externos, 'gasto_publico_por_alumno_eurostat.csv')),
    }
    pib_pc = serie(pib)
    tablas = []
    for año in ediciones:
        t = resultados_pisa(ruta_intermedios, año).merge(centros_pisa(ruta_intermedios, año), on='pais', how='outer')
        t = t.merge(variables_externas(datos, año), on='pais', how='left')
        t['pib_pc_ppp_const2021'] = t['pais'].map(en(pib_pc, año))
        t.insert(0, 'edicion', año)
        tablas.append(t)
    return pd.concat(tablas, ignore_index=True).sort_values(['pais', 'edicion']).reset_index(drop=True)


# ---------------------------------------------------------------- análisis

def residuo_ols(df: pd.DataFrame, y: str, xs: list[str]) -> tuple[pd.Series, pd.Series]:
    """Regresión lineal de `y` sobre `xs` (con constante) en las filas completas.
    Devuelve (residuos indexados como `df`, coeficientes). Un residuo positivo es un
    rendimiento superior al que predicen las `xs`."""
    d = df.dropna(subset=[y] + xs)
    X = np.column_stack([np.ones(len(d))] + [d[x].to_numpy(float) for x in xs])
    beta, *_ = np.linalg.lstsq(X, d[y].to_numpy(float), rcond=None)
    residuo = pd.Series(d[y].to_numpy(float) - X @ beta, index=d.index, name=f'{y}_residuo')
    return residuo, pd.Series(beta, index=['const'] + xs)


def agrupar_perfiles(df: pd.DataFrame, columnas: list[str], ks=range(2, 7), semilla: int = 42, min_tam: int = 5):
    """K-means sobre `columnas` estandarizadas, en las filas completas. Elige el nº de
    grupos con mayor silueta entre los que no dejan ningún grupo con menos de `min_tam`
    filas. Devuelve (etiquetas indexadas como `df`, tabla de siluetas)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    d = df.dropna(subset=columnas)
    z = (d[columnas] - d[columnas].mean()) / d[columnas].std(ddof=0)
    siluetas, modelos = {}, {}
    for k in ks:
        m = KMeans(n_clusters=k, n_init=20, random_state=semilla).fit(z)
        siluetas[k], modelos[k] = silhouette_score(z, m.labels_), m
    validos = [k for k in ks if np.bincount(modelos[k].labels_).min() >= min_tam] or list(ks)
    mejor = max(validos, key=siluetas.get)
    etiquetas = pd.Series(modelos[mejor].labels_, index=d.index, name='perfil')
    return etiquetas, pd.Series(siluetas, name='silueta')
