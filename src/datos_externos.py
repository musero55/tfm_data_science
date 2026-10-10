"""
datos_externos.py
=================

Descarga de datos externos (OCDE y Banco Mundial) para el análisis de presupuesto.
Cada serie se guarda como CSV en `data/raw/externos/`. Las fuentes están descritas
al principio del notebook `01_EDA_presupuestos.ipynb`.

Las APIs son públicas y no necesitan clave. La de la OCDE (SDMX) tiene límite de
consultas por hora, así que `descargar_todo` salta los ficheros que ya existen.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pandas as pd

from src.pisa_pipeline import PAISES_OCDE_TODAS_LAS_EDICIONES

URL_OCDE = 'https://sdmx.oecd.org/public/rest/data'
AGENCIA_EDU = 'OECD.EDU.IMEP'
URL_BANCO_MUNDIAL = 'https://api.worldbank.org/v2'

# Series de la OCDE. `dims`: filtro por dimensión en el orden del DSD, sin la primera
# (país); None = todos los valores. El orden de dimensiones está en el DSD de cada
# dataflow (`sdmx.oecd.org/public/rest/datastructure/OECD.EDU.IMEP/<DSD>/latest`).
SERIES_OCDE = {
    'gasto_por_alumno': {
        'dataflow': 'DSD_EAG_UOE_FIN@DF_UOE_INDIC_FIN_PERSTUD', 'version': '3.2',
        # MEASURE, EDUCATION_LEV, EXP_SOURCE, EXP_DESTINATION, EXPENDITURE_TYPE, PRICE_BASE, UNIT_MEASURE, Q_SHEET
        'dims': ['FIN_PERSTUD', None, '_T', 'INST_EDU', 'DIR_EXP', None, None, None],
    },
    # Salarios docentes desde 2000 (los dataflows no "TREND" solo traen 2023-2025)
    'salarios_reales_docentes': {
        'dataflow': 'DSD_EAG_SAL_TREND@DF_TCH_ACT', 'version': '2.1',
        # MEASURE, UNIT_MEASURE, INST_TYPE_EDU, EDUCATION_LEV, AGE, SEX, PERS_TYPE, PERS_QUAL_LEV, PERS_EXP_LEV, PRICE_BASE
        'dims': [None] * 10,
    },
    'salarios_legales_docentes': {
        'dataflow': 'DSD_EAG_SAL_TREND@DF_TCH_STA', 'version': '2.1',
        'dims': [None] * 10,
    },
    # Dataflows de DSD_EAG_UOE_FIN (mismo orden de dimensiones que gasto_por_alumno).
    # Niveles: 1 primaria, 2 sec. 1.ª etapa, 3 sec. 2.ª etapa, 2_3 ambas, 1T4 primaria a postsecundaria no terciaria
    'gasto_pib_y_fuentes': {
        'dataflow': 'DSD_EAG_UOE_FIN@DF_UOE_FIN_SOURCE_GV_PR_NDOM', 'version': '3.2',
        # fuente de financiación (_T total, S13 gobierno, S1D_NON_EDU privado, S2 exterior); unidades: % PIB, % del gasto, USD PPP, moneda nacional
        'dims': ['EXP', 'ISCED11_1+ISCED11_2+ISCED11_3+ISCED11_2_3+ISCED11_1T4', None, 'INST_EDU', 'DIR_EXP', None, None, None],
    },
    'gasto_pct_gasto_publico': {
        'dataflow': 'DSD_EAG_UOE_FIN@DF_UOE_FIN_INDIC_SHARE_EDU_GOV', 'version': '3.2',
        'dims': ['EXP', 'ISCED11_1+ISCED11_2+ISCED11_3+ISCED11_2_3+ISCED11_1T4+ISCED11_1T8', None, None, None, None, None, None],
    },
    'gasto_personal': {
        'dataflow': 'DSD_EAG_UOE_FIN@DF_UOE_FIN_NATURE_STAFF', 'version': '3.2',
        # solo precios constantes y USD PPP; todos los tipos de remuneración
        'dims': ['EXP', 'ISCED11_1+ISCED11_2+ISCED11_3+ISCED11_2_3', '_T', 'INST_EDU', None, 'Q', 'USD_PPP', None],
    },
    'gasto_corriente_capital': {
        'dataflow': 'DSD_EAG_UOE_FIN@DF_UOE_FIN_NATURE_CUR_CAP', 'version': '3.2',
        'dims': ['EXP', 'ISCED11_1+ISCED11_2+ISCED11_3+ISCED11_2_3', '_T', 'INST_EDU', None, 'Q', 'USD_PPP', None],
    },
    'matricula_equivalente_tc': {
        'dataflow': 'DSD_EAG_UOE_FIN_ENR@DF_UOE_FIN_ENR', 'version': '3.2',
        # MEASURE, EDUCATION_LEV, INTENSITY (FTE = equivalente a tiempo completo), INST_TYPE_EDU, UNIT_MEASURE, Q_SHEET
        'dims': ['ENR', 'ISCED11_1+ISCED11_2+ISCED11_3+ISCED11_2_3', 'FTE', 'INST_EDU', 'PS', None],
    },
    'estadisticas_referencia': {
        'dataflow': 'DSD_EAG_UOE_FIN_ANNEX@DF_UOE_FIN_ANNEX', 'version': '3.2',
        'dims': [None] * 4,   # todas (PIB, PIB per cápita PPP, factor PPP, deflactor, población, gasto público total)
    },
    'horas_docencia_legales': {
        'dataflow': 'DSD_EAG_WT_TREND@DF_ALL', 'version': '2.0',
        'dims': [None] * 5,
    },
    # Cómo se gasta: tipo de gasto (CORE = servicios básicos de enseñanza, ASERV = servicios auxiliares)
    'gasto_alumno_tipo': {
        'dataflow': 'DSD_EAG_UOE_FIN@DF_UOE_INDIC_FIN_PERSTUD', 'version': '3.2',
        'dims': ['FIN_PERSTUD', 'ISCED11_1+ISCED11_2+ISCED11_3+ISCED11_2_3', '_T', 'INST_EDU', 'CORE+ASERV', 'Q', 'USD_PPP_ST', None],
    },
    # Dataflows de DSD_EAG_UOE_NON_FIN_PERS / _STUD: tras el país vienen EDUCATION_LEV, MEASURE, EDUCATION_TYPE,
    # INTENSITY, EDUCATION_FIELD, GRADE, FREQ, ORIGIN, DESTINATION, INST_TYPE_EDU, MOBILITY, UNIT_MEASURE, SEX, AGE
    'docentes_edad': {   # % de profesorado de sec. 1.ª etapa con 50 o más años / menos de 30
        'dataflow': 'DSD_EAG_UOE_NON_FIN_PERS@DF_UOE_NF_PERS_AGE', 'version': '1.1',
        'dims': ['ISCED11_2', None, None, None, None, None, None, None, None, 'INST_EDU', None, None, '_T', 'Y_GE50+Y_LT30'],
    },
    'repeticion': {      # % de repetidores
        'dataflow': 'DSD_EAG_UOE_NON_FIN_STUD@DF_UOE_NF_DIST_RPTR', 'version': '1.1',
        'dims': ['ISCED11_1+ISCED11_24+ISCED11_34', None, None, None, None, None, None, None, None, 'INST_EDU', None, None, '_T', None],
    },
    'escolarizacion_por_edad': {   # % de la población de cada edad matriculada, por nivel (edades 3, 4, 5 y 15)
        'dataflow': 'DSD_EAG_UOE_NON_FIN_STUD@DF_UOE_NF_ENRL_RATE', 'version': '1.1',
        'dims': [None, None, None, None, None, None, None, None, None, 'INST_EDU', None, None, '_T', 'Y3+Y4+Y5+Y15'],
    },
    'alumnos_por_docente': {
        'dataflow': 'DSD_EAG_UOE_NON_FIN_PERS@DF_UOE_NF_PERS_STR', 'version': '1.1',
        'dims': [None] * 14,
    },
    'tamaño_de_clase': {
        'dataflow': 'DSD_EAG_UOE_NON_FIN_PERS@DF_UOE_NF_PERS_CLS', 'version': '1.1',
        'dims': [None] * 14,
    },
}

# Eurostat: gasto público por alumno (EUR, PPS, % del PIB per cápita), solo países de la UE.
# Sirve para rellenar el gasto de Bélgica y Países Bajos en 2012, que la OCDE no publica.
EUROSTAT_GASTO_ALUMNO = 'educ_uoe_fine09'
GEO_EUROSTAT = {'AUT': 'AT', 'BEL': 'BE', 'CZE': 'CZ', 'DEU': 'DE', 'DNK': 'DK', 'ESP': 'ES', 'EST': 'EE',
                'FIN': 'FI', 'FRA': 'FR', 'GRC': 'EL', 'HUN': 'HU', 'IRL': 'IE', 'ITA': 'IT', 'NLD': 'NL',
                'POL': 'PL', 'PRT': 'PT', 'SVK': 'SK', 'SVN': 'SI', 'SWE': 'SE'}

PIB_BANCO_MUNDIAL = 'NY.GDP.PCAP.PP.KD'   # PIB per cápita PPP, USD internacionales constantes de 2021


def _get(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _n_dimensiones(dataflow: str) -> int:
    """Nº de dimensiones del DSD de un dataflow (consulta el catálogo de la OCDE)."""
    import xml.etree.ElementTree as ET
    dsd = dataflow.split('@')[0]
    xml = _get(f'https://sdmx.oecd.org/public/rest/datastructure/{AGENCIA_EDU}/{dsd}/latest', timeout=60)
    ns = '{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure}'
    return len(next(ET.fromstring(xml).iter(ns + 'DimensionList')).findall(ns + 'Dimension'))


def url_oecd(serie: dict, paises: list[str], inicio: int = 2000) -> str:
    """URL de consulta CSV de una serie de `SERIES_OCDE` para los países indicados."""
    dims = serie['dims'] if serie['dims'] is not None else [None] * (_n_dimensiones(serie['dataflow']) - 1)
    clave = '.'.join(['+'.join(paises)] + [d or '' for d in dims])
    return (f"{URL_OCDE}/{AGENCIA_EDU},{serie['dataflow']},{serie['version']}/{clave}"
            f"?startPeriod={inicio}&format=csvfilewithlabels")


def descargar_oecd(nombre: str, ruta_carpeta: str, paises: list[str] | None = None, inicio: int = 2000) -> str:
    """Descarga una serie de `SERIES_OCDE` a `<ruta_carpeta>/<nombre>.csv`."""
    paises = paises or PAISES_OCDE_TODAS_LAS_EDICIONES
    ruta = os.path.join(ruta_carpeta, f'{nombre}.csv')
    datos = _get(url_oecd(SERIES_OCDE[nombre], paises, inicio))   # antes de abrir el fichero: si falla, no deja uno vacío
    with open(ruta, 'wb') as f:
        f.write(datos)
    return ruta


def descargar_pib(ruta_carpeta: str, paises: list[str] | None = None, inicio: int = 2000, fin: int = 2023) -> str:
    """PIB per cápita PPP (Banco Mundial) a `<ruta_carpeta>/pib_per_capita_ppp.csv`
    (columnas: pais, año, valor)."""
    paises = paises or PAISES_OCDE_TODAS_LAS_EDICIONES
    url = (f"{URL_BANCO_MUNDIAL}/country/{';'.join(paises)}/indicator/{PIB_BANCO_MUNDIAL}"
           f"?format=json&date={inicio}:{fin}&per_page=20000")
    _, filas = json.loads(_get(url))
    df = pd.DataFrame(
        [{'pais': f['countryiso3code'], 'año': int(f['date']), 'valor': f['value']} for f in filas]
    ).sort_values(['pais', 'año'])
    ruta = os.path.join(ruta_carpeta, 'pib_per_capita_ppp.csv')
    df.to_csv(ruta, index=False)
    return ruta


def descargar_eurostat(ruta_carpeta: str, inicio: int = 2010) -> str:
    """Gasto público por alumno de Eurostat (`educ_uoe_fine09`) a
    `<ruta_carpeta>/gasto_publico_por_alumno_eurostat.csv`
    (columnas: pais, año, nivel, unidad, valor). Solo países de la UE."""
    geos = ''.join(f'&geo={g}' for g in GEO_EUROSTAT.values())
    años = ''.join(f'&time={a}' for a in range(inicio, 2024))
    url = (f'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{EUROSTAT_GASTO_ALUMNO}'
           f'?format=JSON&lang=EN{geos}{años}')
    j = json.loads(_get(url))
    ids, tam = j['id'], j['size']
    cat = {d: {p: c for c, p in j['dimension'][d]['category']['index'].items()} for d in ids}   # posicion -> codigo
    inverso_geo = {v: k for k, v in GEO_EUROSTAT.items()}
    filas = []
    for pos, valor in j['value'].items():
        pos = int(pos)
        cod = {}
        for d, n in zip(reversed(ids), reversed(tam)):   # índice plano: la última dimensión varía más rápido
            cod[d] = cat[d][pos % n]
            pos //= n
        filas.append({'pais': inverso_geo[cod['geo']], 'año': int(cod['time']), 'nivel': cod['isced11'],
                      'unidad': cod['unit'], 'valor': valor})
    df = pd.DataFrame(filas).sort_values(['pais', 'año', 'nivel', 'unidad'])
    ruta = os.path.join(ruta_carpeta, 'gasto_publico_por_alumno_eurostat.csv')
    df.to_csv(ruta, index=False)
    return ruta


def descargar_todo(ruta_carpeta: str, forzar: bool = False) -> dict[str, str]:
    """Descarga las series de la OCDE, el PIB del Banco Mundial y el gasto de Eurostat. Salta las que ya existen salvo `forzar=True`."""
    os.makedirs(ruta_carpeta, exist_ok=True)
    rutas = {}
    for nombre in SERIES_OCDE:
        ruta = os.path.join(ruta_carpeta, f'{nombre}.csv')
        if forzar or not os.path.exists(ruta):
            print(f'Descargando {nombre}...')
            descargar_oecd(nombre, ruta_carpeta)
        rutas[nombre] = ruta
    ruta = os.path.join(ruta_carpeta, 'pib_per_capita_ppp.csv')
    if forzar or not os.path.exists(ruta):
        print('Descargando pib_per_capita_ppp...')
        descargar_pib(ruta_carpeta)
    rutas['pib_per_capita_ppp'] = ruta
    ruta = os.path.join(ruta_carpeta, 'gasto_publico_por_alumno_eurostat.csv')
    if forzar or not os.path.exists(ruta):
        print('Descargando gasto_publico_por_alumno_eurostat...')
        descargar_eurostat(ruta_carpeta)
    rutas['gasto_publico_por_alumno_eurostat'] = ruta
    return rutas


# ---------------------------------------------------------------- enlaces y comprobación

URL_EXPLORADOR = 'https://data-explorer.oecd.org/vis'
PORTAL_BM = f'https://data.worldbank.org/indicator/{PIB_BANCO_MUNDIAL}'
PORTAL_EUROSTAT = f'https://ec.europa.eu/eurostat/databrowser/view/{EUROSTAT_GASTO_ALUMNO}/default/table?lang=en'


def url_explorador(dataflow: str, version: str) -> str:
    """Página del dataflow en el OECD Data Explorer (donde se ven y descargan los datos)."""
    return (f"{URL_EXPLORADOR}?df%5Bds%5D=DisseminateFinalDMZ&df%5Bid%5D={dataflow.replace('@', '%40')}"
            f"&df%5Bag%5D={AGENCIA_EDU}&df%5Bvs%5D={version}")


def enlaces_fuentes() -> pd.DataFrame:
    """Una fila por fichero descargado: proveedor, portal donde se ven los datos y consulta
    de API de muestra (España, último dato) que devuelve el mismo formato que la descarga."""
    filas = [{'serie': n, 'fichero': f'{n}.csv', 'proveedor': 'OCDE', 'dataflow': s['dataflow'], 'version': s['version'],
              'formato': 'csv', 'portal': url_explorador(s['dataflow'], s['version']),
              'api': url_oecd(s, ['ESP'], 2000) + '&lastNObservations=1'} for n, s in SERIES_OCDE.items()]
    filas.append({'serie': 'pib_per_capita_ppp', 'fichero': 'pib_per_capita_ppp.csv', 'proveedor': 'Banco Mundial',
                  'dataflow': PIB_BANCO_MUNDIAL, 'version': 'API v2', 'formato': 'wb', 'portal': PORTAL_BM,
                  'api': f'{URL_BANCO_MUNDIAL}/country/ESP/indicator/{PIB_BANCO_MUNDIAL}?format=json&date=2022'})
    filas.append({'serie': 'gasto_publico_por_alumno_eurostat', 'fichero': 'gasto_publico_por_alumno_eurostat.csv',
                  'proveedor': 'Eurostat', 'dataflow': EUROSTAT_GASTO_ALUMNO, 'version': '-', 'formato': 'eurostat',
                  'portal': PORTAL_EUROSTAT,
                  'api': f'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{EUROSTAT_GASTO_ALUMNO}'
                         '?format=JSON&lang=EN&geo=ES&time=2020'})
    return pd.DataFrame(filas)


def _estado_api(url: str, formato: str) -> str:
    try:
        datos = _get(url, 120)
        if formato == 'csv':
            n = max(len(datos.decode('utf-8').strip().splitlines()) - 1, 0)
        elif formato == 'wb':
            n = len(json.loads(datos)[1])
        else:
            n = len(json.loads(datos)['value'])
        return f'ok ({n} filas)' if n else 'sin datos'
    except urllib.error.HTTPError as e:
        return 'límite de consultas (429): reintentar más tarde' if e.code == 429 else f'ERROR: {e}'
    except Exception as e:
        return f'ERROR: {e}'


def comprobar_fuentes(api: bool = True) -> pd.DataFrame:
    """Comprueba cada enlace de `enlaces_fuentes`: el portal de la OCDE se valida contra su
    catálogo (la web es una SPA y siempre responde 200); los demás portales, con una petición
    HTTP; la API, pidiendo la muestra (`api=False` la omite: la OCDE limita las consultas por hora).
    Devuelve la tabla con `portal_ok` y `api_ok`."""
    import xml.etree.ElementTree as ET
    t = enlaces_fuentes()
    ns = '{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure}'
    try:
        raiz = ET.fromstring(_get(f'https://sdmx.oecd.org/public/rest/dataflow/{AGENCIA_EDU}', 120))
        catalogo = {d.get('id'): d.get('version') for d in raiz.iter(ns + 'Dataflow')}
    except Exception as e:
        catalogo = None
        print(f'No se pudo leer el catálogo de la OCDE: {e}')

    def portal_ok(f):
        if f['proveedor'] == 'OCDE':
            if catalogo is None:
                return 'sin comprobar'
            v = catalogo.get(f['dataflow'])
            return 'ok' if v == f['version'] else (f'versión actual {v}' if v else 'no existe')
        try:
            _get(f['portal'], 60)
            return 'ok'
        except Exception as e:
            return f'ERROR: {e}'

    t['portal_ok'] = [portal_ok(f) for _, f in t.iterrows()]
    t['api_ok'] = [_estado_api(f['api'], f['formato']) if api else 'sin comprobar' for _, f in t.iterrows()]
    return t


def cargar_oecd(ruta_csv: str) -> pd.DataFrame:
    """Lee un CSV de la OCDE descargado con `descargar_oecd` y deja las columnas
    `pais`, `año`, `valor` y las dimensiones presentes con nombres cortos
    (`medida`, `nivel`, `institucion`, `fuente`, `tipo_gasto`, `unidad`, `precios`,
    `edad`, `PERS_EXP_LEV`). Descarta las filas sin valor."""
    d = pd.read_csv(ruta_csv, low_memory=False)
    d = d.rename(columns={'REF_AREA': 'pais', 'TIME_PERIOD': 'año', 'OBS_VALUE': 'valor',
                          'EDUCATION_LEV': 'nivel', 'INST_TYPE_EDU': 'institucion',
                          'UNIT_MEASURE': 'unidad', 'PRICE_BASE': 'precios', 'EXP_SOURCE': 'fuente',
                          'EXPENDITURE_TYPE': 'tipo_gasto', 'MEASURE': 'medida',
                          'EXP_DESTINATION': 'institucion', 'AGE': 'edad'})   # en los dataflows de gasto, el tipo de institución es EXP_DESTINATION
    if 'año' not in d.columns:   # los dataflows de salarios llaman a la columna de tiempo REF_PERIOD
        d = d.rename(columns={'REF_PERIOD': 'año'})
    columnas = ['pais', 'año', 'valor'] + [c for c in ('medida', 'nivel', 'institucion', 'fuente', 'tipo_gasto',
                                                      'unidad', 'precios', 'edad', 'PERS_EXP_LEV') if c in d.columns]
    return d.loc[d['valor'].notna(), columnas]
