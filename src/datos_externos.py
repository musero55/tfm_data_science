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
    # Gini de la renta disponible (base de datos de distribución de ingresos; otra agencia de la OCDE)
    'gini_ocde': {
        'agencia': 'OECD.WISE.INE', 'dataflow': 'DSD_WISE_IDD@DF_IDD', 'version': '1.0',
        # FREQ, MEASURE, STATISTICAL_OPERATION, UNIT_MEASURE, AGE, METHODOLOGY, DEFINITION, POVERTY_LINE
        'dims': ['A', 'INC_DISP_GINI', None, None, '_T', 'METH2012', 'D_CUR', None],
    },
}

# Eurostat: gasto público por alumno (EUR, PPS, % del PIB per cápita), solo países de la UE.
# Sirve para rellenar el gasto de Bélgica y Países Bajos en 2012, que la OCDE no publica.
EUROSTAT_GASTO_ALUMNO = 'educ_uoe_fine09'
GEO_EUROSTAT = {'AUT': 'AT', 'BEL': 'BE', 'CZE': 'CZ', 'DEU': 'DE', 'DNK': 'DK', 'ESP': 'ES', 'EST': 'EE',
                'FIN': 'FI', 'FRA': 'FR', 'GRC': 'EL', 'HUN': 'HU', 'IRL': 'IE', 'ITA': 'IT', 'NLD': 'NL',
                'POL': 'PL', 'PRT': 'PT', 'SVK': 'SK', 'SVN': 'SI', 'SWE': 'SE'}

PIB_BANCO_MUNDIAL = 'NY.GDP.PCAP.PP.KD'   # PIB per cápita PPP, USD internacionales constantes de 2021
GINI_BANCO_MUNDIAL = 'SI.POV.GINI'        # Gini (0-100) de encuestas de hogares


def _get(url: str, timeout: int = 300) -> bytes:
    peticion = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(peticion, timeout=timeout) as respuesta:
        return respuesta.read()


def _n_dimensiones(dataflow: str, agencia: str = AGENCIA_EDU) -> int:
    """Nº de dimensiones del DSD de un dataflow (consulta el catálogo de la OCDE)."""
    import xml.etree.ElementTree as ET
    dsd = dataflow.split('@')[0]
    xml = _get(f'https://sdmx.oecd.org/public/rest/datastructure/{agencia}/{dsd}/latest', timeout=60)
    espacio_nombres_sdmx = '{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure}'
    return len(next(ET.fromstring(xml).iter(espacio_nombres_sdmx + 'DimensionList')).findall(espacio_nombres_sdmx + 'Dimension'))


def url_oecd(serie: dict, paises: list[str], inicio: int = 2000) -> str:
    """URL de consulta CSV de una serie de `SERIES_OCDE` para los países indicados."""
    agencia = serie.get('agencia', AGENCIA_EDU)
    dims = serie['dims'] if serie['dims'] is not None else [None] * (_n_dimensiones(serie['dataflow'], agencia) - 1)
    clave = '.'.join(['+'.join(paises)] + [dimension or '' for dimension in dims])
    return (f"{URL_OCDE}/{agencia},{serie['dataflow']},{serie['version']}/{clave}"
            f"?startPeriod={inicio}&format=csvfilewithlabels")


def descargar_oecd(nombre: str, ruta_carpeta: str, paises: list[str] | None = None, inicio: int = 2000) -> str:
    """Descarga una serie de `SERIES_OCDE` a `<ruta_carpeta>/<nombre>.csv`."""
    paises = paises or PAISES_OCDE_TODAS_LAS_EDICIONES
    ruta = os.path.join(ruta_carpeta, f'{nombre}.csv')
    datos = _get(url_oecd(SERIES_OCDE[nombre], paises, inicio))   # antes de abrir el fichero: si falla, no deja uno vacío
    with open(ruta, 'wb') as fichero:
        fichero.write(datos)
    return ruta


def descargar_banco_mundial(indicador: str, nombre: str, ruta_carpeta: str, paises: list[str] | None = None,
                            inicio: int = 2000, fin: int = 2023) -> str:
    """Indicador del Banco Mundial a `<ruta_carpeta>/<nombre>.csv` (columnas: pais, año, valor)."""
    paises = paises or PAISES_OCDE_TODAS_LAS_EDICIONES
    url = (f"{URL_BANCO_MUNDIAL}/country/{';'.join(paises)}/indicator/{indicador}"
           f"?format=json&date={inicio}:{fin}&per_page=20000")
    _, filas = json.loads(_get(url))
    datos = pd.DataFrame(
        [{'pais': observacion['countryiso3code'], 'año': int(observacion['date']), 'valor': observacion['value']} for observacion in filas]
    ).sort_values(['pais', 'año'])
    ruta = os.path.join(ruta_carpeta, f'{nombre}.csv')
    datos.to_csv(ruta, index=False)
    return ruta


def descargar_pib(ruta_carpeta: str, paises: list[str] | None = None, inicio: int = 2000, fin: int = 2023) -> str:
    """PIB per cápita PPP (Banco Mundial) a `<ruta_carpeta>/pib_per_capita_ppp.csv`."""
    return descargar_banco_mundial(PIB_BANCO_MUNDIAL, 'pib_per_capita_ppp', ruta_carpeta, paises, inicio, fin)


# Indicadores macroeconómicos del Banco Mundial (WDI): código -> (nombre de columna, qué mide, tema)
INDICADORES_BANCO_MUNDIAL = {
    'NY.GDP.MKTP.KD.ZG': ('crecimiento_pib_pct_anual', 'Crecimiento anual del PIB (%)', 'Ciclo económico'),
    'FP.CPI.TOTL.ZG': ('inflacion_precios_consumo_pct_anual', 'Inflación: variación anual de los precios al consumo (%)', 'Ciclo económico'),
    'SL.UEM.TOTL.ZS': ('desempleo_total_pct_poblacion_activa', 'Desempleo total (% de la población activa; estimación de la OIT)', 'Mercado de trabajo'),
    'SL.UEM.1524.ZS': ('desempleo_juvenil_15_24_pct_poblacion_activa', 'Desempleo juvenil de 15 a 24 años (% de la población activa de esa edad)', 'Mercado de trabajo'),
    'SL.TLF.CACT.ZS': ('participacion_laboral_pct_poblacion_15_o_mas', 'Participación laboral (% de la población de 15 años o más)', 'Mercado de trabajo'),
    'NV.SRV.TOTL.ZS': ('valor_añadido_servicios_pct_pib', 'Valor añadido de los servicios (% del PIB)', 'Estructura productiva'),
    'NV.IND.TOTL.ZS': ('valor_añadido_industria_pct_pib', 'Valor añadido de la industria (% del PIB)', 'Estructura productiva'),
    'NV.AGR.TOTL.ZS': ('valor_añadido_agricultura_pct_pib', 'Valor añadido de la agricultura (% del PIB)', 'Estructura productiva'),
    'NE.TRD.GNFS.ZS': ('comercio_exterior_pct_pib', 'Comercio exterior: exportaciones más importaciones (% del PIB)', 'Estructura productiva'),
    'GB.XPD.RSDV.GD.ZS': ('gasto_investigacion_desarrollo_pct_pib', 'Gasto en investigación y desarrollo (% del PIB)', 'Estructura productiva'),
    'SP.URB.TOTL.IN.ZS': ('poblacion_urbana_pct', 'Población urbana (% de la población total)', 'Demografía y territorio'),
    'SP.POP.0014.TO.ZS': ('poblacion_0_14_años_pct', 'Población de 0 a 14 años (% de la población total)', 'Demografía y territorio'),
    'SP.POP.65UP.TO.ZS': ('poblacion_65_o_mas_años_pct', 'Población de 65 años o más (% de la población total)', 'Demografía y territorio'),
    'IT.NET.USER.ZS': ('usuarios_internet_pct_poblacion', 'Personas que usan internet (% de la población)', 'Desarrollo y servicios'),
    'SH.XPD.CHEX.GD.ZS': ('gasto_sanitario_pct_pib', 'Gasto sanitario corriente (% del PIB)', 'Desarrollo y servicios'),
}


def descargar_macro_banco_mundial(ruta_carpeta: str, paises: list[str] | None = None, inicio: int = 2010, fin: int = 2023) -> str:
    """Los indicadores de `INDICADORES_BANCO_MUNDIAL` a `<ruta_carpeta>/macro_banco_mundial.csv`
    (formato largo: pais, año, indicador (código del Banco Mundial), valor)."""
    paises = paises or PAISES_OCDE_TODAS_LAS_EDICIONES
    observaciones = []
    for indicador in INDICADORES_BANCO_MUNDIAL:
        url = (f"{URL_BANCO_MUNDIAL}/country/{';'.join(paises)}/indicator/{indicador}"
               f"?format=json&date={inicio}:{fin}&per_page=20000")
        _, filas = json.loads(_get(url))
        observaciones += [{'pais': fila['countryiso3code'], 'año': int(fila['date']), 'indicador': indicador, 'valor': fila['value']} for fila in filas]
    datos = pd.DataFrame(observaciones).sort_values(['indicador', 'pais', 'año'])
    ruta = os.path.join(ruta_carpeta, 'macro_banco_mundial.csv')
    datos.to_csv(ruta, index=False)
    return ruta


def descargar_eurostat(ruta_carpeta: str, inicio: int = 2010) -> str:
    """Gasto público por alumno de Eurostat (`educ_uoe_fine09`) a
    `<ruta_carpeta>/gasto_publico_por_alumno_eurostat.csv`
    (columnas: pais, año, nivel, unidad, valor). Solo países de la UE."""
    geos = ''.join(f'&geo={codigo_geo}' for codigo_geo in GEO_EUROSTAT.values())
    años = ''.join(f'&time={año_consulta}' for año_consulta in range(inicio, 2024))
    url = (f'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{EUROSTAT_GASTO_ALUMNO}'
           f'?format=JSON&lang=EN{geos}{años}')
    respuesta = json.loads(_get(url))
    ids, tamaños_dimensiones = respuesta['id'], respuesta['size']
    categorias = {dimension: {posicion_categoria: codigo for codigo, posicion_categoria in respuesta['dimension'][dimension]['category']['index'].items()} for dimension in ids}   # posicion -> codigo
    inverso_geo = {v: k for k, v in GEO_EUROSTAT.items()}
    filas = []
    for posicion_plana, valor in respuesta['value'].items():
        posicion_plana = int(posicion_plana)
        codigos_observacion = {}
        for dimension, tamaño_dimension in zip(reversed(ids), reversed(tamaños_dimensiones)):   # índice plano: la última dimensión varía más rápido
            codigos_observacion[dimension] = categorias[dimension][posicion_plana % tamaño_dimension]
            posicion_plana //= tamaño_dimension
        filas.append({'pais': inverso_geo[codigos_observacion['geo']], 'año': int(codigos_observacion['time']), 'nivel': codigos_observacion['isced11'],
                      'unidad': codigos_observacion['unit'], 'valor': valor})
    datos = pd.DataFrame(filas).sort_values(['pais', 'año', 'nivel', 'unidad'])
    ruta = os.path.join(ruta_carpeta, 'gasto_publico_por_alumno_eurostat.csv')
    datos.to_csv(ruta, index=False)
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
    ruta = os.path.join(ruta_carpeta, 'gini_banco_mundial.csv')
    if forzar or not os.path.exists(ruta):
        print('Descargando gini_banco_mundial...')
        descargar_banco_mundial(GINI_BANCO_MUNDIAL, 'gini_banco_mundial', ruta_carpeta)
    rutas['gini_banco_mundial'] = ruta
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


def url_explorador(dataflow: str, version: str, agencia: str = AGENCIA_EDU) -> str:
    """Página del dataflow en el OECD Data Explorer (donde se ven y descargan los datos)."""
    return (f"{URL_EXPLORADOR}?df%5Bds%5D=DisseminateFinalDMZ&df%5Bid%5D={dataflow.replace('@', '%40')}"
            f"&df%5Bag%5D={agencia}&df%5Bvs%5D={version}")


def enlaces_fuentes() -> pd.DataFrame:
    """Una fila por fichero descargado: proveedor, portal donde se ven los datos y consulta
    de API de muestra (España, último dato) que devuelve el mismo formato que la descarga."""
    filas = [{'serie': nombre_serie, 'fichero': f'{nombre_serie}.csv', 'proveedor': 'OCDE', 'agencia': definicion_serie.get('agencia', AGENCIA_EDU),
              'dataflow': definicion_serie['dataflow'], 'version': definicion_serie['version'],
              'formato': 'csv', 'portal': url_explorador(definicion_serie['dataflow'], definicion_serie['version'], definicion_serie.get('agencia', AGENCIA_EDU)),
              'api': url_oecd(definicion_serie, ['ESP'], 2000) + '&lastNObservations=1'} for nombre_serie, definicion_serie in SERIES_OCDE.items()]
    filas.append({'serie': 'pib_per_capita_ppp', 'fichero': 'pib_per_capita_ppp.csv', 'proveedor': 'Banco Mundial',
                  'dataflow': PIB_BANCO_MUNDIAL, 'version': 'API v2', 'formato': 'wb', 'portal': PORTAL_BM,
                  'api': f'{URL_BANCO_MUNDIAL}/country/ESP/indicator/{PIB_BANCO_MUNDIAL}?format=json&date=2022'})
    filas.append({'serie': 'gini_banco_mundial', 'fichero': 'gini_banco_mundial.csv', 'proveedor': 'Banco Mundial',
                  'dataflow': GINI_BANCO_MUNDIAL, 'version': 'API v2', 'formato': 'wb',
                  'portal': f'https://data.worldbank.org/indicator/{GINI_BANCO_MUNDIAL}',
                  'api': f'{URL_BANCO_MUNDIAL}/country/ESP/indicator/{GINI_BANCO_MUNDIAL}?format=json&date=2018'})
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
            n_filas = max(len(datos.decode('utf-8').strip().splitlines()) - 1, 0)
        elif formato == 'wb':
            n_filas = len(json.loads(datos)[1])
        else:
            n_filas = len(json.loads(datos)['value'])
        return f'ok ({n_filas} filas)' if n_filas else 'sin datos'
    except urllib.error.HTTPError as e:
        return 'límite de consultas (429): reintentar más tarde' if error.code == 429 else f'ERROR: {error}'
    except Exception as e:
        return f'ERROR: {error}'


def comprobar_fuentes(api: bool = True) -> pd.DataFrame:
    """Comprueba cada enlace de `enlaces_fuentes`: el portal de la OCDE se valida contra su
    catálogo (la web es una SPA y siempre responde 200); los demás portales, con una petición
    HTTP; la API, pidiendo la muestra (`api=False` la omite: la OCDE limita las consultas por hora).
    Devuelve la tabla con `portal_ok` y `api_ok`."""
    import xml.etree.ElementTree as ET
    tabla_enlaces = enlaces_fuentes()
    espacio_nombres_sdmx = '{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure}'
    catalogos = {}
    for agencia in tabla_enlaces.loc[tabla_enlaces['proveedor'] == 'OCDE', 'agencia'].unique():
        try:
            raiz = ET.fromstring(_get(f'https://sdmx.oecd.org/public/rest/dataflow/{agencia}', 120))
            catalogos[agencia] = {dataflow_catalogo.get('id'): dataflow_catalogo.get('version') for dataflow_catalogo in raiz.iter(espacio_nombres_sdmx + 'Dataflow')}
        except Exception as e:
            print(f'No se pudo leer el catálogo de la OCDE ({agencia}): {error}')

    def portal_ok(enlace):
        if enlace['proveedor'] == 'OCDE':
            catalogo = catalogos.get(enlace['agencia'])
            if catalogo is None:
                return 'sin comprobar'
            version_actual = catalogo.get(enlace['dataflow'])
            return 'ok' if version_actual == enlace['version'] else (f'versión actual {version_actual}' if version_actual else 'no existe')
        try:
            _get(enlace['portal'], 60)
            return 'ok'
        except Exception as e:
            return f'ERROR: {error}'

    tabla_enlaces['portal_ok'] = [portal_ok(enlace) for _, enlace in tabla_enlaces.iterrows()]
    tabla_enlaces['api_ok'] = [_estado_api(enlace['api'], enlace['formato']) if api else 'sin comprobar' for _, enlace in tabla_enlaces.iterrows()]
    return tabla_enlaces


def cargar_oecd(ruta_csv: str) -> pd.DataFrame:
    """Lee un CSV de la OCDE descargado con `descargar_oecd` y deja las columnas
    `pais`, `año`, `valor` y las dimensiones presentes con nombres cortos
    (`medida`, `nivel`, `institucion`, `fuente`, `tipo_gasto`, `unidad`, `precios`,
    `edad`, `PERS_EXP_LEV`). Descarta las filas sin valor."""
    datos = pd.read_csv(ruta_csv, low_memory=False)
    datos = datos.rename(columns={'REF_AREA': 'pais', 'TIME_PERIOD': 'año', 'OBS_VALUE': 'valor',
                          'EDUCATION_LEV': 'nivel', 'INST_TYPE_EDU': 'institucion',
                          'UNIT_MEASURE': 'unidad', 'PRICE_BASE': 'precios', 'EXP_SOURCE': 'fuente',
                          'EXPENDITURE_TYPE': 'tipo_gasto', 'MEASURE': 'medida',
                          'EXP_DESTINATION': 'institucion', 'AGE': 'edad'})   # en los dataflows de gasto, el tipo de institución es EXP_DESTINATION
    if 'año' not in datos.columns:   # los dataflows de salarios llaman a la columna de tiempo REF_PERIOD
        datos = datos.rename(columns={'REF_PERIOD': 'año'})
    columnas = ['pais', 'año', 'valor'] + [columna for columna in ('medida', 'nivel', 'institucion', 'fuente', 'tipo_gasto',
                                                      'unidad', 'precios', 'edad', 'PERS_EXP_LEV') if columna in datos.columns]
    return datos.loc[datos['valor'].notna(), columnas]
