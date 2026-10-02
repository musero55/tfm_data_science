"""
pisa_pipeline.py
=================

Funciones reutilizables para el procesamiento de microdatos PISA (ediciones en
formato digital, 2015 en adelante) para los tres dominios de competencia
principales: Matemáticas (MATH), Lectura (READ) y Ciencias (SCIE).


⚠️ IMPORTANTE — variabilidad entre ediciones:
    - El formato del PUF cambia entre ciclos: 2015/2018/2022 se distribuyen en
      SAS (.sas7bdat), pero 2025 solo trae SPSS (.sav). `encontrar_archivo`
      detecta el formato automáticamente por extensión (ver
      `LECTORES_PYREADSTAT`), y los nombres de fichero también cambian de
      capitalización entre ciclos (revisa los que descargues realmente de la
      OCDE).
    - No todas las variables de contexto están disponibles en todas las
      ediciones (p. ej. ICTRES depende de la participación opcional del país
      en el módulo ICT ese año concreto).
    - Los índices derivados (ESCS, HOMEPOS, etc.) pueden estar en escalas no
      directamente comparables entre ciclos. Para comparaciones longitudinales
      rigurosas, la OCDE publica "rescaled trend indices" específicos —
      revísalo si vas a comparar ediciones entre sí, no solo procesarlas por
      separado.
"""

from __future__ import annotations

import os
import gc
import numpy as np
import pandas as pd
import pyreadstat
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (necesario para habilitar IterativeImputer)
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge



# 0. Configuración de ediciones PISA

DOMINIOS = ['MATH', 'READ', 'SCIE']

# Dominio principal (foco) de cada edición desde el paso a formato digital
# (mayor número de ítems / mayor precisión de medición ese ciclo concreto).
#
# 'patron_estudiantes' / 'patron_colegios' son subcadenas que deben aparecer en
# el nombre del fichero (comparación insensible a mayúsculas/minúsculas), no
# el nombre exacto del archivo ni su extensión — `encontrar_archivo` prueba
# tanto .sas7bdat como .sav. Esto evita depender de la capitalización concreta
# que use la OCDE en cada edición (p. ej. 'CY08MSP_STU_QQQ.SAS7BDAT' en 2022
# vs 'cy07_msu_stu_qqq.sas7bdat' en 2018): basta con que el patrón aparezca en
# el nombre real, sea cual sea su capitalización o formato.
#
# 'alias_estudiantes' / 'alias_colegios' ({nombre_canónico: nombre_en_el_fichero})
# traducen a los nombres que usa el pipeline las columnas que una edición
# llama distinto. 'n_pv' es el número de valores plausibles de la edición
# (10 desde 2015, 5 en ediciones anteriores).
EDICIONES_PISA = {
    # PISA 2012 solo se distribuye como .txt de ancho fijo + script .sas: se
    # convierte a .sav con `src/pisa_txt_a_sav.py` antes de usarlo aquí.
    2012: {
        'dominio_principal': 'MATH',  # Matemáticas fue dominio focal en 2003, 2012 y 2022
        'patron_estudiantes': 'stu12',
        'patron_colegios': 'scq12',
        'n_pv': 5,
        'alias_estudiantes': {
            'CNTSCHID': 'SCHOOLID',
            'CNTSTUID': 'StIDStd',
            'HISEI': 'hisei',
            'PAREDINT': 'PARED',
            'CULTPOSS': 'CULTPOS',
        },
        'alias_colegios': {'CNTSCHID': 'SCHOOLID', 'W_SCHGRNRABWT': 'W_FSCHWT'},
    },
    2015: {
        'dominio_principal': 'SCIE',
        'patron_estudiantes': 'stu_qqq',
        'patron_colegios': 'sch_qqq',
        # Como 2012, el fichero de 2015 trae estas dos en minúscula / con otro nombre
        'alias_estudiantes': {'HISEI': 'hisei', 'PAREDINT': 'PARED'},
    },
    2018: {
        'dominio_principal': 'READ',
        'patron_estudiantes': 'stu_qqq',
        'patron_colegios': 'sch_qqq',
    },
    2022: {
        'dominio_principal': 'MATH',
        'patron_estudiantes': 'stu_qqq',
        'patron_colegios': 'sch_qqq',
    },
    2025: {
        'dominio_principal': 'SCIE',  # 3ª vez que Ciencias es el dominio focal, tras 2006 y 2015
        'patron_estudiantes': 'stu_qqq',
        'patron_colegios': 'sch_qqq',
    },
}

COLUMNAS_BASICAS = ['CNT', 'CNTSCHID', 'STRATUM', 'W_FSTUWT']

# Identificador de alumno: no interviene en el diseño muestral, pero permite
# unir con ficheros externos por alumno (p. ej. `escs_trend` de la OCDE). Nunca
# se imputa. Si una edición no lo trae, solo se avisa.
COLUMNAS_ID_ALUMNO = ['CNTSTUID']

# Peso base del centro (ajustado por no-respuesta) en el fichero de colegios.
# Al submuestrear escuelas, la tabla de colegios también necesita el factor de
# inflación: W_SCHGRNRABWT_adj. En 2012 se llama W_FSCHWT (alias en EDICIONES_PISA).
COLUMNAS_PESO_COLEGIO = ['W_SCHGRNRABWT']

# Subcarpeta de `ruta_intermedios` donde se guardan los dos parquets anuales.
CARPETA_DATOS_ANUALES = 'datos_anuales'
# Subcarpeta con las mismas tablas tras imputar las columnas de contexto
# (las que estén en `COLUMNAS_CONTEXTO_*` y tengan nulos); el resto sigue igual.
CARPETA_DATOS_ANUALES_IMPUTADOS = 'datos_anuales_imputados'

# Países miembros de la OCDE durante TODAS las ediciones 2012-2022 y con datos en
# todas ellas (33). Derivado del flag `OECD` de los ficheros de centros: Letonia,
# Lituania, Colombia y Costa Rica se incorporaron después de 2012; Luxemburgo es
# miembro desde siempre, pero no tiene datos en PISA 2022, así que no puede
# formar parte de un panel completo. No es la lista de miembros actuales.
PAISES_OCDE_TODAS_LAS_EDICIONES = [
    'AUS', 'AUT', 'BEL', 'CAN', 'CHE', 'CHL', 'CZE', 'DEU', 'DNK', 'ESP', 'EST',
    'FIN', 'FRA', 'GBR', 'GRC', 'HUN', 'IRL', 'ISL', 'ISR', 'ITA', 'JPN', 'KOR',
    'MEX', 'NLD', 'NOR', 'NZL', 'POL', 'PRT', 'SVK', 'SVN', 'SWE', 'TUR', 'USA',
]

# Índices de contexto a nivel estudiante
COLUMNAS_CONTEXTO_ESTUDIANTE = [
    'ESCS', 'HISEI', 'PAREDINT', 'HOMEPOS', 'WEALTH',
    'CULTPOSS', 'HEDRES', 'ICTRES',
    'ANXMAT', 'MATHEFF', 'MATHINT', 'MATHBEH',
    'BELONG', 'BULLY', 'DISCLIMA', 'TEACHSUP',
    'OUTHOURS',  # tiempo de estudio fuera del horario escolar (2012 y 2015)
    'ICTWKDY', 'ICTWKEND',  # frecuencia de uso TIC entre semana/fin de semana (2022/2025)
    'ICTDISTR',  # malestar por contenido online/ciberacoso (2022/2025)
    'SKIPPING', 'TARDYSD',  # absentismo/impuntualidad (2022/2025)
]

# Índices de contexto a nivel centro
COLUMNAS_CONTEXTO_COLEGIO = [
    'STRATIO', 'PROPMATH', 'SCHLCLI', 'EDUSHORT', 'STAFFSHORT',
    'SCHLTYPE',  # titularidad del centro: 1=privado independiente, 2=privado
                 # concertado (dependiente del gobierno), 3=público. Presente
                 # en las 4 ediciones (2015-2025).
    'CLSIZE',  # tamaño de clase, presente en las 4 ediciones.
]


def columnas_pv(dominio: str, n_pv: int = 10) -> list[str]:
    """Nombres de los valores plausibles (PV1..PV`n_pv`) de un dominio.

    PISA publica 10 PV por dominio desde 2015 y 5 en ediciones anteriores
    (p. ej. 2012). Las funciones que agregan filtran por las columnas que
    existan, así que `n_pv=10` sirve para cualquier edición.
    """
    if dominio not in DOMINIOS:
        raise ValueError(f"Dominio '{dominio}' no reconocido. Usa uno de {DOMINIOS}")
    return [f'PV{i}{dominio}' for i in range(1, n_pv + 1)]


# La OCDE no siempre distribuye la misma edición en el mismo formato: PISA
# 2015/2018/2022 se pidieron en SAS, pero el PUF de PISA 2025 solo trae .sav
# (SPSS). pyreadstat expone una función de lectura por formato con la misma
# firma (usecols, metadataonly...), así que basta con elegir la función según
# la extensión real del fichero para que el resto del pipeline no distinga
# entre formatos.
LECTORES_PYREADSTAT = {
    '.sas7bdat': pyreadstat.read_sas7bdat,
    '.sav': pyreadstat.read_sav,
}


def lector_para(ruta_archivo: str):
    """Devuelve la función de pyreadstat (`read_sas7bdat`/`read_sav`) adecuada
    para `ruta_archivo` según su extensión."""
    extension = os.path.splitext(ruta_archivo)[1].lower()
    if extension not in LECTORES_PYREADSTAT:
        raise ValueError(
            f"Formato '{extension}' no soportado ({ruta_archivo}). "
            f"Formatos soportados: {list(LECTORES_PYREADSTAT)}"
        )
    return LECTORES_PYREADSTAT[extension]


def encontrar_archivo(ruta_carpeta: str, patron: str, extensiones: tuple[str, ...] | None = None) -> str:
    """Busca en `ruta_carpeta` un archivo cuyo nombre contenga `patron`
    (insensible a mayúsculas/minúsculas) y termine en alguna de `extensiones`
    (por defecto, cualquiera de los formatos soportados por pyreadstat: SAS o
    SPSS — la OCDE no siempre distribuye la misma edición en el mismo formato).

    Evita depender del nombre exacto/capitalización que use la OCDE en cada
    edición: basta con reconocer el patrón común ('stu_qqq', 'sch_qqq').

    Lanza FileNotFoundError con un mensaje explícito si no encuentra ningún
    candidato, o si encuentra más de uno (para no elegir uno al azar).
    """
    extensiones = tuple(e.lower() for e in (extensiones or LECTORES_PYREADSTAT.keys()))
    patron_low = patron.lower()

    candidatos = [
        f for f in os.listdir(ruta_carpeta)
        if patron_low in f.lower() and f.lower().endswith(extensiones)
    ]

    if not candidatos:
        raise FileNotFoundError(
            f"No se ha encontrado ningún archivo con patrón '{patron}' y "
            f"extensión en {extensiones} en {ruta_carpeta}. "
            f"Archivos presentes: {os.listdir(ruta_carpeta)}"
        )
    if len(candidatos) > 1:
        raise FileNotFoundError(
            f"Se han encontrado varios archivos que coinciden con el patrón "
            f"'{patron}' en {ruta_carpeta}: {candidatos}. Afina el patrón."
        )

    return os.path.join(ruta_carpeta, candidatos[0])


def columnas_disponibles(
    ruta_archivo: str,
    columnas: list[str],
    alias: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Compara `columnas` (nombres canónicos del pipeline) contra las columnas
    realmente presentes en `ruta_archivo` (lee solo metadata, no los datos).
    Devuelve (presentes, ausentes), siempre con nombres canónicos.

    `alias` ({canónico: nombre_en_el_fichero}) traduce las columnas que esa
    edición llama distinto antes de comprobar su existencia.

    Necesario porque no todas las ediciones administran todos los módulos de
    contexto (p. ej. ICT), así que una columna que existe en el fichero de un
    año puede no existir en el de otro.
    """
    alias = alias or {}
    _, meta = lector_para(ruta_archivo)(ruta_archivo, metadataonly=True)
    columnas_reales = set(meta.column_names)
    presentes = [c for c in columnas if alias.get(c, c) in columnas_reales]
    ausentes = [c for c in columnas if alias.get(c, c) not in columnas_reales]
    return presentes, ausentes


def _traducir_columnas(
    columnas: list[str] | None, alias: dict[str, str] | None
) -> tuple[list[str] | None, dict[str, str]]:
    """Nombres canónicos -> (nombres a leer del fichero, renombrado inverso
    nombre_fichero -> canónico para aplicar tras la lectura). Con
    `columnas=None` se leen todas las del fichero (`a_leer=None`) y se renombran
    todas las que tengan alias."""
    alias = alias or {}
    if columnas is None:
        return None, {v: k for k, v in alias.items() if v != k}
    a_leer = [alias.get(c, c) for c in columnas]
    renombrar = {alias[c]: c for c in columnas if c in alias and alias[c] != c}
    return a_leer, renombrar


# 1. Selección de escuelas y submuestreo estratificado

def leer_escuelas_unicas(
    ruta_estudiantes: str,
    cols_basicas: list[str],
    alias: dict[str, str] | None = None,
    paises: list[str] | None = None,
) -> pd.DataFrame:
    """Lee las columnas identificativas básicas y devuelve una fila por escuela
    única (CNT, CNTSCHID, STRATUM), necesaria para diseñar el submuestreo. Si
    se pasa `paises`, solo se consideran las escuelas de esos países (el
    submuestreo es por estrato, así que filtrar antes no cambia qué escuelas se
    eligen en los países que se conservan)."""
    a_leer, renombrar = _traducir_columnas(cols_basicas, alias)
    df_basic, _ = lector_para(ruta_estudiantes)(ruta_estudiantes, usecols=a_leer)
    df_basic = df_basic.rename(columns=renombrar)
    if paises is not None:
        df_basic = df_basic[df_basic['CNT'].isin(paises)]
    escuelas_unicas = df_basic.drop_duplicates(subset=['CNT', 'CNTSCHID', 'STRATUM']).copy()
    del df_basic
    gc.collect()
    return escuelas_unicas


def submuestreo_estratificado(
    escuelas_unicas: pd.DataFrame,
    fraccion: float = 0.20,
    seed: int = 42,
) -> tuple[set, dict, pd.DataFrame]:
    """Selecciona un porcentaje de escuelas por cada estrato (país + STRATUM),
    garantizando un mínimo de 2 escuelas cuando el estrato tiene 2 o más, y
    calcula el factor de inflación asociado a cada estrato.

    Devuelve
    --------
    escuelas_seleccionadas : set de tuplas (CNT, STRATUM, CNTSCHID)
    factores_dict : dict {(CNT, STRATUM): factor_inflacion}
    df_groups : dataframe resumen por estrato (trazabilidad / EDA del muestreo)
    """
    grupos = escuelas_unicas.groupby(['CNT', 'STRATUM'])
    df_groups = grupos.size().reset_index(name='total_escuelas')

    df_groups['n_submuestra'] = np.ceil(df_groups['total_escuelas'] * fraccion).astype(int)
    df_groups['n_submuestra'] = np.where(
        (df_groups['total_escuelas'] >= 2) & (df_groups['n_submuestra'] < 2),
        2,
        df_groups['n_submuestra'],
    )
    df_groups['factor_inflacion'] = df_groups['total_escuelas'] / df_groups['n_submuestra']

    groups_dict = df_groups.set_index(['CNT', 'STRATUM']).to_dict('index')

    escuelas_seleccionadas: set = set()
    factores_dict: dict = {}

    for (pais, estrato), grupo in grupos:
        datos_estrato = groups_dict[(pais, estrato)]
        n_colegios = datos_estrato['n_submuestra']
        # Se asigna una única vez por estrato (antes se reasignaba en cada
        # iteración del bucle interno de colegios de forma redundante)
        factores_dict[(pais, estrato)] = datos_estrato['factor_inflacion']

        muestra = grupo.sample(n=n_colegios, random_state=seed)
        for colegio_id in muestra['CNTSCHID'].values:
            escuelas_seleccionadas.add((pais, estrato, colegio_id))

    return escuelas_seleccionadas, factores_dict, df_groups



# 2. Lectura de estudiantes filtrada y ajuste de ponderaciones

def leer_estudiantes_submuestra(
    ruta_estudiantes: str,
    columnas: list[str] | None,
    escuelas_seleccionadas: set | None = None,
    factores_dict: dict | None = None,
    chunk_size: int = 10000,
    alias: dict[str, str] | None = None,
    paises: list[str] | None = None,
) -> pd.DataFrame:
    """Lee el fichero de estudiantes por bloques y conserva solo las filas
    pedidas. `columnas` y el resultado usan nombres canónicos (`columnas=None`
    lee todas las del fichero); `alias` traduce los que el fichero llame
    distinto.

    * Con `escuelas_seleccionadas` y `factores_dict` (submuestreo): conserva
      solo esas escuelas y ajusta W_FSTUWT según el factor de inflación del
      estrato (`factor_inflacion`, `W_FSTUWT_adj`).
    * Sin ellos (muestra completa): conserva todos los alumnos de `paises` (o
      de todos los países si es None) y no toca ningún peso.
    """
    a_leer, renombrar = _traducir_columnas(columnas, alias)
    reader = pyreadstat.read_file_in_chunks(
        lector_para(ruta_estudiantes),
        ruta_estudiantes,
        chunksize=chunk_size,
        usecols=a_leer,
    )

    chunks_filtrados = []
    for chunk, _ in reader:
        chunk = chunk.rename(columns=renombrar)
        if chunk.columns.duplicated().any():
            raise ValueError(f'Columnas duplicadas tras aplicar los alias: '
                             f'{chunk.columns[chunk.columns.duplicated()].tolist()}')

        if escuelas_seleccionadas is None:
            mascara = chunk['CNT'].isin(paises) if paises is not None else pd.Series(True, index=chunk.index)
            chunk_filtrado = chunk[mascara].copy()
            if chunk_filtrado.empty:
                continue
        else:
            clave_colegio = pd.Series(
                list(zip(chunk['CNT'], chunk['STRATUM'], chunk['CNTSCHID'])),
                index=chunk.index,
            )
            chunk_filtrado = chunk[clave_colegio.isin(escuelas_seleccionadas)].copy()
            if chunk_filtrado.empty:
                continue

            clave_factor = pd.Series(
                list(zip(chunk_filtrado['CNT'], chunk_filtrado['STRATUM'])),
                index=chunk_filtrado.index,
            )
            chunk_filtrado['factor_inflacion'] = clave_factor.map(factores_dict)
            chunk_filtrado['W_FSTUWT_adj'] = chunk_filtrado['W_FSTUWT'] * chunk_filtrado['factor_inflacion']

        chunks_filtrados.append(chunk_filtrado)

    if not chunks_filtrados:
        raise ValueError(
            "No se ha encontrado ningún estudiante que cumpla el filtro. "
            "Revisa el fichero de origen, los países y el set de escuelas."
        )

    return pd.concat(chunks_filtrados, ignore_index=True)


def calcular_senwt(
    df_alumnos: pd.DataFrame,
    peso_col: str = 'W_FSTUWT',
    nombre: str = 'SENWT',
    pais_col: str = 'CNT',
    total_senado: int = 5000,
) -> pd.DataFrame:
    """Peso de "senado": reescala `peso_col` para que cada país sume
    `total_senado` (5.000, convención OCDE). Los ficheros de 2015 en adelante ya
    traen `SENWT`; el de 2012 no, y se calcula igual con esta función."""
    df_alumnos = df_alumnos.copy()
    suma_pesos_pais = df_alumnos.groupby(pais_col)[peso_col].transform('sum')
    df_alumnos[nombre] = df_alumnos[peso_col] * (total_senado / suma_pesos_pais)
    return df_alumnos


def calcular_senwt_adj(
    df_alumnos: pd.DataFrame,
    peso_col: str = 'W_FSTUWT_adj',
    pais_col: str = 'CNT',
    total_senado: int = 5000,
) -> pd.DataFrame:
    """Peso de senado tras un submuestreo: la suma de `W_FSTUWT_adj` por país
    vuelve a sumar `total_senado` (columna `SENWT_adj`)."""
    return calcular_senwt(df_alumnos, peso_col, 'SENWT_adj', pais_col, total_senado)


def leer_colegios(
    ruta_colegios: str,
    columnas: list[str] | None,
    alias: dict[str, str] | None = None,
) -> pd.DataFrame:
    a_leer, renombrar = _traducir_columnas(columnas, alias)
    df_colegios, _ = lector_para(ruta_colegios)(ruta_colegios, usecols=a_leer)
    return df_colegios.rename(columns=renombrar)


def unir_alumnos_colegios(
    df_alumnos: pd.DataFrame,
    df_colegios: pd.DataFrame,
    claves=('CNT', 'STRATUM', 'CNTSCHID'),
) -> pd.DataFrame:
    return pd.merge(df_alumnos, df_colegios, on=list(claves), how='left')



# 3. Optimización de tipos y análisis de nulos

def optimizar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast de columnas float64 a un tipo numérico más pequeño cuando es
    posible, para reducir memoria."""
    df = df.copy()
    # Los identificadores no se pasan a float32: con 8 dígitos perderían
    # precisión si tuvieran algún nulo.
    for col in df.select_dtypes(include=['float64', 'float32']).columns:
        if col in ('CNTSCHID', 'CNTSTUID'):
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
        if df[col].dtype in ['float64', 'float32']:
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df


def porcentaje_nulos(df: pd.DataFrame) -> pd.Series:
    return (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)


def porcentaje_nulos_por_pais(df: pd.DataFrame, columna: str, pais_col: str = 'CNT') -> pd.DataFrame:
    resultado = df[columna].isna().groupby(df[pais_col]).mean() * 100
    return (
        resultado.reset_index(name=f'pct_nulos_{columna}')
        .sort_values(f'pct_nulos_{columna}', ascending=False)
    )



# 4. Imputación multivariante (MICE)

def imputar_multivariante(
    df: pd.DataFrame,
    columnas_a_imputar: list[str] | None = None,
    seed: int = 42,
    max_iter: int = 10,
    excluir_de_imputacion: list[str] | None = None,
) -> pd.DataFrame:
    """Imputa nulos con IterativeImputer (MICE, estimador BayesianRidge) y
    conserva las columnas indicadoras `<col>_missing` generadas por
    add_indicator=True.

    A diferencia de la versión original del notebook 1, aquí SÍ se incorporan
    al dataframe final las columnas indicadoras (antes se generaban y se
    descartaban sin usarse).

    Los PV1..PV10 de cada dominio se excluyen SIEMPRE de la imputación: su
    ausencia no es un nulo de contexto missing-at-random, sino no-respuesta
    total (el estudiante no llegó a hacer la prueba). Intentar rellenarlos con
    una regresión sobre variables de contexto no tiene base metodológica (la
    OCDE los genera con un modelo de respuesta al ítem) y en la práctica
    produce matrices de diseño mal condicionadas: cuando faltan, faltan los 30
    PV a la vez, así que sus 30 columnas indicadoras `_missing` quedan
    duplicadas entre sí. Las filas sin ningún PV se eliminan antes de imputar
    el resto de columnas.

    Se usa `BayesianRidge` (el estimador que recomienda sklearn por defecto
    para `IterativeImputer`) en vez de `Ridge`: varios índices de contexto
    derivados de las mismas preguntas del cuestionario (HOMEPOS, WEALTH,
    CULTPOSS, HEDRES, ICTRES, ESCS...) faltan juntos cuando el estudiante se
    salta esa sección, lo que deja sus indicadoras `_missing` muy
    correlacionadas entre sí y producía matrices mal condicionadas con
    `Ridge` — incluso subiendo `alpha` a 10 (probado en PISA 2018/2022/2025)
    seguían quedando `LinAlgWarning` sueltos (17 en PISA 2022). `BayesianRidge`
    estima su propia fuerza de regularización a partir de los datos en cada
    iteración en vez de usar un `alpha` fijo elegido a mano, y comprobado
    empíricamente sobre PISA 2022: elimina el `LinAlgWarning` por completo y
    da valores imputados prácticamente idénticos a `Ridge(alpha=10)`
    (correlación 1.000, diferencias de milésimas) — mismo resultado, sin la
    colinealidad residual y sin tener que justificar un `alpha` a mano.

    Los identificadores de `COLUMNAS_BASICAS` (`CNT`, `CNTSCHID`, `STRATUM`,
    `W_FSTUWT`) también se excluyen siempre: no tiene sentido "imputar" un
    código de centro o de país vía regresión sobre índices de contexto. En
    PISA 2025, un 2.45% de estudiantes tienen `CNTSCHID` nulo en el propio
    fichero de origen (posible enmascarado de privacidad), lo que hace fallar
    el merge con el fichero de centros — `STAFFSHORT`/`EDUSHORT` quedan nulos
    exactamente para esas mismas filas. Incluir `CNTSCHID` (escala arbitraria
    de miles de valores únicos) como columna a imputar agravaba muchísimo la
    colinealidad (`rcond` ~1e-16, matriz prácticamente singular) sin aportar
    nada: aunque se "rellenase" un CNTSCHID, no recupera los datos de centro
    ya perdidos en el merge.
    """
    df = df.copy()

    columnas_pv_todas = sum((columnas_pv(d) for d in DOMINIOS), [])
    columnas_pv_presentes = [c for c in columnas_pv_todas if c in df.columns]
    if columnas_pv_presentes:
        filas_sin_pv = df[columnas_pv_presentes].isnull().any(axis=1)
        if filas_sin_pv.any():
            print(f'  Se eliminan {filas_sin_pv.sum()} fila(s) sin ningún valor '
                  f'plausible (no realizaron la prueba); no se imputan.')
            df = df[~filas_sin_pv].reset_index(drop=True)

    excluir = (
        set(columnas_pv_presentes) | set(COLUMNAS_BASICAS) | set(COLUMNAS_ID_ALUMNO)
        | set(excluir_de_imputacion or [])
    )

    # Solo se imputan columnas con algún nulo Y algún valor real. Se decide tras
    # eliminar las filas sin PV: una columna cuyos nulos estaban solo en esas
    # filas ya no tiene nada que imputar, y una columna entera nula (módulo no
    # administrado ese año) no se puede imputar ni la conserva IterativeImputer,
    # lo que descuadraba los nombres de las indicadoras `_missing`.
    candidatas = df.columns if columnas_a_imputar is None else columnas_a_imputar
    columnas_a_imputar = [
        c for c in candidatas
        if c in df.columns and c not in excluir
        and df[c].isnull().any() and df[c].notna().any()
    ]

    if not columnas_a_imputar:
        return df

    imputador = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=max_iter,
        random_state=seed,
        add_indicator=True,
    )

    array_imputado = imputador.fit_transform(df[columnas_a_imputar])

    n_indicadoras = array_imputado.shape[1] - len(columnas_a_imputar)
    cols_indicadoras = [f'{col}_missing' for col in columnas_a_imputar][:n_indicadoras]

    df_imputado = pd.DataFrame(
        array_imputado,
        columns=columnas_a_imputar + cols_indicadoras,
        index=df.index,
    )

    for col in columnas_a_imputar:
        df[col] = df_imputado[col]

    for col in cols_indicadoras:
        df[col] = df_imputado[col].astype(bool)

    assert df[columnas_a_imputar].isnull().sum().sum() == 0, (
        'Error: todavía quedan nulos tras la imputación'
    )

    return df



# 5. Agregación por país

def agregar_por_pais(
    df: pd.DataFrame,
    indices: list[str],
    peso_col: str = 'W_FSTUWT',
    pais_col: str = 'CNT',
) -> pd.DataFrame:
    """Media ponderada de cada índice (ya calculado como columna) por país."""
    return df.groupby(pais_col).apply(
        lambda x: pd.Series({
            indice: np.average(x[indice], weights=x[peso_col])
            for indice in indices
        }),
        include_groups=False,
    )


def agregar_pv_por_pais(
    df: pd.DataFrame,
    dominio: str,
    peso_col: str = 'W_FSTUWT',
    pais_col: str = 'CNT',
) -> pd.Series:
    """Media ponderada de los PV1..PV10 de un dominio, agrupada por país.

    A diferencia de `agregar_por_pais` (que pondera un índice ya calculado por
    fila), aquí primero se promedian los 10 valores plausibles del dominio por
    estudiante y después se pondera esa media por `peso_col`, agrupando por
    país. Devuelve una Serie vacía si el dominio no tiene columnas PV en `df`
    (p. ej. si se eliminaron antes por algún motivo).
    """
    cols_pv = [c for c in columnas_pv(dominio) if c in df.columns]
    if not cols_pv:
        return pd.Series(dtype=float, name=dominio)

    df_temp = pd.DataFrame({
        'media_pv': df[cols_pv].mean(axis=1),
        peso_col: df[peso_col],
        pais_col: df[pais_col],
    })

    return df_temp.groupby(pais_col).apply(
        lambda x: np.average(x['media_pv'], weights=x[peso_col]),
        include_groups=False,
    )



# 6. Orquestación completa por edición

def _normalizar_ids(df: pd.DataFrame) -> pd.DataFrame:
    """CNTSCHID y CNTSTUID son identificadores, no cantidades: se normalizan
    siempre a string (algunas ediciones los traen como int, PISA 2012 y 2025
    como string con ceros a la izquierda). Sin esto, concatenar ediciones con
    dtypes distintos para la misma columna rompe la escritura a parquet.
    OJO: usar el dtype nullable 'string' de pandas, no `.astype(str)` — este
    último convierte un nulo real en el string literal 'None'/'nan', lo que
    rompe silenciosamente `.isna()` para las filas sin identificador."""
    for id_col in ('CNTSCHID', 'CNTSTUID'):
        if id_col in df.columns:
            df[id_col] = df[id_col].astype('string')
    return df


# Columnas derivadas por el pipeline que acompañan a cualquier vista de análisis
# (los *_adj y factor_inflacion solo existen si se submuestreó)
_COLUMNAS_DERIVADAS_ESTUDIANTE = ['SENWT', 'factor_inflacion', 'W_FSTUWT_adj', 'SENWT_adj', 'EDICION', 'DOMINIO_PRINCIPAL']
_COLUMNAS_DERIVADAS_COLEGIO = ['factor_inflacion', 'EDICION']


def vista_estudiantes(df: pd.DataFrame) -> pd.DataFrame:
    """Vista de análisis de la tabla de estudiantes: identificadores, pesos,
    PV de los tres dominios y `COLUMNAS_CONTEXTO_ESTUDIANTE` (más las
    indicadoras `<col>_missing` si ya se imputó). Las tablas guardadas en
    parquet conservan todas las columnas originales de PISA; esta vista solo
    acota lo que usan el EDA y las agregaciones de este pipeline."""
    pv = [c for c in sum((columnas_pv(d) for d in DOMINIOS), []) if c in df.columns]
    base = COLUMNAS_BASICAS + COLUMNAS_ID_ALUMNO + _COLUMNAS_DERIVADAS_ESTUDIANTE + pv + COLUMNAS_CONTEXTO_ESTUDIANTE
    indicadoras = [f'{c}_missing' for c in COLUMNAS_CONTEXTO_ESTUDIANTE if f'{c}_missing' in df.columns]
    cols = [c for c in dict.fromkeys(base + indicadoras) if c in df.columns]
    return df[cols].copy()


def vista_colegios(df: pd.DataFrame) -> pd.DataFrame:
    """Vista de análisis de la tabla de colegios (ver `vista_estudiantes`)."""
    base = ['CNT', 'STRATUM', 'CNTSCHID'] + COLUMNAS_PESO_COLEGIO + [f'{c}_adj' for c in COLUMNAS_PESO_COLEGIO]         + _COLUMNAS_DERIVADAS_COLEGIO + COLUMNAS_CONTEXTO_COLEGIO
    indicadoras = [f'{c}_missing' for c in COLUMNAS_CONTEXTO_COLEGIO if f'{c}_missing' in df.columns]
    cols = [c for c in dict.fromkeys(base + indicadoras) if c in df.columns]
    return df[cols].copy()


def _objetos_a_string(df: pd.DataFrame) -> pd.DataFrame:
    """Las columnas de tipo object (texto, o texto mezclado con números en los
    ficheros originales) se pasan al dtype nullable 'string' para que pyarrow
    pueda escribirlas a parquet sin fallar por tipos mezclados."""
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].map(lambda x: x if pd.isna(x) else str(x)).astype('string')
    return df


def rutas_datos_anuales(
    ruta_intermedios: str, año: int, imputados: bool = False
) -> tuple[str, str]:
    """Rutas de los dos parquets anuales (estudiantes, colegios) de una edición:
    `<ruta_intermedios>/datos_anuales/` (sin imputar) o, con `imputados=True`,
    `<ruta_intermedios>/datos_anuales_imputados/`."""
    carpeta = os.path.join(
        ruta_intermedios,
        CARPETA_DATOS_ANUALES_IMPUTADOS if imputados else CARPETA_DATOS_ANUALES,
    )
    return (
        os.path.join(carpeta, f'pisa{año}_estudiantes.parquet'),
        os.path.join(carpeta, f'pisa{año}_colegios.parquet'),
    )


def procesar_edicion(
    año: int,
    ruta_raw: str,
    ruta_intermedios: str,
    fraccion: float | None = None,
    seed: int = 42,
    chunk_size: int = 10000,
    paises: list[str] | None = PAISES_OCDE_TODAS_LAS_EDICIONES,
    columnas_estudiante: list[str] | None = None,
    columnas_colegio: list[str] | None = None,
    guardar_checkpoint: bool = True,
    usar_checkpoint_si_existe: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline de una edición PISA. Devuelve `(df_estudiantes, df_colegios)`:
    dos tablas separadas, NO unidas, que se guardan en
    `<ruta_intermedios>/datos_anuales/` como `pisa<año>_estudiantes.parquet` y
    `pisa<año>_colegios.parquet` (sin imputar: los nulos siguen ahí).

    1. Filtro de países (`paises`, por defecto los miembros OCDE de todas las
       ediciones; `None` conserva todos).
    2. Muestra: por defecto (`fraccion=None`) se conservan **todas** las
       escuelas y alumnos de esos países con los pesos originales de la OCDE
       (`W_FSTUWT`, `SENWT`, `W_SCHGRNRABWT`); `SENWT` se calcula en 2012, que no
       lo trae. Con `fraccion` entre 0 y 1 se hace submuestreo estratificado de
       escuelas y se añaden `factor_inflacion`, `W_FSTUWT_adj`, `SENWT_adj` y
       `W_SCHGRNRABWT_adj`.
    3. Estudiantes: lectura por bloques; colegios: una fila por escuela.
    4. Optimización de tipos y normalización de identificadores.
    5. Etiquetado de edición (y dominio principal del ciclo en estudiantes).

    Por defecto se conservan TODAS las columnas de los ficheros originales
    (`columnas_estudiante=None`, `columnas_colegio=None`), con los nombres que
    usa el pipeline para las que una edición llame distinto (`alias_*` de
    `EDICIONES_PISA`); el resto de columnas mantiene su nombre original. Las
    columnas de contexto que se analizan (`COLUMNAS_CONTEXTO_*`) se
    comprueban contra el fichero y se avisa de las que falten. Pasando una
    lista se restringe la lectura a esas columnas (más las imprescindibles).

    La tabla de estudiantes incluye las puntuaciones (PV1..PV10, o PV1..PV5 en
    2012) de los TRES dominios (MATH, READ, SCIE). Ambas tablas comparten `CNT`,
    `STRATUM` y `CNTSCHID` por si se quieren unir más adelante
    (`unir_alumnos_colegios`). Los pesos replicados (W_FSTURWT1..80; W_FSTR1..80
    en 2012) son los originales: solo valen tal cual con la muestra completa.

    Si `usar_checkpoint_si_existe` es True y ya existen los dos parquets de esta
    edición, se cargan directamente y se evita releer/reprocesar los ficheros de
    origen (que es el paso costoso). Pasa `usar_checkpoint_si_existe=False` para
    forzar el reprocesamiento (necesario si cambias `paises` o `fraccion`).
    """
    if año not in EDICIONES_PISA:
        raise ValueError(f"Edición {año} no configurada en EDICIONES_PISA")

    ruta_est, ruta_col = rutas_datos_anuales(ruta_intermedios, año)
    if usar_checkpoint_si_existe and os.path.exists(ruta_est) and os.path.exists(ruta_col):
        print(f'  Parquets anuales ya existen, se cargan sin reprocesar el origen: {os.path.dirname(ruta_est)}')
        df_estudiantes = _normalizar_ids(pd.read_parquet(ruta_est))
        df_colegios = _normalizar_ids(pd.read_parquet(ruta_col))
        if paises is not None:
            df_estudiantes = df_estudiantes[df_estudiantes['CNT'].isin(paises)].reset_index(drop=True)
            df_colegios = df_colegios[df_colegios['CNT'].isin(paises)].reset_index(drop=True)
        return df_estudiantes, df_colegios

    cfg = EDICIONES_PISA[año]
    alias_est = cfg.get('alias_estudiantes')
    alias_col = cfg.get('alias_colegios')

    ruta_estudiantes = encontrar_archivo(ruta_raw, cfg['patron_estudiantes'])
    ruta_colegios = encontrar_archivo(ruta_raw, cfg['patron_colegios'])

    # 1. Filtro de países y submuestreo de escuelas
    submuestrear = fraccion is not None and fraccion < 1
    escuelas_unicas = leer_escuelas_unicas(ruta_estudiantes, COLUMNAS_BASICAS, alias_est, paises)
    if paises is not None:
        faltan = sorted(set(paises) - set(escuelas_unicas['CNT'].unique()))
        if faltan:
            print(f'  Aviso PISA {año}: países pedidos sin datos en esta edición: {faltan}')
    if submuestrear:
        escuelas_sel, factores, _ = submuestreo_estratificado(escuelas_unicas, fraccion, seed)
    else:
        escuelas_sel, factores = None, None

    # 2. Estudiantes. No todas las ediciones administran todos los módulos de
    # contexto: se comprueba contra la metadata real del fichero y se avisa de
    # lo que falte en vez de dejar que pyreadstat reviente con un usecols
    # inexistente.
    n_pv = cfg.get('n_pv', 10)
    columnas_pv_totales = sum((columnas_pv(d, n_pv) for d in DOMINIOS), [])
    imprescindibles_est = COLUMNAS_BASICAS + COLUMNAS_ID_ALUMNO + columnas_pv_totales
    _, ausentes_est = columnas_disponibles(
        ruta_estudiantes,
        list(dict.fromkeys(imprescindibles_est + COLUMNAS_CONTEXTO_ESTUDIANTE)),
        alias_est,
    )
    basicas_ausentes = [c for c in COLUMNAS_BASICAS if c in ausentes_est]
    if basicas_ausentes:
        raise ValueError(
            f"PISA {año}: faltan columnas básicas imprescindibles en {ruta_estudiantes}: "
            f"{basicas_ausentes}"
        )
    if ausentes_est:
        print(f'  Aviso PISA {año}: columnas no encontradas en el fichero de '
              f'estudiantes: {ausentes_est}')

    if columnas_estudiante is None:
        columnas_a_leer_est = None  # todas
    else:
        deseadas = list(dict.fromkeys(imprescindibles_est + columnas_estudiante))
        columnas_a_leer_est = [c for c in deseadas if c not in ausentes_est]

    df_estudiantes = leer_estudiantes_submuestra(
        ruta_estudiantes, columnas_a_leer_est, escuelas_sel, factores, chunk_size, alias_est, paises
    )
    if submuestrear:
        df_estudiantes = calcular_senwt_adj(df_estudiantes)
    elif 'SENWT' not in df_estudiantes.columns:
        df_estudiantes = calcular_senwt(df_estudiantes)
    df_estudiantes = optimizar_tipos(df_estudiantes)
    df_estudiantes = _normalizar_ids(df_estudiantes)
    df_estudiantes = _objetos_a_string(df_estudiantes)
    # assign (en vez de asignar columna a columna) evita la fragmentación del
    # DataFrame, que con cientos de columnas dispara PerformanceWarning
    df_estudiantes = df_estudiantes.assign(EDICION=año, DOMINIO_PRINCIPAL=cfg['dominio_principal'])

    # 3. Colegios: solo las escuelas seleccionadas, una fila por escuela
    claves_col = ['CNT', 'STRATUM', 'CNTSCHID']
    _, ausentes_col = columnas_disponibles(
        ruta_colegios,
        list(dict.fromkeys(claves_col + COLUMNAS_PESO_COLEGIO + COLUMNAS_CONTEXTO_COLEGIO)),
        alias_col,
    )
    claves_ausentes = [c for c in claves_col if c in ausentes_col]
    if claves_ausentes:
        raise ValueError(
            f"PISA {año}: faltan columnas clave imprescindibles en {ruta_colegios}: {claves_ausentes}"
        )
    if ausentes_col:
        print(f'  Aviso PISA {año}: columnas no encontradas en el fichero de '
              f'centros: {ausentes_col}')

    if columnas_colegio is None:
        columnas_a_leer_col = None  # todas
    else:
        deseadas = list(dict.fromkeys(claves_col + COLUMNAS_PESO_COLEGIO + columnas_colegio))
        columnas_a_leer_col = [c for c in deseadas if c not in ausentes_col]
    df_colegios = leer_colegios(ruta_colegios, columnas_a_leer_col, alias_col)
    if df_colegios.columns.duplicated().any():
        raise ValueError(f'Columnas duplicadas tras aplicar los alias en centros: '
                         f'{df_colegios.columns[df_colegios.columns.duplicated()].tolist()}')

    if submuestrear:
        clave = pd.Series(
            list(zip(df_colegios['CNT'], df_colegios['STRATUM'], df_colegios['CNTSCHID'])),
            index=df_colegios.index,
        )
        df_colegios = df_colegios[clave.isin(escuelas_sel)].copy()
        df_colegios['factor_inflacion'] = pd.Series(
            list(zip(df_colegios['CNT'], df_colegios['STRATUM'])), index=df_colegios.index
        ).map(factores)
        for peso in COLUMNAS_PESO_COLEGIO:
            if peso in df_colegios.columns:
                df_colegios[f'{peso}_adj'] = df_colegios[peso] * df_colegios['factor_inflacion']
    elif paises is not None:
        df_colegios = df_colegios[df_colegios['CNT'].isin(paises)].copy()
    df_colegios = optimizar_tipos(df_colegios.reset_index(drop=True))
    df_colegios = _normalizar_ids(df_colegios)
    df_colegios = _objetos_a_string(df_colegios)
    df_colegios = df_colegios.assign(EDICION=año)

    # 4. Guardado: dos parquets por año en datos_anuales/ (permite re-arrancar
    # sin releer el origen)
    if guardar_checkpoint:
        os.makedirs(os.path.dirname(ruta_est), exist_ok=True)
        df_estudiantes.to_parquet(ruta_est, engine='pyarrow', index=False)
        df_colegios.to_parquet(ruta_col, engine='pyarrow', index=False)

    del escuelas_unicas
    gc.collect()

    return df_estudiantes, df_colegios
