"""
pisa_pipeline.py
=================

Procesamiento de microdatos PISA 2012+ (dominios MATH, READ, SCIE).

Flujo (ver `notebooks/fase_1_data_ingestion_eda/00_pipeline_multiedicion_PISA.ipynb`):

    1. `procesar_edicion`: lee estudiantes y centros de una edición, filtra países y
       guarda 2 parquets sin imputar y con todas las columnas en `datos_anuales/`,
       más las columnas `*_TREND` de `escs_trend.csv`.
    2. `leer_vista_estudiantes` / `leer_vista_colegios`: leen del parquet solo las
       columnas curadas, para EDA.
    3. `agregar_pv_por_pais` / `agregar_por_pais`: medias ponderadas por país.
    4. `validar_tablas` y `generar_diccionario_columnas`: control de calidad y
       documentación de las tablas.

Estudiantes y centros son siempre tablas separadas (comparten CNT, STRATUM, CNTSCHID).

⚠️ Las ediciones difieren: formato (2012 solo en .txt, convertible con
`pisa_txt_a_sav.py`; 2025 solo .sav), nombres de columna (`alias_*` en
`EDICIONES_PISA`), nº de valores plausibles (5 en 2012) y módulos de contexto
administrados. Los índices de contexto no son necesariamente comparables entre
ciclos; solo ESCS, HOMEPOS, HISEI y PAREDINT tienen versión reescalada
(`escs_trend`, 2012-2018, integradas como columnas `*_TREND`).
"""

from __future__ import annotations

import os
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyreadstat


# 0. Configuración

DOMINIOS = ['MATH', 'READ', 'SCIE']

# Por edición: dominio principal del ciclo; 'patron_*' = subcadena del nombre de
# fichero (sin distinguir mayúsculas ni extensión); 'n_pv' = valores plausibles
# (10 desde 2015, 5 antes); 'alias_*' = {nombre_canónico: nombre_en_el_fichero}.
EDICIONES_PISA = {
    2012: {  # solo existe como .txt + .sas: convertir antes con `pisa_txt_a_sav.py`
        'dominio_principal': 'MATH',
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
        'alias_estudiantes': {'BEINGBULLIED': 'BULLIED'},
    },
    2025: {
        'dominio_principal': 'SCIE',
        'patron_estudiantes': 'stu_qqq',
        'patron_colegios': 'sch_qqq',
    },
}

COLUMNAS_BASICAS = ['CNT', 'CNTSCHID', 'STRATUM', 'W_FSTUWT']

# Id de alumno: sirve para unir con ficheros externos (`escs_trend`).
COLUMNAS_ID_ALUMNO = ['CNTSTUID']

# Peso del centro (en 2012 el fichero lo llama W_FSCHWT; ver alias).
COLUMNAS_PESO_COLEGIO = ['W_SCHGRNRABWT']

CARPETA_DATOS_ANUALES = 'datos_anuales'   # tablas maestras sin imputar

# Miembros de la OCDE en todas las ediciones 2012-2022 y con datos en todas (33).
# Se excluyen LVA, LTU, COL y CRI (entraron después de 2012) y LUX (miembro, pero
# sin datos en PISA 2022).
PAISES_OCDE_TODAS_LAS_EDICIONES = [
    'AUS', 'AUT', 'BEL', 'CAN', 'CHE', 'CHL', 'CZE', 'DEU', 'DNK', 'ESP', 'EST',
    'FIN', 'FRA', 'GBR', 'GRC', 'HUN', 'IRL', 'ISL', 'ISR', 'ITA', 'JPN', 'KOR',
    'MEX', 'NLD', 'NOR', 'NZL', 'POL', 'PRT', 'SVK', 'SVN', 'SWE', 'TUR', 'USA',
]

# Índices reescalados por la OCDE (`escs_trend.csv`, 2012-2018, comparables con
# 2022). {columna original: columna del csv}. Se añaden como `<col>_TREND`.
INDICES_TREND = {
    'ESCS': 'escs_trend', 'HISEI': 'hisei_trend',
    'HOMEPOS': 'homepos_trend', 'PAREDINT': 'paredint_trend',
}
COLUMNAS_TREND = [f'{c}_TREND' for c in INDICES_TREND]
CICLO_ESCS_TREND = {2012: 5, 2015: 6, 2018: 7}          # ciclo de cada edición en el csv
_MODULO_IDS_ESCS_TREND = {2012: None, 2015: 100000, 2018: 100000}  # 2015/2018: sin prefijo de país

# Contexto del estudiante que se analiza (no todo existe en todas las ediciones).
COLUMNAS_CONTEXTO_ESTUDIANTE = [
    'ESCS', 'HISEI', 'PAREDINT', 'HOMEPOS', 'WEALTH',
    'CULTPOSS', 'HEDRES', 'ICTRES',
    'ANXMAT', 'MATHEFF', 'MATHINT', 'MATHBEH',
    'BELONG',
    'BEINGBULLIED',  # acoso sufrido; solo 2018 (BEINGBULLIED) y 2022 (BULLIED, con alias)
    'DISCLIMA', 'TEACHSUP',
    'OUTHOURS',  # estudio fuera del horario escolar (2012 y 2015)
    'ICTWKDY', 'ICTWKEND', 'ICTDISTR',  # uso TIC y malestar online (2022)
    'SKIPPING', 'TARDYSD',  # absentismo e impuntualidad (2022)
    'ESCS_TREND', 'HISEI_TREND', 'HOMEPOS_TREND', 'PAREDINT_TREND',  # añadidas por el pipeline
]

# Contexto del centro. SCHLTYPE: 1=privado independiente, 2=privado concertado, 3=público.
COLUMNAS_CONTEXTO_COLEGIO = [
    'STRATIO', 'PROPMATH', 'SCHLCLI', 'EDUSHORT', 'STAFFSHORT', 'SCHLTYPE', 'CLSIZE',
]


def columnas_pv(dominio: str, n_pv: int = 10) -> list[str]:
    """Nombres PV1..PV`n_pv` de un dominio. Las funciones que los usan filtran
    por los que existan, así que `n_pv=10` vale también para 2012 (5 PV)."""
    if dominio not in DOMINIOS:
        raise ValueError(f"Dominio '{dominio}' no reconocido. Usa uno de {DOMINIOS}")
    return [f'PV{i}{dominio}' for i in range(1, n_pv + 1)]


# pyreadstat tiene una función por formato con la misma firma
LECTORES_PYREADSTAT = {
    '.sas7bdat': pyreadstat.read_sas7bdat,
    '.sav': pyreadstat.read_sav,
}


def lector_para(ruta_archivo: str):
    """Función de pyreadstat adecuada a la extensión de `ruta_archivo`."""
    extension = os.path.splitext(ruta_archivo)[1].lower()
    if extension not in LECTORES_PYREADSTAT:
        raise ValueError(
            f"Formato '{extension}' no soportado ({ruta_archivo}). "
            f"Formatos soportados: {list(LECTORES_PYREADSTAT)}"
        )
    return LECTORES_PYREADSTAT[extension]


def encontrar_archivo(ruta_carpeta: str, patron: str, extensiones: tuple[str, ...] | None = None) -> str:
    """Único archivo de `ruta_carpeta` cuyo nombre contiene `patron` (sin distinguir
    mayúsculas) y tiene una extensión soportada. Falla si no hay ninguno o hay varios."""
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
    """(presentes, ausentes) de `columnas` (nombres canónicos) en el fichero.
    Solo lee metadatos. `alias` traduce los nombres que la edición llama distinto."""
    alias = alias or {}
    _, meta = lector_para(ruta_archivo)(ruta_archivo, metadataonly=True)
    columnas_reales = set(meta.column_names)
    presentes = [c for c in columnas if alias.get(c, c) in columnas_reales]
    ausentes = [c for c in columnas if alias.get(c, c) not in columnas_reales]
    return presentes, ausentes


def _traducir_columnas(
    columnas: list[str] | None, alias: dict[str, str] | None
) -> tuple[list[str] | None, dict[str, str]]:
    """(nombres a leer del fichero, renombrado fichero->canónico). Con
    `columnas=None` se leen todas y se renombran todas las que tengan alias."""
    alias = alias or {}
    if columnas is None:
        return None, {v: k for k, v in alias.items() if v != k}
    a_leer = [alias.get(c, c) for c in columnas]
    renombrar = {alias[c]: c for c in columnas if c in alias and alias[c] != c}
    return a_leer, renombrar


# 1. Submuestreo estratificado (opcional)

def leer_escuelas_unicas(
    ruta_estudiantes: str,
    cols_basicas: list[str],
    alias: dict[str, str] | None = None,
    paises: list[str] | None = None,
) -> pd.DataFrame:
    """Una fila por escuela (CNT, CNTSCHID, STRATUM), opcionalmente solo de `paises`."""
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
    """Elige `fraccion` de las escuelas de cada estrato (CNT + STRATUM), mínimo 2
    si el estrato tiene 2 o más. Devuelve (escuelas elegidas como tuplas
    (CNT, STRATUM, CNTSCHID), {(CNT, STRATUM): factor_inflacion}, resumen por estrato)."""
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
        factores_dict[(pais, estrato)] = datos_estrato['factor_inflacion']

        muestra = grupo.sample(n=datos_estrato['n_submuestra'], random_state=seed)
        for colegio_id in muestra['CNTSCHID'].values:
            escuelas_seleccionadas.add((pais, estrato, colegio_id))

    return escuelas_seleccionadas, factores_dict, df_groups


# 2. Lectura de estudiantes y centros

def leer_estudiantes_submuestra(
    ruta_estudiantes: str,
    columnas: list[str] | None,
    escuelas_seleccionadas: set | None = None,
    factores_dict: dict | None = None,
    chunk_size: int = 10000,
    alias: dict[str, str] | None = None,
    paises: list[str] | None = None,
) -> pd.DataFrame:
    """Lee los estudiantes por bloques (`columnas=None`: todas, con nombres canónicos).

    Con `escuelas_seleccionadas` y `factores_dict` conserva solo esas escuelas y
    añade `factor_inflacion` y `W_FSTUWT_adj`. Sin ellos conserva todos los
    alumnos de `paises` (o de todos) con los pesos originales.
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
    """Añade el peso "senado": `peso_col` reescalado para que cada país sume
    `total_senado` (convención OCDE). Los ficheros de 2015+ ya traen `SENWT`; 2012 no."""
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
    """`calcular_senwt` sobre `W_FSTUWT_adj` (tras un submuestreo), columna `SENWT_adj`."""
    return calcular_senwt(df_alumnos, peso_col, 'SENWT_adj', pais_col, total_senado)


def leer_colegios(
    ruta_colegios: str,
    columnas: list[str] | None,
    alias: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Lee el fichero de centros (`columnas=None`: todas) con nombres canónicos."""
    a_leer, renombrar = _traducir_columnas(columnas, alias)
    df_colegios, _ = lector_para(ruta_colegios)(ruta_colegios, usecols=a_leer)
    return df_colegios.rename(columns=renombrar)


def unir_alumnos_colegios(
    df_alumnos: pd.DataFrame,
    df_colegios: pd.DataFrame,
    claves=('CNT', 'STRATUM', 'CNTSCHID'),
) -> pd.DataFrame:
    """Left join alumnos-centros (no se usa: las tablas se mantienen separadas)."""
    return pd.merge(df_alumnos, df_colegios, on=list(claves), how='left')


# 3. Nulos

def porcentaje_nulos(df: pd.DataFrame) -> pd.Series:
    """% de nulos por columna, de mayor a menor."""
    return (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)


def porcentaje_nulos_por_pais(df: pd.DataFrame, columna: str, pais_col: str = 'CNT') -> pd.DataFrame:
    """% de nulos de `columna` en cada país, de mayor a menor."""
    resultado = df[columna].isna().groupby(df[pais_col]).mean() * 100
    return (
        resultado.reset_index(name=f'pct_nulos_{columna}')
        .sort_values(f'pct_nulos_{columna}', ascending=False)
    )


# 4. Agregación por país

def agregar_por_pais(
    df: pd.DataFrame,
    indices: list[str],
    peso_col: str = 'W_FSTUWT',
    pais_col: str = 'CNT',
) -> pd.DataFrame:
    """Media ponderada por país de cada columna de `indices`."""
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
    """Media por país de los PV de un dominio: promedia los PV de cada alumno y
    pondera por `peso_col`. Serie vacía si `df` no tiene PV de ese dominio."""
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


# 5. Orquestación por edición

def _normalizar_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Ids de centro y alumno a dtype 'string' (vienen como número o como texto con
    ceros, que se conservan). Los numéricos pasan por Int64 para no quedar como
    '724.0'; no usar `astype(str)`: convertiría los nulos en el texto 'None'."""
    for id_col in ('CNTSCHID', 'CNTSTUID'):
        if id_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[id_col]):
                df[id_col] = df[id_col].astype('Int64')
            df[id_col] = df[id_col].astype('string')
    return df


# Columnas añadidas por el pipeline (las *_adj y factor_inflacion solo con submuestreo)
_COLUMNAS_DERIVADAS_ESTUDIANTE = ['SENWT', 'factor_inflacion', 'W_FSTUWT_adj', 'SENWT_adj', 'EDICION', 'DOMINIO_PRINCIPAL']
_COLUMNAS_DERIVADAS_COLEGIO = ['factor_inflacion', 'EDICION']


def _columnas_vista_estudiantes(nombres) -> list[str]:
    pv = [c for c in sum((columnas_pv(d) for d in DOMINIOS), []) if c in nombres]
    base = COLUMNAS_BASICAS + COLUMNAS_ID_ALUMNO + _COLUMNAS_DERIVADAS_ESTUDIANTE + pv + COLUMNAS_CONTEXTO_ESTUDIANTE
    return [c for c in dict.fromkeys(base) if c in nombres]


def _columnas_vista_colegios(nombres) -> list[str]:
    base = (['CNT', 'STRATUM', 'CNTSCHID'] + COLUMNAS_PESO_COLEGIO
            + [f'{c}_adj' for c in COLUMNAS_PESO_COLEGIO]
            + _COLUMNAS_DERIVADAS_COLEGIO + COLUMNAS_CONTEXTO_COLEGIO)
    return [c for c in dict.fromkeys(base) if c in nombres]


def vista_estudiantes(df: pd.DataFrame) -> pd.DataFrame:
    """Copia de `df` con solo ids, pesos, PV y `COLUMNAS_CONTEXTO_ESTUDIANTE`. Los
    parquets guardan todas las columnas; esto acota lo que usa el EDA."""
    return df[_columnas_vista_estudiantes(set(df.columns))].copy()


def vista_colegios(df: pd.DataFrame) -> pd.DataFrame:
    """Como `vista_estudiantes`, para la tabla de centros."""
    return df[_columnas_vista_colegios(set(df.columns))].copy()


def leer_vista_estudiantes(ruta_parquet: str) -> pd.DataFrame:
    """`vista_estudiantes` leyendo del parquet solo esas columnas (sin cargar todas)."""
    nombres = pq.ParquetFile(ruta_parquet).schema.names
    return pd.read_parquet(ruta_parquet, columns=_columnas_vista_estudiantes(set(nombres)))


def leer_vista_colegios(ruta_parquet: str) -> pd.DataFrame:
    """`vista_colegios` leyendo del parquet solo esas columnas."""
    nombres = pq.ParquetFile(ruta_parquet).schema.names
    return pd.read_parquet(ruta_parquet, columns=_columnas_vista_colegios(set(nombres)))


def añadir_indices_trend(df: pd.DataFrame, año: int, ruta_csv: str) -> pd.DataFrame:
    """Añade `ESCS_TREND`, `HISEI_TREND`, `HOMEPOS_TREND` y `PAREDINT_TREND` desde
    `escs_trend.csv` de la OCDE, uniendo por país, centro y alumno. Los índices
    originales no se tocan. 2022 no está en el csv (su escala es la de referencia):
    sus columnas `_TREND` son copia de las originales. Falla si casa <99,9%."""
    if año not in CICLO_ESCS_TREND:
        return df.assign(**{f'{c}_TREND': df[c] for c in INDICES_TREND if c in df.columns})

    modulo = _MODULO_IDS_ESCS_TREND[año]

    def clave(serie):
        x = pd.to_numeric(serie, errors='coerce').astype('Int64')
        return x % modulo if modulo else x

    t = pd.read_csv(
        ruta_csv, usecols=['cycle', 'cnt', 'schoolid', 'studentid', *INDICES_TREND.values()],
        dtype={'schoolid': str, 'studentid': str},
    )
    t = t[t['cycle'] == CICLO_ESCS_TREND[año]]
    t = t.assign(
        CNT=t['cnt'], _s=clave(t['schoolid']), _st=clave(t['studentid'])
    ).drop(columns=['cycle', 'cnt', 'schoolid', 'studentid'])

    k = pd.DataFrame({'CNT': df['CNT'], '_s': clave(df['CNTSCHID']), '_st': clave(df['CNTSTUID'])})
    m = k.merge(t, how='left', on=['CNT', '_s', '_st'], validate='many_to_one', indicator=True)
    assert len(m) == len(df), 'El merge con escs_trend ha cambiado el número de filas'
    casan = (m['_merge'] == 'both').mean()
    if casan < 0.999:
        raise ValueError(f'PISA {año}: solo casa el {casan:.2%} de los alumnos con escs_trend.csv')
    print(f'  escs_trend: casa el {casan:.2%} de los alumnos')
    return df.assign(**{f'{c}_TREND': m[v].to_numpy() for c, v in INDICES_TREND.items()})


def _objetos_a_string(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas object a dtype 'string' (pyarrow falla con tipos mezclados)."""
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].map(lambda x: x if pd.isna(x) else str(x)).astype('string')
    return df


def rutas_datos_anuales(ruta_intermedios: str, año: int) -> tuple[str, str]:
    """Rutas (estudiantes, colegios) de los parquets de una edición en `datos_anuales/`."""
    carpeta = os.path.join(ruta_intermedios, CARPETA_DATOS_ANUALES)
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
    ruta_escs_trend: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Procesa una edición y devuelve `(df_estudiantes, df_colegios)`, sin imputar.
    Los guarda en `datos_anuales/` (`pisa<año>_estudiantes/colegios.parquet`).

    * `paises`: países a conservar (None = todos).
    * `fraccion=None`: muestra completa con pesos originales (`SENWT` se calcula en
      2012). Con `fraccion` en (0, 1): submuestreo de escuelas por estrato y pesos
      ajustados `*_adj`.
    * `columnas_*=None`: todas las columnas originales, con los nombres canónicos
      de `alias_*`. Una lista restringe la lectura a esas (más las imprescindibles).
    * Avisa de las columnas de `COLUMNAS_CONTEXTO_*` que el fichero no tiene.
    * Con `ruta_escs_trend`, añade las columnas `*_TREND` (ver `añadir_indices_trend`).
    * Con `usar_checkpoint_si_existe`, si los parquets ya existen se cargan sin
      releer el origen (ponlo a False si cambias `paises` o `fraccion`).
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

    # 1. Países y (opcional) submuestreo de escuelas
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

    # 2. Estudiantes (se comprueban antes las columnas contra los metadatos del fichero)
    n_pv = cfg.get('n_pv', 10)
    columnas_pv_totales = sum((columnas_pv(d, n_pv) for d in DOMINIOS), [])
    imprescindibles_est = COLUMNAS_BASICAS + COLUMNAS_ID_ALUMNO + columnas_pv_totales
    _, ausentes_est = columnas_disponibles(
        ruta_estudiantes,
        list(dict.fromkeys(
            imprescindibles_est
            + [c for c in COLUMNAS_CONTEXTO_ESTUDIANTE if c not in COLUMNAS_TREND]
        )),
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
    df_estudiantes = _normalizar_ids(df_estudiantes)
    df_estudiantes = _objetos_a_string(df_estudiantes)
    if ruta_escs_trend:
        df_estudiantes = añadir_indices_trend(df_estudiantes, año, ruta_escs_trend)
    # assign evita la fragmentación del DataFrame (PerformanceWarning con cientos de columnas)
    df_estudiantes = df_estudiantes.assign(EDICION=año, DOMINIO_PRINCIPAL=cfg['dominio_principal'])

    # 3. Colegios
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
    df_colegios = df_colegios.reset_index(drop=True)
    df_colegios = _normalizar_ids(df_colegios)
    df_colegios = _objetos_a_string(df_colegios)
    df_colegios = df_colegios.assign(EDICION=año)

    # 4. Guardado (también sirve de punto de reinicio)
    if guardar_checkpoint:
        os.makedirs(os.path.dirname(ruta_est), exist_ok=True)
        df_estudiantes.to_parquet(ruta_est, engine='pyarrow', index=False)
        df_colegios.to_parquet(ruta_col, engine='pyarrow', index=False)

    del escuelas_unicas
    gc.collect()

    return df_estudiantes, df_colegios


# 6. Validación y documentación de las tablas maestras

def validar_tablas(
    df_est: pd.DataFrame,
    df_col: pd.DataFrame,
    paises: list[str] | None = None,
) -> pd.DataFrame:
    """Comprueba la integridad de las tablas de una edición y devuelve un resumen
    (valor, ok). Lanza AssertionError si falla alguna comprobación.

    Claves: alumno = (CNT, CNTSTUID) (en 2012 el id solo es único dentro del país);
    centro = (CNT, CNTSCHID).
    """
    pv = [c for c in sum((columnas_pv(d) for d in DOMINIOS), []) if c in df_est.columns]
    idx_est = pd.MultiIndex.from_frame(df_est[['CNT', 'CNTSCHID']])
    idx_col = pd.MultiIndex.from_frame(df_col[['CNT', 'CNTSCHID']])
    desvio_senwt = (df_est.groupby('CNT')['SENWT'].sum() - 5000).abs().max()

    comprobaciones = {
        'países distintos a los pedidos': (
            0 if paises is None else len(set(df_est['CNT']) ^ set(paises))
        ),
        'alumnos duplicados (CNT, CNTSTUID)': int(df_est.duplicated(['CNT', 'CNTSTUID']).sum()),
        'centros duplicados (CNT, CNTSCHID)': int(df_col.duplicated(['CNT', 'CNTSCHID']).sum()),
        'ids nulos': int(df_est[['CNTSCHID', 'CNTSTUID']].isna().sum().sum() + df_col['CNTSCHID'].isna().sum()),
        'alumnos sin centro en la tabla de colegios': int((~idx_est.isin(idx_col)).sum()),
        'centros sin alumnos': int((~idx_col.isin(idx_est.unique())).sum()),
        'pesos de alumno nulos o <= 0': int((df_est['W_FSTUWT'].isna() | (df_est['W_FSTUWT'] <= 0)).sum()),
        'pesos de centro nulos o <= 0': int((df_col['W_SCHGRNRABWT'].isna() | (df_col['W_SCHGRNRABWT'] <= 0)).sum()),
        'valores plausibles nulos': int(df_est[pv].isna().sum().sum()),
        'desvío máximo de SENWT respecto a 5000 por país': float(desvio_senwt),
    }
    resumen = pd.DataFrame({'valor': comprobaciones})
    # SENWT: se tolera un desvío de redondeo de 0,01 sobre 5000
    tolerancia = pd.Series(1e-9, index=resumen.index)
    tolerancia['desvío máximo de SENWT respecto a 5000 por país'] = 0.01
    resumen['ok'] = resumen['valor'] <= tolerancia
    assert resumen['ok'].all(), f'Validación fallida:\n{resumen[~resumen["ok"]]}'
    return resumen


def generar_diccionario_columnas(
    ruta_raw_base: str, ediciones: list[int], ruta_salida: str | None = None
) -> pd.DataFrame:
    """Diccionario (EDICION, TABLA, COLUMNA, NOMBRE_ORIGINAL, ETIQUETA) de las
    columnas de los ficheros originales, con los nombres canónicos de los parquets.
    Solo lee metadatos. Incluye las columnas añadidas por el pipeline. Si se da
    `ruta_salida`, lo guarda como CSV."""
    filas = []
    for año in ediciones:
        cfg = EDICIONES_PISA[año]
        carpeta = os.path.join(ruta_raw_base, str(año))
        for tabla in ('estudiantes', 'colegios'):
            alias = cfg.get(f'alias_{tabla}') or {}
            inverso = {v: k for k, v in alias.items()}
            ruta = encontrar_archivo(carpeta, cfg[f'patron_{tabla}'])
            _, meta = lector_para(ruta)(ruta, metadataonly=True)
            for nombre, etiqueta in zip(meta.column_names, meta.column_labels):
                filas.append((año, tabla, inverso.get(nombre, nombre), nombre, etiqueta or ''))
            filas.append((año, tabla, 'EDICION', '', 'Añadida por el pipeline: año de la edición'))
        filas.append((año, 'estudiantes', 'DOMINIO_PRINCIPAL', '', 'Añadida por el pipeline: dominio focal del ciclo'))
        for c in INDICES_TREND:
            origen = 'copia del original (2022 es la escala de referencia)' if año not in CICLO_ESCS_TREND \
                else 'reescalado por la OCDE (escs_trend.csv), comparable con 2022'
            filas.append((año, 'estudiantes', f'{c}_TREND', '', f'Añadida por el pipeline: {c} {origen}'))
        if not any(f[0] == año and f[1] == 'estudiantes' and f[2] == 'SENWT' for f in filas):
            filas.append((año, 'estudiantes', 'SENWT', '',
                          'Añadida por el pipeline: peso senado (W_FSTUWT reescalado, cada país suma 5000)'))
    diccionario = pd.DataFrame(
        filas, columns=['EDICION', 'TABLA', 'COLUMNA', 'NOMBRE_ORIGINAL', 'ETIQUETA']
    )
    if ruta_salida:
        diccionario.to_csv(ruta_salida, index=False, encoding='utf-8-sig')
    return diccionario
