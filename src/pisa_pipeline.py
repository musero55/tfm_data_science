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
EDICIONES_PISA = {
    2015: {
        'dominio_principal': 'SCIE',
        'patron_estudiantes': 'stu_qqq',
        'patron_colegios': 'sch_qqq',
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

# Índices de contexto a nivel estudiante
COLUMNAS_CONTEXTO_ESTUDIANTE = [
    'ESCS', 'HISEI', 'PAREDINT', 'HOMEPOS', 'WEALTH',
    'CULTPOSS', 'HEDRES', 'ICTRES',
    'ANXMAT', 'MATHEFF', 'MATHINT', 'MATHBEH',
    'BELONG', 'BULLY', 'DISCLIMA', 'TEACHSUP',
    'OUTHOURS',  # tiempo de estudio fuera del horario escolar (solo 2015)
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


def columnas_pv(dominio: str) -> list[str]:
    """Nombres de los 10 valores plausibles (PV1..PV10) de un dominio."""
    if dominio not in DOMINIOS:
        raise ValueError(f"Dominio '{dominio}' no reconocido. Usa uno de {DOMINIOS}")
    return [f'PV{i}{dominio}' for i in range(1, 11)]


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


def columnas_disponibles(ruta_archivo: str, columnas: list[str]) -> tuple[list[str], list[str]]:
    """Compara `columnas` contra las columnas realmente presentes en `ruta_archivo`
    (lee solo metadata, no los datos). Devuelve (presentes, ausentes).

    Necesario porque no todas las ediciones administran todos los módulos de
    contexto (p. ej. ICT), así que una columna que existe en el fichero de un
    año puede no existir en el de otro.
    """
    _, meta = lector_para(ruta_archivo)(ruta_archivo, metadataonly=True)
    columnas_reales = set(meta.column_names)
    presentes = [c for c in columnas if c in columnas_reales]
    ausentes = [c for c in columnas if c not in columnas_reales]
    return presentes, ausentes


# 1. Selección de escuelas y submuestreo estratificado

def leer_escuelas_unicas(ruta_estudiantes: str, cols_basicas: list[str]) -> pd.DataFrame:
    """Lee las columnas identificativas básicas y devuelve una fila por escuela
    única (CNT, CNTSCHID, STRATUM), necesaria para diseñar el submuestreo."""
    df_basic, _ = lector_para(ruta_estudiantes)(ruta_estudiantes, usecols=cols_basicas)
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
    columnas: list[str],
    escuelas_seleccionadas: set,
    factores_dict: dict,
    chunk_size: int = 10000,
) -> pd.DataFrame:
    """Lee el fichero de estudiantes por bloques, conserva solo las escuelas
    seleccionadas en el submuestreo y ajusta W_FSTUWT según el factor de
    inflación del estrato correspondiente."""
    reader = pyreadstat.read_file_in_chunks(
        lector_para(ruta_estudiantes),
        ruta_estudiantes,
        chunksize=chunk_size,
        usecols=columnas,
    )

    chunks_filtrados = []
    for chunk, _ in reader:
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
            "No se ha encontrado ningún estudiante de las escuelas seleccionadas. "
            "Revisa el fichero de origen y el set de escuelas."
        )

    return pd.concat(chunks_filtrados, ignore_index=True)


def calcular_senwt_adj(
    df_alumnos: pd.DataFrame,
    peso_col: str = 'W_FSTUWT_adj',
    pais_col: str = 'CNT',
    total_senado: int = 5000,
) -> pd.DataFrame:
    """Calcula el peso de senado ajustado: la suma de pesos por país tras el
    submuestreo vuelve a sumar `total_senado` (5.000 por defecto, según
    metodología OCDE)."""
    df_alumnos = df_alumnos.copy()
    suma_pesos_pais = df_alumnos.groupby(pais_col)[peso_col].transform('sum')
    df_alumnos['SENWT_adj'] = df_alumnos[peso_col] * (total_senado / suma_pesos_pais)
    return df_alumnos


def leer_colegios(ruta_colegios: str, columnas: list[str]) -> pd.DataFrame:
    df_colegios, _ = lector_para(ruta_colegios)(ruta_colegios, usecols=columnas)
    return df_colegios


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
    for col in df.select_dtypes(include=['float64', 'float32']).columns:
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

    excluir = set(columnas_pv_presentes) | set(COLUMNAS_BASICAS) | set(excluir_de_imputacion or [])

    if columnas_a_imputar is None:
        columnas_a_imputar = [
            c for c in df.columns if c not in excluir and df[c].isnull().any()
        ]
    else:
        columnas_a_imputar = [c for c in columnas_a_imputar if c not in excluir]

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
    peso_col: str = 'W_FSTUWT_adj',
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
    peso_col: str = 'W_FSTUWT_adj',
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

def procesar_edicion(
    año: int,
    ruta_raw: str,
    ruta_intermedios: str,
    fraccion: float = 0.20,
    seed: int = 42,
    chunk_size: int = 10000,
    columnas_contexto_estudiante: list[str] | None = None,
    columnas_contexto_colegio: list[str] | None = None,
    guardar_checkpoint: bool = True,
    usar_checkpoint_si_existe: bool = True,
) -> pd.DataFrame:
    """Pipeline completo de una edición PISA:
    1. Submuestreo estratificado de escuelas + factor de inflación.
    2. Lectura filtrada de estudiantes con ajuste de W_FSTUWT y SENWT.
    3. Lectura y unión con datos de centro.
    4. Optimización de tipos.
    5. Etiquetado de edición y dominio principal del ciclo.
    6. (Opcional) checkpoint en parquet antes de imputar nulos.

    Incluye las puntuaciones (PV1..PV10) de los TRES dominios (MATH, READ,
    SCIE), no solo del dominio principal del ciclo, para poder modelar
    cualquiera de los tres o compararlos.

    Si `usar_checkpoint_si_existe` es True y ya existe el parquet de checkpoint
    de esta edición en `ruta_intermedios`, se carga directamente y se evita
    releer/reprocesar los `.sas7bdat` (que es el paso costoso). Pasa
    `usar_checkpoint_si_existe=False` para forzar el reprocesamiento.
    """
    if año not in EDICIONES_PISA:
        raise ValueError(f"Edición {año} no configurada en EDICIONES_PISA")

    ruta_out = os.path.join(ruta_intermedios, f'pisa{año}_submuestra_global_nulos.parquet')
    if usar_checkpoint_si_existe and os.path.exists(ruta_out):
        print(f'  Checkpoint ya existe, se carga sin reprocesar el SAS: {ruta_out}')
        df_checkpoint = pd.read_parquet(ruta_out)
        df_checkpoint['CNTSCHID'] = df_checkpoint['CNTSCHID'].astype('string')
        return df_checkpoint

    cfg = EDICIONES_PISA[año]

    columnas_contexto_estudiante = columnas_contexto_estudiante or COLUMNAS_CONTEXTO_ESTUDIANTE
    columnas_contexto_colegio = columnas_contexto_colegio or COLUMNAS_CONTEXTO_COLEGIO

    ruta_estudiantes = encontrar_archivo(ruta_raw, cfg['patron_estudiantes'])
    ruta_colegios = encontrar_archivo(ruta_raw, cfg['patron_colegios'])

    # 1. Submuestreo de escuelas
    escuelas_unicas = leer_escuelas_unicas(ruta_estudiantes, COLUMNAS_BASICAS)
    escuelas_sel, factores, _ = submuestreo_estratificado(escuelas_unicas, fraccion, seed)

    # 2. Columnas de interés: IDs + PVs de los 3 dominios + contexto.
    # No todas las ediciones administran todos los módulos de contexto: se
    # comprueba contra la metadata real del SAS y se avisa de lo que falte en
    # vez de dejar que pyreadstat reviente con un usecols inexistente.
    columnas_pv_totales = sum((columnas_pv(d) for d in DOMINIOS), [])
    columnas_interes_deseadas = list(dict.fromkeys(
        COLUMNAS_BASICAS + columnas_pv_totales + columnas_contexto_estudiante
    ))
    columnas_interes, ausentes_est = columnas_disponibles(ruta_estudiantes, columnas_interes_deseadas)

    basicas_ausentes = [c for c in COLUMNAS_BASICAS if c in ausentes_est]
    if basicas_ausentes:
        raise ValueError(
            f"PISA {año}: faltan columnas básicas imprescindibles en {ruta_estudiantes}: "
            f"{basicas_ausentes}"
        )
    if ausentes_est:
        print(f'  Aviso PISA {año}: columnas de contexto no encontradas en el fichero de '
              f'estudiantes, se omiten: {ausentes_est}')

    # 3. Lectura filtrada + ajuste de pesos
    df_alumnos = leer_estudiantes_submuestra(
        ruta_estudiantes, columnas_interes, escuelas_sel, factores, chunk_size
    )
    df_alumnos = calcular_senwt_adj(df_alumnos)

    # 4. Colegios (misma comprobación de columnas disponibles)
    columnas_colegio_deseadas = ['CNT', 'STRATUM', 'CNTSCHID'] + columnas_contexto_colegio
    columnas_colegio, ausentes_col = columnas_disponibles(ruta_colegios, columnas_colegio_deseadas)
    claves_ausentes = [c for c in ('CNT', 'STRATUM', 'CNTSCHID') if c in ausentes_col]
    if claves_ausentes:
        raise ValueError(
            f"PISA {año}: faltan columnas clave imprescindibles en {ruta_colegios}: {claves_ausentes}"
        )
    if ausentes_col:
        print(f'  Aviso PISA {año}: columnas de contexto no encontradas en el fichero de '
              f'centros, se omiten: {ausentes_col}')
    df_colegios = leer_colegios(ruta_colegios, columnas_colegio)

    # 5. Unión y optimización
    df_global = unir_alumnos_colegios(df_alumnos, df_colegios)
    df_global = optimizar_tipos(df_global)
    # CNTSCHID es un identificador, no una cantidad: se normaliza siempre a
    # string (algunas ediciones lo traen como int, PISA 2025 como string con
    # ceros a la izquierda). Sin esto, concatenar ediciones con dtypes
    # distintos para la misma columna rompe la escritura a parquet.
    # OJO: usar el dtype nullable 'string' de pandas, no `.astype(str)` — este
    # último convierte un nulo real en el string literal 'None'/'nan', lo que
    # rompe silenciosamente `.isna()` para las filas sin CNTSCHID.
    df_global['CNTSCHID'] = df_global['CNTSCHID'].astype('string')

    # 6. Metadatos de edición
    df_global['EDICION'] = año
    df_global['DOMINIO_PRINCIPAL'] = cfg['dominio_principal']

    # 7. Checkpoint antes de imputar (permite re-arrancar sin releer SAS)
    if guardar_checkpoint:
        os.makedirs(ruta_intermedios, exist_ok=True)
        df_global.to_parquet(ruta_out, engine='pyarrow', index=False)

    del df_alumnos, df_colegios, escuelas_unicas
    gc.collect()

    return df_global