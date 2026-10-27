"""
recursos_alumno.py
==================

Recursos del hogar del alumno armonizados entre las ediciones de PISA 2012, 2015, 2018 y 2022, a partir
de las preguntas de posesiones del cuestionario del alumno (las que componen `HOMEPOS`):

    * `ITEMS_HOGAR`: columna de cada ítem en cada edición y tipo de respuesta, que fija cómo se recodifica.
    * `leer_items_hogar`: ítems armonizados de una edición, una fila por alumno, con sus claves y pesos.
    * `SUBINDICES` / `construir_subindices`: subíndices por suma de ítems (recursos de estudio, riqueza
      del hogar y libros en casa), con un mínimo de ítems respondidos (`RESPUESTAS_MINIMAS`).
    * `alfa_cronbach`: coherencia interna de los ítems de un subíndice.
    * `calibrar_rasch` / `puntuacion_eap`: comprobación con IRT (modelo de Rasch): dificultad de cada ítem y
      puntuación de cada alumno según los ítems que respondió.

Los datos se leen de las tablas maestras (`data/raw/datos_anuales/`, ver `pisa_pipeline.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from girth import INVALID_RESPONSE, rasch_mml

from src.pisa_pipeline import rutas_datos_anuales

EDICIONES_RECURSOS = [2012, 2015, 2018, 2022]
CLAVES_ALUMNO = ['CNT', 'CNTSCHID', 'CNTSTUID']
PESOS_ALUMNO = ['W_FSTUWT', 'SENWT']

# Tipos de respuesta. Sí/no: 1 = sí, 2 = no. Cuántos (coches, baños): 1 = ninguno, 2 = uno, 3 = dos,
# 4 = tres o más, igual en todas las ediciones. Tramos (libros en casa): 1 = 0-10 ... 6 = más de 500; en
# 2022 la escala empieza en 1 = ningún libro y llega a 7. Hay alguno (tipos de libros): hasta 2018, sí/no;
# en 2022 se pregunta cuántos libros hay de cada tipo (códigos 1 a 4, de menos a más, con 1 = ninguno) y
# se convierte en 'tiene al menos uno'. El código 5 de 2022 se trata como 'no lo sé': sus alumnos tienen
# un ESCS y un número de libros intermedios, no los más altos (sección 2 de 01_EDA_recursos_alumno).
SI_NO, CUANTOS, TRAMOS, HAY_ALGUNO = 'sí/no', 'cuántos', 'tramos', 'hay alguno'

# {ítem: (tipo de respuesta, {edición: columna})}. Una edición que falta es una pregunta que no se hizo.
ITEMS_HOGAR = {
    'habitacion_propia': (SI_NO, {2012: 'ST26Q02', 2015: 'ST011Q02TA', 2018: 'ST011Q02TA', 2022: 'ST250Q01JA'}),
    'ordenador_estudios': (SI_NO, {2012: 'ST26Q04', 2015: 'ST011Q04TA', 2018: 'ST011Q04TA', 2022: 'ST250Q02JA'}),
    'internet': (SI_NO, {2012: 'ST26Q06', 2015: 'ST011Q06TA', 2018: 'ST011Q06TA', 2022: 'ST250Q05JA'}),
    'escritorio': (SI_NO, {2012: 'ST26Q01', 2015: 'ST011Q01TA', 2018: 'ST011Q01TA'}),
    'sitio_tranquilo': (SI_NO, {2012: 'ST26Q03', 2015: 'ST011Q03TA', 2018: 'ST011Q03TA'}),
    'software_educativo': (SI_NO, {2012: 'ST26Q05', 2015: 'ST011Q05TA', 2018: 'ST011Q05TA', 2022: 'ST250Q03JA'}),
    'libros_ayuda_estudios': (HAY_ALGUNO, {2012: 'ST26Q10', 2015: 'ST011Q10TA', 2018: 'ST011Q10TA', 2022: 'ST256Q10JA'}),
    'libros_tecnicos': (HAY_ALGUNO, {2012: 'ST26Q11', 2015: 'ST011Q11TA', 2018: 'ST011Q11TA', 2022: 'ST256Q08JA'}),
    'diccionario': (HAY_ALGUNO, {2012: 'ST26Q12', 2015: 'ST011Q12TA', 2018: 'ST011Q12TA', 2022: 'ST256Q09JA'}),
    'coches': (CUANTOS, {2012: 'ST27Q04', 2015: 'ST012Q02TA', 2018: 'ST012Q02TA', 2022: 'ST251Q01JA'}),
    'baños': (CUANTOS, {2012: 'ST27Q05', 2015: 'ST012Q03TA', 2018: 'ST012Q03TA', 2022: 'ST251Q03JA'}),
    'libros_en_casa': (TRAMOS, {2012: 'ST28Q01', 2015: 'ST013Q01TA', 2018: 'ST013Q01TA', 2022: 'ST255Q01JA'}),
}

# Subíndices: suma de sus ítems. Recursos de estudio reúne los ítems de `HEDRES` (recursos educativos del
# hogar de la OCDE) que existen en las cuatro ediciones.
SUBINDICES = {
    'recursos_estudio': ['habitacion_propia', 'ordenador_estudios', 'internet', 'libros_ayuda_estudios', 'libros_tecnicos', 'diccionario'],
    'riqueza_hogar': ['coches', 'baños'],
    'libros': ['libros_en_casa'],
}
# Ítems respondidos que hacen falta para calcular un subíndice (los demás subíndices necesitan todos): con
# menos ítems que el total, la suma se estima como la media de los respondidos por el número de ítems.
RESPUESTAS_MINIMAS = {'recursos_estudio': 4}


def _recodificar(valores: pd.Series, tipo: str, edicion: int) -> pd.Series:
    """Respuesta original -> escala común. Cualquier otro código (no respuesta, no sabe, no aplicable) queda vacío."""
    if tipo == SI_NO:
        return valores.map({1: 1.0, 2: 0.0})
    if tipo == CUANTOS:
        return valores.map({1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0})
    if tipo == HAY_ALGUNO:
        return valores.map({1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0}) if edicion >= 2022 else valores.map({1: 1.0, 2: 0.0})
    # Tramos de libros: en 2022, 'ningún libro' y '1-10 libros' se juntan en la categoría '0-10' de las ediciones anteriores
    if edicion >= 2022:
        return valores.map({1: 1.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0, 6: 5.0, 7: 6.0})
    return valores.map({codigo: float(codigo) for codigo in range(1, 7)})


def leer_items_hogar(ruta_raw_base: str, edicion: int, columnas_extra: list[str] | tuple = ()) -> pd.DataFrame:
    """Una fila por alumno de `edicion` con `CLAVES_ALUMNO`, `year`, `PESOS_ALUMNO`, las `columnas_extra`
    que existan en esa edición (por ejemplo índices oficiales o la nota) y los `ITEMS_HOGAR` recodificados
    a la escala común (los que no se preguntaron en esa edición no aparecen)."""
    origen = {item: columnas[edicion] for item, (_, columnas) in ITEMS_HOGAR.items() if edicion in columnas}
    ruta = rutas_datos_anuales(ruta_raw_base, edicion)[0]
    extra = [columna for columna in columnas_extra if columna in pq.read_schema(ruta).names]
    datos = pd.read_parquet(ruta, columns=CLAVES_ALUMNO + PESOS_ALUMNO + extra + list(origen.values()))
    resultado = datos[CLAVES_ALUMNO + PESOS_ALUMNO + extra].astype({clave: str for clave in CLAVES_ALUMNO})
    resultado.insert(3, 'year', edicion)
    for item, columna in origen.items():
        resultado[item] = _recodificar(datos[columna], ITEMS_HOGAR[item][0], edicion).to_numpy()
    return resultado.reset_index(drop=True)


def construir_subindices(datos: pd.DataFrame) -> pd.DataFrame:
    """Añade a `datos` cada subíndice de `SUBINDICES` como suma de sus ítems: la media de los ítems que el
    alumno respondió por el número de ítems del subíndice. Queda vacío si respondió menos de los que pide
    `RESPUESTAS_MINIMAS` (todos, si el subíndice no aparece ahí)."""
    datos = datos.copy()
    for subindice, items in SUBINDICES.items():
        respuestas = datos.reindex(columns=items)
        minimo = RESPUESTAS_MINIMAS.get(subindice, len(items))
        datos[subindice] = (respuestas.mean(axis=1) * len(items)).where(respuestas.notna().sum(axis=1) >= minimo)
    return datos


def alfa_cronbach(datos: pd.DataFrame, items: list[str], peso_col: str | None = None) -> float:
    """Alfa de Cronbach de `items` sobre los alumnos que los tienen todos: k/(k-1) por (1 - suma de las
    varianzas de cada ítem / varianza de la suma). Con ítems de sí/no equivale al KR-20."""
    completos = datos.dropna(subset=items)
    pesos = completos[peso_col].to_numpy() if peso_col else np.ones(len(completos))

    def varianza(valores):
        media = np.average(valores, weights=pesos)
        return np.average((valores - media) ** 2, weights=pesos)

    numero_items = len(items)
    suma_varianzas = sum(varianza(completos[item].to_numpy()) for item in items)
    return numero_items / (numero_items - 1) * (1 - suma_varianzas / varianza(completos[items].sum(axis=1).to_numpy()))


def calibrar_rasch(datos: pd.DataFrame, items: list[str], peso_col: str = 'SENWT', alumnos: int = 40000, semilla: int = 42) -> pd.Series:
    """Dificultad de cada ítem de `items` (de sí/no) con el modelo de Rasch: cuanto más negativa, más alumnos lo
    tienen. Se estima con una muestra de `alumnos` sacada con probabilidad proporcional a `peso_col` (la librería
    `girth` no admite pesos); los alumnos sin ningún ítem respondido no entran, y los vacíos cuentan como no
    respondidos. Un ítem sin ninguna respuesta en `datos` (por ejemplo, una pregunta que un país no hizo) queda
    vacío en lugar de estimarse."""
    con_dato = [item for item in items if datos[item].notna().any()]
    con_respuesta = datos[datos[con_dato].notna().any(axis=1)]
    probabilidad = (con_respuesta[peso_col] / con_respuesta[peso_col].sum()).to_numpy()
    muestra = con_respuesta.iloc[np.random.default_rng(semilla).choice(len(con_respuesta), size=alumnos, replace=True, p=probabilidad)]
    respuestas = muestra[con_dato].to_numpy()
    matriz = np.where(np.isnan(respuestas), INVALID_RESPONSE, respuestas).astype(int).T   # ítems x alumnos
    return pd.Series(rasch_mml(matriz)['Difficulty'], index=con_dato).reindex(items)


def puntuacion_eap(datos: pd.DataFrame, items: list[str], dificultades: pd.Series, minimo: int | None = None) -> pd.Series:
    """Puntuación de cada alumno en el modelo de Rasch con las `dificultades` dadas: la media de su nivel según
    los ítems que respondió (estimación EAP, con una distribución normal estándar de partida). Solo usa los ítems
    respondidos, así que no hace falta prorratear; queda vacía si el alumno respondió menos de `minimo` ítems
    (por defecto, ninguno)."""
    niveles = np.linspace(-6, 6, 121)
    probabilidad_si = 1 / (1 + np.exp(-(niveles[:, None] - dificultades[items].to_numpy()[None, :])))   # niveles x ítems
    respuestas = np.nan_to_num(datos[items].to_numpy(), nan=-1)
    patrones, patron_de_cada_alumno = np.unique(respuestas, axis=0, return_inverse=True)   # pocas combinaciones distintas
    verosimilitud = np.ones((len(patrones), len(niveles)))
    for posicion in range(len(items)):
        verosimilitud[patrones[:, posicion] == 1] *= probabilidad_si[:, posicion]
        verosimilitud[patrones[:, posicion] == 0] *= 1 - probabilidad_si[:, posicion]
    posterior = verosimilitud * np.exp(-niveles ** 2 / 2)
    puntuacion = pd.Series((posterior @ niveles / posterior.sum(axis=1))[patron_de_cada_alumno.ravel()], index=datos.index)
    respondidos = datos[items].notna().sum(axis=1)
    return puntuacion.where(respondidos >= (minimo if minimo is not None else 1))
