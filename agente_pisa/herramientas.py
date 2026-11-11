"""
Herramientas (Tool Calling) del agente PISA, organizadas en tres vías:

  a) Estadísticas descriptivas en vivo (Pandas sobre los alumnos del modelo, salida JSON compacta):
     get_math_stats, calculate_correlation
  b) Contexto estático (sin cálculo): PNG pre-generados y notas de metodología (metodologia/*.md):
     get_context_chart, get_methodology
  c) Resultados de los modelos, sin cargar ningún modelo (cada herramienta entra en `pisa_tools` si existe su tabla):
     get_regression_weights (coeficientes de la regresión, diferencias entre perfiles y simulación lineal) y
     get_xgboost_patterns (importancia, contribución y forma a partir de los SHAP del XGBoost de cada alumno).

Todas las cifras son de matemáticas, la nota del alumno en este TFM, en los 12 países del panel común.
Las medias se ponderan con SENWT: dentro de un país y edición equivale al peso del alumno. En el panel cada país
pesa casi lo mismo: el SENWT suma igual en cada país y edición con todos los alumnos de PISA, pero no todos entran
en el modelo, así que aquí cada uno pesa entre el 6 y el 12 %. Es el mismo peso que usan la regresión y el XGBoost.

Las herramientas que muestran una imagen devuelven su referencia en la clave "imagen" del JSON;
la app la recoge de los ToolMessage, por lo que el LLM no necesita repetirla en su respuesta.
"""
import difflib
import json
import math
import unicodedata
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from config import (
    AVISO_2018_CORTO, AVISO_POCO_REPRESENTATIVO, DIR_METODOLOGIA, NOMBRES_PAISES, PAISES_PANEL, PISA_YEARS,
    TAMANOS, TIPOS_CENTRO, UBICACIONES, VARIABLES_NUMERICAS, etiqueta,
)
from datos import (
    cargar_alumnos, cargar_catalogo_graficos, cargar_coeficientes, cargar_conductas, cargar_shap, conductas_disponibles,
    regresion_disponible, shap_disponible,
)

# ==========================================================
#  NORMALIZADORES Y UTILIDADES
# ==========================================================

MINIMO_ALUMNOS = 30   # por debajo, el grupo se marca como "pocos datos"

# Sinónimos para que el usuario pueda preguntar de forma genérica ("privados", "rurales", "chicas"...)
SINONIMOS = {
    "tipo_centro": {
        "Público": ["publico", "publicos", "public", "estatal", "pub"],
        "Concertado": ["concertado", "concertados", "charter", "con", "privado concertado"],
        "Privado": ["privado", "privados", "private", "pri", "privado independiente"],
    },
    "ubicacion": {
        "Urbano": ["urbano", "urbanos", "urbana", "ciudad", "urban", "urb"],
        "Rural": ["rural", "rurales", "pueblo", "rur"],
    },
    "tamano": {
        "Pequeño": ["pequeno", "pequenos", "pequena", "small", "peq", "escuela pequena"],
        "Mediano": ["mediano", "medianos", "mediana", "medium", "med", "instituto mediano"],
        "Grande": ["grande", "grandes", "large", "macro", "macroinstituto", "macro-instituto"],
    },
    "nivel_compromiso": {
        "Bajo": ["bajo", "baja", "low", "poco"],
        "Medio": ["medio", "media", "medium", "intermedio"],
        "Alto": ["alto", "alta", "high", "mucho"],
    },
    "nivel_recursos": {
        "Hasta 4": ["hasta 4", "bajo", "bajos", "pocos", "low", "4"],
        "5 de 6": ["5 de 6", "medio", "medios", "5", "medium"],
        "Los 6": ["los 6", "alto", "altos", "todos", "completos", "6", "high", "all"],
    },
    "sexo": {
        "Chica": ["chica", "chicas", "mujer", "mujeres", "femenino", "girl", "girls", "female"],
        "Chico": ["chico", "chicos", "hombre", "hombres", "masculino", "boy", "boys", "male"],
    },
    "origen": {
        "Nativo": ["nativo", "nativos", "native", "autoctono"],
        "Inmigrante": ["inmigrante", "inmigrantes", "immigrant", "extranjero", "extranjeros"],
    },
}

# Nombres de los parámetros de la herramienta -> columna de alumnos_modelo
GRUPOS = {
    "tipo_centro": "tipo_centro", "ubicacion": "ubicacion", "tamano": "tamano", "compromiso": "nivel_compromiso",
    "recursos": "nivel_recursos", "sexo": "sexo", "origen": "origen", "pais": "CNT", "edicion": "year",
}

SINONIMOS_VARIABLES = {
    "nota": ["nota", "matematicas", "rendimiento", "puntuacion", "math", "score"],
    "Indice_Compromiso_Competencia": ["compromiso", "indice", "indice de compromiso", "competencia", "engagement"],
    "recursos_estudio": ["recursos", "recursos de estudio", "recursos estudio"],
    "riqueza_hogar": ["riqueza", "riqueza del hogar", "wealth"],
    "cantidad_libros": ["libros", "libros en casa", "books"],
    "HISEI_TREND": ["hisei", "ocupacion", "ocupacion de los padres", "estatus"],
    "PAREDINT_TREND": ["paredint", "estudios de los padres", "estudios padres", "educacion padres"],
    "composicion_social_centro": ["composicion social", "composicion social del centro", "escs del centro"],
    "dotacion_centro": ["dotacion", "dotacion del centro", "recursos del centro"],
}


def _sin_acentos(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto).strip().lower())
    return "".join(c for c in texto if not unicodedata.combining(c)).replace("_", " ")


def _json(datos: dict) -> str:
    """Serializa la salida de una herramienta en JSON compacto (menos tokens que texto o tablas)."""
    return json.dumps(datos, ensure_ascii=False, separators=(",", ":"), default=str)


def _r(valor, decimales: Optional[int] = None):
    """Redondea según la magnitud (menos dígitos = menos tokens) y convierte NaN en None para el JSON."""
    if valor is None or (isinstance(valor, (float, np.floating)) and np.isnan(valor)):
        return None
    if not isinstance(valor, (int, float, np.number)):
        return valor
    valor = float(valor)
    if decimales is None:
        decimales = 0 if abs(valor) >= 1000 else 1 if abs(valor) >= 10 else 2
    redondeado = round(valor, decimales)
    return int(redondeado) if decimales == 0 else redondeado


def _error(mensaje: str) -> str:
    return _json({"error": mensaje})


def _years(year) -> list[int]:
    """'all' -> todas las ediciones; un año -> lista con ese año (validado)."""
    if str(year).lower().strip() in ("all", "todos", "todas"):
        return PISA_YEARS
    y = int(year)
    if y not in PISA_YEARS:
        raise ValueError(f"El año '{year}' no es válido. Usa {PISA_YEARS} o 'all'.")
    return [y]


def normalizar_valor(columna: str, valor: Optional[str]) -> Optional[str]:
    """Traduce términos genéricos ('privados', 'rural', 'chicas'...) a la categoría de la columna."""
    if valor is None or _sin_acentos(valor) in ("", "none", "todos", "todas", "all", "cualquiera"):
        return None
    texto = _sin_acentos(valor)
    todos = {}
    for categoria, sinonimos in SINONIMOS[columna].items():
        for s in [_sin_acentos(categoria)] + sinonimos:
            todos[s] = categoria
    if texto in todos:
        return todos[texto]
    parecido = difflib.get_close_matches(texto, list(todos), n=1, cutoff=0.75)
    if parecido:
        return todos[parecido[0]]
    raise ValueError(f"Valor '{valor}' no reconocido para {etiqueta(columna)}. Usa: {', '.join(SINONIMOS[columna])}.")


def normalizar_pais(pais: Optional[str]) -> Optional[str]:
    """Código ISO del país a partir del código o del nombre; solo países del panel común."""
    if pais is None or _sin_acentos(pais) in ("", "none", "todos", "all", "panel", "ocde", "oecd"):
        return None
    texto = _sin_acentos(pais)
    if texto.upper() in PAISES_PANEL:
        return texto.upper()
    nombres = {_sin_acentos(nombre): iso for iso, nombre in NOMBRES_PAISES.items()}
    parecido = difflib.get_close_matches(texto, list(nombres), n=1, cutoff=0.75)
    if parecido:
        return nombres[parecido[0]]
    raise ValueError(f"'{pais}' no está en el panel de 12 países: {', '.join(PAISES_PANEL)}.")


def normalizar_variable(variable: str) -> str:
    """Columna numérica a partir de su nombre o de un sinónimo."""
    if variable in VARIABLES_NUMERICAS:
        return variable
    texto = _sin_acentos(variable)
    todos = {s: col for col, lista in SINONIMOS_VARIABLES.items() for s in lista}
    if texto in todos:
        return todos[texto]
    parecido = difflib.get_close_matches(texto, list(todos), n=1, cutoff=0.6)
    if parecido:
        return todos[parecido[0]]
    raise ValueError(f"Variable '{variable}' no disponible. Usa: {', '.join(SINONIMOS_VARIABLES)}.")


def media_ponderada(df: pd.DataFrame, columna: str) -> Optional[float]:
    """Media ponderada con SENWT ignorando los vacíos."""
    validos = df[columna].notna()
    pesos = df.loc[validos, "SENWT"]
    return float(np.average(df.loc[validos, columna], weights=pesos)) if pesos.sum() > 0 else None


def filtrar_alumnos(year="all", CNT: Optional[str] = None, **filtros) -> tuple[pd.DataFrame, dict, list[str]]:
    """Alumnos filtrados por edición, país y cualquier columna de SINONIMOS. Devuelve (datos, filtros, avisos)."""
    df = cargar_alumnos()
    years = _years(year)
    df = df[df["year"].isin(years)]
    aplicados = {}
    pais = normalizar_pais(CNT)
    if pais:
        df = df[df["CNT"] == pais]
        aplicados["pais"] = pais
    for columna, valor in filtros.items():
        categoria = normalizar_valor(columna, valor)
        if categoria:
            df = df[df[columna] == categoria]
            aplicados[columna] = categoria
    avisos = []
    if 2018 in years:
        avisos.append(AVISO_2018_CORTO)
    if df["perfil_poco_representativo"].any():
        avisos.append(AVISO_POCO_REPRESENTATIVO)
    return df, aplicados, avisos


# ==============================================================================
# a) ESTADÍSTICAS DESCRIPTIVAS EN VIVO
# ==============================================================================

@tool
def get_math_stats(year: Union[int, str] = 2022, CNT: str = "", variable: str = "nota", tipo_centro: str = "",
                   ubicacion: str = "", tamano: str = "", compromiso: str = "", recursos: str = "", sexo: str = "",
                   origen: str = "", group_by: str = "") -> str:
    """
    Weighted mean of the math score ('nota') or another student variable for any subgroup.
    - year: 2012, 2015, 2018, 2022 or 'all'. CNT: country, empty = 12-country panel.
    - variable: nota, compromiso, recursos, riqueza, libros, ocupacion, estudios padres, composicion social, dotacion.
    - Filters take everyday words: tipo_centro publico|concertado|privado, ubicacion urbano|rural,
      tamano pequeño|mediano|grande, compromiso|recursos bajo|medio|alto, sexo chica|chico, origen nativo|inmigrante.
    - group_by: tipo_centro, ubicacion, tamano, compromiso, recursos, sexo, origen, pais (ranking) or edicion.
    """
    try:
        if group_by and group_by not in GRUPOS:
            return _error(f"group_by '{group_by}' no válido. Usa: {', '.join(GRUPOS)}.")
        columna = normalizar_variable(variable)
        df, aplicados, avisos = filtrar_alumnos(year, CNT, tipo_centro=tipo_centro, ubicacion=ubicacion,
                                                tamano=tamano, nivel_compromiso=compromiso, nivel_recursos=recursos,
                                                sexo=sexo, origen=origen)
        if df.empty:
            return _error("No hay alumnos con esos filtros.")

        varias_ediciones = df["year"].nunique() > 1 and group_by != "edicion"

        def resumen(bloque: pd.DataFrame):
            if varias_ediciones:
                return {int(y): _r(media_ponderada(g, columna)) for y, g in bloque.groupby("year")}
            return _r(media_ponderada(bloque, columna))

        salida = {"variable": etiqueta(columna), "ambito": aplicados.pop("pais", "panel de 12 países"),
                  "edicion": "2012-2022" if str(year).lower() in ("all", "todos", "todas") else int(year),
                  "filtros": aplicados, "n_alumnos": len(df), "media": resumen(df)}
        if group_by:
            col_grupo = GRUPOS[group_by]
            peso_total = df["SENWT"].sum()
            grupos, pocos = {}, []
            for categoria, g in df.groupby(col_grupo, observed=True):
                clave = int(categoria) if col_grupo == "year" else str(categoria)
                if col_grupo in ("CNT", "year"):   # su % solo refleja cuántos alumnos entran en el modelo: no aporta
                    grupos[clave] = resumen(g)
                else:
                    grupos[clave] = {"media": resumen(g), "pct": _r(100 * g["SENWT"].sum() / peso_total, 1)}
                if len(g) < MINIMO_ALUMNOS:
                    pocos.append(str(clave))
            salida[f"por_{group_by}"] = grupos
            if pocos:
                avisos.append(f"Pocos alumnos en: {', '.join(pocos)}")
        if avisos:
            salida["avisos"] = avisos
        return _json(salida)
    except Exception as e:
        return _error(f"Estadísticas: {e}")


@tool
def calculate_correlation(variable_a: str, variable_b: str = "nota", year: Union[int, str] = 2022, CNT: str = "",
                          tipo_centro: str = "", ubicacion: str = "", tamano: str = "") -> str:
    """
    Weighted correlation between two student variables (names as in get_math_stats). Report the mean r within each
    country-edition (the fair comparison). Optional filters: CNT, tipo_centro, ubicacion, tamano.
    """
    try:
        col_a, col_b = normalizar_variable(variable_a), normalizar_variable(variable_b)
        df, aplicados, avisos = filtrar_alumnos(year, CNT, tipo_centro=tipo_centro, ubicacion=ubicacion, tamano=tamano)
        df = df.dropna(subset=[col_a, col_b])
        if len(df) < MINIMO_ALUMNOS:
            return _error(f"Solo {len(df)} alumnos con datos en ambas variables.")

        def r_ponderada(g: pd.DataFrame) -> float:
            w = g["SENWT"].to_numpy(dtype=float)
            x, y = g[col_a].to_numpy(dtype=float), g[col_b].to_numpy(dtype=float)
            x, y = x - np.average(x, weights=w), y - np.average(y, weights=w)
            return float(np.sum(w * x * y) / np.sqrt(np.sum(w * x * x) * np.sum(w * y * y)))

        dentro = [r_ponderada(g) for _, g in df.groupby(["CNT", "year"], observed=True) if len(g) >= MINIMO_ALUMNOS]
        r = float(np.mean(dentro)) if dentro else r_ponderada(df)
        salida = {
            "a": etiqueta(col_a), "b": etiqueta(col_b), "ambito": aplicados.pop("pais", "panel de 12 países"),
            "filtros": aplicados, "n_alumnos": len(df),
            "r_dentro_pais_edicion": _r(r, 2), "r_global": _r(r_ponderada(df), 2),
            "fuerza": "fuerte" if abs(r) >= 0.5 else "moderada" if abs(r) >= 0.3 else "débil",
        }
        if avisos:
            salida["avisos"] = avisos
        return _json(salida)
    except Exception as e:
        return _error(f"Correlación: {e}")


# ==============================================================================
# b) CONTEXTO ESTÁTICO (imágenes pre-generadas)
# ==============================================================================

def _puntuar_grafico(palabras: list[str], id_grafico: str, descripcion: str) -> float:
    """Parecido medio de las palabras buscadas con el texto del gráfico, con un extra si están en su identificador."""
    texto = _sin_acentos(f"{id_grafico} {descripcion}").replace(",", " ").replace("(", " ").replace(")", " ").split()
    id_palabras = _sin_acentos(id_grafico).split()
    parecido = np.mean([max(difflib.SequenceMatcher(None, p, t).ratio() for t in texto) for p in palabras])
    en_id = np.mean([max(difflib.SequenceMatcher(None, p, t).ratio() for t in id_palabras) >= 0.8 for p in palabras])
    return float(parecido + 0.2 * en_id)


@tool
def get_context_chart(keyword: str = "") -> str:
    """
    A pre-generated chart of the analysis, chosen by keyword (e.g. 'recursos', 'regresion perfiles', 'robustez',
    'Australia', 'xgboost importancia'). Also lists the five most related charts.
    """
    try:
        catalogo = cargar_catalogo_graficos()
        if catalogo.empty:
            return _error("No hay gráficos de contexto disponibles todavía.")
        # Palabras con contenido (fuera artículos y preposiciones cortas); sin ellas, el primer gráfico
        palabras = [p for p in _sin_acentos(keyword).split() if len(p) > 3]
        if palabras:
            puntuacion = catalogo.apply(lambda f: _puntuar_grafico(palabras, f["id"], f["descripcion"]), axis=1)
            catalogo = catalogo.loc[puntuacion.sort_values(ascending=False, kind="stable").index]
        elegido = catalogo.iloc[0]
        return _json({"imagen": elegido["ruta"], "descripcion": elegido["descripcion"],
                      "otros_relacionados": list(catalogo["descripcion"].iloc[1:6])})
    except Exception as e:
        return _error(f"Gráfico de contexto: {e}")


# Temas de metodología: uno por fichero de metodologia/ (añadir un .md basta para que el agente lo vea)
TEMAS_METODOLOGIA = tuple(sorted(ruta.stem for ruta in DIR_METODOLOGIA.glob("*.md")))


@tool
def get_methodology(topic: Literal[TEMAS_METODOLOGIA]) -> str:
    """
    The project's methodology note on a topic: how indices, variables and school profiles were built, missing
    answers, sample and limitations, how to read the models ('modelos') and which results are fragile ('robustez').
    """
    try:
        return _json({"tema": topic, "texto": (DIR_METODOLOGIA / f"{topic}.md").read_text(encoding="utf-8")})
    except Exception as e:
        return _error(f"Metodología: {e}")


# ==============================================================================
# c) RESULTADOS DE LOS MODELOS (sin cargar ningún modelo)
# ==============================================================================

# Variables de la regresión y unidad de su coeficiente en unidades originales (las de alumnos_modelo)
UNIDADES_REGRESION = {
    "Indice_Compromiso_Competencia": "puntos de nota por cada punto más del índice de compromiso (≈ 1 desviación "
                                     "típica), que combina tres conductas: no faltar a clase, llegar puntual y dedicar "
                                     "a los deberes un tiempo intermedio. Para una conducta concreta, ver 'conductas'",
    "recursos_estudio": "puntos de nota por cada uno más de los seis recursos de estudio: habitación propia, "
                        "ordenador, internet, libros de ayuda, libros técnicos y diccionario",
    "HISEI_TREND": "puntos por cada punto más de ocupación de los padres (escala de 11 a 89)",
    "PAREDINT_TREND": "puntos por cada año más de estudios de los padres",
    "riqueza_hogar": "puntos por cada coche o baño más (de 0 a 6)",
    "cantidad_libros": "puntos por cada tramo más de libros en casa (de 1 a 6)",
    "composicion_social_centro": "puntos por cada punto más del origen social medio del centro (≈ 2 desviaciones típicas)",
    "dotacion_centro": "puntos por cada punto más de dotación del centro (≈ 1 desviación típica)",
    "sexo_femenino": "puntos de las chicas frente a los chicos",
    "inmigrante": "puntos de los inmigrantes frente a los nativos",
}
NOMBRES_REGRESION = {**VARIABLES_NUMERICAS, "sexo_femenino": "Chica (frente a chico)",
                     "inmigrante": "Inmigrante (frente a nativo)"}
SINONIMOS_REGRESION = {
    **{col: lista for col, lista in SINONIMOS_VARIABLES.items() if col != "nota"},
    "sexo_femenino": ["sexo", "chica", "chicas", "genero", "mujer", "mujeres"],
    "inmigrante": ["inmigrante", "inmigrantes", "inmigracion", "origen"],
    "efecto_perfil": ["perfil", "perfiles", "perfil de centro", "tipo de centro", "centro", "estrato", "estratos fijos"],
}
# Las conductas del compromiso llevan al compromiso, que devuelve el coeficiente de cada una
SINONIMOS_REGRESION["Indice_Compromiso_Competencia"] = SINONIMOS_REGRESION["Indice_Compromiso_Competencia"] + [
    "faltar", "faltar a clase", "absentismo", "asistencia", "llegar tarde", "impuntualidad", "puntualidad", "deberes",
    "tiempo de deberes", "horas de deberes", "conductas"]
# Etiqueta de cada combinación de conductas (para la forma del compromiso en el XGBoost)
TEXTO_IMPUNTUALIDAD = {0: "siempre puntual", 1: "llega tarde 1 o 2 veces", 2: "llega tarde 3 o más veces"}
VARIABLES_POR_PERFIL = ("Indice_Compromiso_Competencia", "recursos_estudio")   # las únicas con coeficiente por perfil
PERFIL_REFERENCIA = "PUB_URB_MED"
# Valores mínimo y máximo de cada variable en los datos: una simulación no puede salirse de ellos
RANGO = {"Indice_Compromiso_Competencia": (-2.1, 1.54), "recursos_estudio": (0, 6), "HISEI_TREND": (11, 89),
         "PAREDINT_TREND": (3, 16), "riqueza_hogar": (0, 6), "cantidad_libros": (1, 6), "composicion_social_centro": (-2.53, 1.2),
         "dotacion_centro": (-4.32, 3.98), "sexo_femenino": (0, 1), "inmigrante": (0, 1)}
# Resultados que no aguantan las comprobaciones de robustez (metodologia/robustez.md): se avisan en la propia salida
# cuando la consulta incluye las ediciones indicadas. (variable, estrato) -> (ediciones, aviso)
# La subida del compromiso de 2012 a 2022, el otro resultado frágil, va en CONCLUSION_CAMBIO.
AVISOS_FRAGILES = {
    ("recursos_estudio", "PRI_URB_MACRO"): ({2022}, "Resultado frágil: la subida de 2012 a 2022 del coeficiente de los "
        "recursos en el privado urbano grande (unos 14 puntos por desviación típica) depende solo de Australia: sin "
        "ella la subida baja a 6,8 y no se distingue de cero; el coeficiente no se anula. No es su diferencia de nota "
        "(+10) (nota de robustez)"),
}
# Conclusión del estudio sobre el cambio de 2012 a 2022 (metodologia/robustez.md). Va la primera en la salida para que
# la respuesta empiece por ella: con la cautela al final, el modelo abría con «ha aumentado (p = 0,029)»
CONCLUSION_CAMBIO = ("En el conjunto de los 12 países, la relación del compromiso y de los recursos con la nota apenas "
                     "cambia de 2012 a 2022 y es parecida en todos los perfiles de centro. Los cambios que aparecen a "
                     "primera vista se deben a un país o a un cambio en la pregunta")
EXPLICACION_CAMBIO = {
    "Indice_Compromiso_Competencia": "La subida del índice (+1,5) es frágil, no un cambio real: viene solo de los "
        "deberes, cuya pregunta cambia en cada edición, y deja de distinguirse de cero al quitar algunos países. Faltar "
        "a clase y llegar tarde pesan igual en toda la década. No llamarla significativa ni decir que ha aumentado",
    "recursos_estudio": "Su cambio no se distingue de cero. El único salto, la subida de su coeficiente en el privado "
        "urbano grande (unos 14 puntos por desviación típica de 2012 a 2022), depende solo de Australia",
}
LECTURA_CAMBIO_INDICE = "aparente y frágil: no contarlo como aumento ni como cambio significativo (ver 'conclusion_2012_2022')"
LECTURA_CAMBIO_CONDUCTA = {"deberes_tiempo_intermedio": "la pregunta de deberes cambia en cada edición: no es un "
                                                        "cambio real de la relación"}
AVISOS_SIMULACION = {
    "recursos_estudio": "Cuenta lineal. Casi todos los alumnos tienen ya 5 o 6 de los seis recursos (ver "
                        "'recursos_del_alumnado'): «el doble» o «muchos más» solo es posible para quien tiene pocos. "
                        "Cerca de 6 la ganancia real es menor (efecto techo)",
    "Indice_Compromiso_Competencia": "El índice da el mismo peso a las tres conductas, pero no pesan igual: para una "
                                     "conducta concreta («deja de faltar», «llega puntual»), usar 'conductas', no esta "
                                     "cuenta. El índice solo toma 12 valores, de -2,1 a 1,5",
}


def _variable_regresion(variable: str) -> str:
    """Variable de la regresión a partir de su nombre o de un sinónimo ('perfil' = diferencia entre perfiles)."""
    if variable in SINONIMOS_REGRESION:
        return variable
    texto = _sin_acentos(variable)
    todos = {s: col for col, lista in SINONIMOS_REGRESION.items() for s in lista}
    if texto in todos:
        return todos[texto]
    parecido = difflib.get_close_matches(texto, list(todos), n=1, cutoff=0.6)
    if parecido:
        return todos[parecido[0]]
    raise ValueError(f"Variable '{variable}' no está en la regresión. Usa: compromiso, recursos, ocupacion, "
                     "estudios padres, riqueza, libros, composicion social, dotacion, sexo, inmigrante o perfil.")


def _perfiles(tipo_centro, ubicacion, tamano) -> Optional[list[str]]:
    """Códigos de perfil (PUB_URB_MED...) que cumplen los filtros; None si no se filtra."""
    elegidos = [normalizar_valor("tipo_centro", tipo_centro), normalizar_valor("ubicacion", ubicacion),
                normalizar_valor("tamano", tamano)]
    if not any(elegidos):
        return None
    catalogos = [TIPOS_CENTRO, UBICACIONES, TAMANOS]
    codigos = [f"{t}_{u}_{m}" for t in TIPOS_CENTRO for u in UBICACIONES for m in TAMANOS]
    return [c for c in codigos
            if all(e is None or catalogo[parte] == e for catalogo, parte, e in zip(catalogos, c.split("_"), elegidos))]


def _p(valor: float):
    """p-valor legible: los muy pequeños como '<0.001' (con 0.0 el modelo escribiría «p = 0»)."""
    return "<0.001" if valor < 0.001 else _r(valor, 3)


def _nombre_perfil(codigo: str) -> str:
    tipo, ubicacion, tamano = codigo.split("_")
    return f"{TIPOS_CENTRO[tipo]} {UBICACIONES[ubicacion].lower()} {TAMANOS[tamano].lower()}"


def _bloque(filas: pd.DataFrame, cambio: Optional[float]) -> dict:
    """Coeficiente con intervalo y p de cada edición; con varias, también el cambio de 2012 a 2022."""
    def fila(f) -> dict:
        datos = {"coef": _r(f.coeficiente), "ic": [_r(f.ic_inf), _r(f.ic_sup)], "p": _p(f.p_valor)}
        if pd.notna(f.coeficiente_dt):   # para comparar variables entre sí, aunque se pidan por separado
            datos["por_desv_tipica"] = _r(f.coeficiente_dt)
        if cambio is not None:
            datos["cambio_total_nota"] = _r(f.coeficiente * cambio)
            datos["ic_cambio_total"] = sorted([_r(f.ic_inf * cambio), _r(f.ic_sup * cambio)])
        return datos

    por_edicion = {int(f.edicion): fila(f) for f in filas.itertuples()}
    if len(por_edicion) == 1:
        return next(iter(por_edicion.values()))
    extremos = filas.set_index("edicion")
    if {2012, 2022} <= set(extremos.index):
        diferencia = extremos.loc[2022, "coeficiente"] - extremos.loc[2012, "coeficiente"]
        # Muestras independientes: el error del cambio combina los de las dos ediciones
        error = math.hypot(extremos.loc[2012, "error_tipico"], extremos.loc[2022, "error_tipico"])
        por_edicion["cambio_2012_2022"] = {"dif": _r(diferencia),
                                           "p": _p(math.erfc(abs(diferencia / error) / math.sqrt(2)))}
    return por_edicion


def _contexto_recursos(years: list[int]) -> dict:
    """Cuántos de los seis recursos tiene el alumnado en cada edición (media y reparto, ponderado con SENWT)."""
    alumnos = cargar_alumnos()
    contexto = {}
    for year in years:
        g = alumnos[alumnos["year"] == year]
        cuantos = np.floor(g["recursos_estudio"] + 0.5)   # respuestas parciales reescaladas (4,8...) al entero
        pesos = g["SENWT"]
        pct = lambda filtro: _r(100 * pesos[filtro].sum() / pesos.sum(), 0)
        contexto[year] = {"media_de_6": _r(np.average(g["recursos_estudio"], weights=pesos), 1),
                          "pct_con_los_6": pct(cuantos == 6), "pct_con_5": pct(cuantos == 5),
                          "pct_con_3_o_menos": pct(cuantos <= 3)}
    return contexto[years[0]] if len(years) == 1 else contexto


def _contexto_conductas(years: list[int]) -> dict:
    """Qué parte del alumnado tiene cada conducta del compromiso en cada edición (ponderado con SENWT)."""
    alumnos = cargar_alumnos()
    contexto = {}
    for year in years:
        g = alumnos[alumnos["year"] == year]
        pesos = g["SENWT"]
        pct = lambda filtro: _r(100 * pesos[filtro].sum() / pesos.sum(), 0)
        contexto[year] = {"pct_ha_faltado": pct(g["SKIPPING"] == 1), "pct_llega_tarde_alguna_vez": pct(g["TARDYSD"] >= 1),
                          "pct_tarde_3_o_mas": pct(g["TARDYSD"] == 2), "pct_deberes_tiempo_intermedio": pct(g["deberes_en_u"] == 1),
                          "pct_las_tres_bien": pct((g["SKIPPING"] == 0) & (g["TARDYSD"] == 0) & (g["deberes_en_u"] == 1))}
    return contexto[years[0]] if len(years) == 1 else contexto


NOMBRES_CONDUCTAS = {"faltar_a_clase": "Faltar a clase", "llegar_tarde": "Llegar tarde",
                     "deberes_tiempo_intermedio": "Deberes en tiempo intermedio"}
NOTA_CONDUCTAS = ("Puntos de nota de cada conducta (no por desviación típica), de un modelo con las tres por separado "
                  "y los mismos controles. Solo hay 2012 y 2022, y no se guardó el error de cada año: no dar intervalo "
                  "ni p de cada cifra; el p es solo del cambio de 2012 a 2022")


def _conductas() -> dict:
    """Puntos de nota de cada conducta en 2012 y 2022 (modelo con las tres por separado) y su cambio."""
    def cambio(f) -> dict:
        datos = {"dif": _r(f.cambio), "p": _p(f.p_cambio)}
        if f.conducta in LECTURA_CAMBIO_CONDUCTA:
            datos["lectura"] = LECTURA_CAMBIO_CONDUCTA[f.conducta]
        return datos

    return {NOMBRES_CONDUCTAS[f.conducta]: {"que_compara": f.descripcion, "puntos_2012": _r(f.coef_2012),
                                            "puntos_2022": _r(f.coef_2022), "cambio_2012_2022": cambio(f)}
            for f in cargar_conductas().itertuples()}


@tool
def get_regression_weights(variable: str = "", year: Union[int, str] = 2022, tipo_centro: str = "",
                           ubicacion: str = "", tamano: str = "", cambio: Optional[float] = None,
                           desde: Optional[float] = None) -> str:
    """
    Linear regression of the math score, one per edition: coefficients with interval and p.
    No country filter: one model for the 12 countries together.
    - To compare variables ("which matters more?") call with NO variable: all ranked by 'por_desv_tipica'.
    - variable: compromiso (also faltar, llegar tarde, deberes: adds 'conductas', use it for one behaviour), recursos,
      ocupacion, estudios padres, riqueza, libros, composicion social, dotacion, sexo, inmigrante. With
      tipo_centro/ubicacion/tamano: its coefficient inside those profiles (only compromiso and recursos).
    - variable='perfil': gap of each profile vs public urban medium, everything else equal (private vs public: no filter).
    - cambio (and optional starting value desde): linear simulation. Resources are items out of six: "double" for a
      student with 3 -> desde=3, cambio=3; losing the computer -> cambio=-1.
    - year: one edition or 'all' (one call; gives the 2012-2022 change and its p).
    """
    try:
        coeficientes = cargar_coeficientes()
        years = _years(year)
        coeficientes = coeficientes[coeficientes["edicion"].isin(years)]
        perfiles = _perfiles(tipo_centro, ubicacion, tamano)
        avisos = [AVISO_2018_CORTO] if 2018 in years else []
        comun = coeficientes[coeficientes["regresion"] == "comun"]

        # Sin variable: peso de todas, ordenadas por puntos por desviación típica (una sola edición)
        if not variable:
            edicion = years[-1]
            filas = comun[(comun["edicion"] == edicion) & (comun["estrato"] == "todos")
                          & comun["variable"].isin(UNIDADES_REGRESION)]
            filas = filas.reindex(filas["coeficiente_dt"].abs().sort_values(ascending=False, na_position="last").index)
            salida = {"edicion": edicion, "variables": [
                {"variable": NOMBRES_REGRESION[f.variable], "por_desv_tipica": _r(f.coeficiente_dt), "coef": _r(f.coeficiente),
                 "unidad": UNIDADES_REGRESION[f.variable], "p": _p(f.p_valor)} for f in filas.itertuples()]}
            if len(years) > 1:
                avisos.append("Peso de todas las variables solo de 2022; para la evolución, pedir cada variable con year='all'")
            if avisos:
                salida["avisos"] = avisos
            return _json(salida)

        columna = _variable_regresion(variable)
        salida = {"edicion": "2012-2022" if len(years) > 1 else years[0]}

        if columna == "efecto_perfil":
            # Diferencia de nota de cada perfil frente al de referencia, con lo demás igual
            filas = comun[comun["variable"] == "efecto_perfil"]
            if perfiles is not None:
                filas = filas[filas["estrato"].isin(perfiles)]
            salida["referencia"] = f"{_nombre_perfil(PERFIL_REFERENCIA)} (diferencia 0 por definición)"
            salida["unidad"] = "puntos de diferencia frente a la referencia, con alumnos iguales en todo lo demás"
            salida["por_perfil"] = {_nombre_perfil(p): _bloque(g, None) for p, g in filas.groupby("estrato")}
            if len(years) > 1:
                # 14 perfiles x 4 ediciones: sin intervalo y con los distintos de cero ya listados, porque repasarlos
                # todos llevaba al modelo a copiar mal cifras o a razonar en bucle hasta quedarse sin salida
                for bloque in salida["por_perfil"].values():
                    for datos in bloque.values():
                        datos.pop("ic", None)
                salida["distintos_de_cero"] = {
                    int(e): [f"{_nombre_perfil(f.estrato)} {_r(f.coeficiente):+} (p {_p(f.p_valor)})" for f in g.itertuples()]
                    for e, g in filas[filas["p_valor"] < 0.05].groupby("edicion")}
            avisos.append("Diferencias con los 12 países juntos: no hay cifras por país ni se comprobaron quitando "
                          "países. No es la cautela de Australia, que es sobre el coeficiente de los recursos")
            if cambio is not None:
                avisos.append("La simulación no se aplica a la diferencia entre perfiles")
        else:
            if columna in EXPLICACION_CAMBIO and {2012, 2022} <= set(years):
                salida["conclusion_2012_2022"] = f"{CONCLUSION_CAMBIO}. {EXPLICACION_CAMBIO[columna]}"
            salida["variable"] = NOMBRES_REGRESION[columna]
            salida["unidad"] = UNIDADES_REGRESION[columna]
            minimo, maximo = RANGO[columna]
            if cambio is not None:
                if desde is not None:
                    if not minimo <= desde <= maximo:
                        return _error(f"El punto de partida {desde} está fuera de la escala ({minimo} a {maximo}).")
                    if not minimo <= desde + cambio <= maximo:
                        return _error(f"Partiendo de {desde}, el cambio solo puede ir de {_r(minimo - desde)} a "
                                      f"{_r(maximo - desde)}: la escala va de {minimo} a {maximo}.")
                    salida.update({"desde": desde, "hasta": _r(desde + cambio)})
                elif abs(cambio) > maximo - minimo:
                    return _error(f"Un cambio de {cambio} se sale de la escala de la variable ({minimo} a {maximo}).")
                salida["cambio_simulado"] = cambio
                if columna in AVISOS_SIMULACION:
                    avisos.append(AVISOS_SIMULACION[columna])
            if columna == "recursos_estudio":
                salida["recursos_del_alumnado"] = _contexto_recursos(years)
            elif columna == "Indice_Compromiso_Competencia" and conductas_disponibles():
                salida["conductas"] = _conductas()
                salida["nota_conductas"] = NOTA_CONDUCTAS
                salida["conductas_del_alumnado"] = _contexto_conductas(years)
            filas_comun = comun[(comun["variable"] == columna) & (comun["estrato"] == "todos")]
            if perfiles is not None and columna in VARIABLES_POR_PERFIL:
                # Lo pedido (el perfil) va primero; el común queda detrás, solo para comparar
                filas = coeficientes[(coeficientes["regresion"] == "por_perfil") & (coeficientes["variable"] == columna)
                                     & coeficientes["estrato"].isin(perfiles)]
                salida["por_perfil"] = {_nombre_perfil(p): _bloque(g, cambio) for p, g in filas.groupby("estrato")}
                salida["para_comparar_todos_los_perfiles"] = _bloque(filas_comun, cambio)
            else:
                filas = filas_comun
                salida["coeficiente_comun"] = _bloque(filas_comun, cambio)
            if perfiles is not None and columna not in VARIABLES_POR_PERFIL:
                avisos.append(f"{NOMBRES_REGRESION[columna]} tiene un solo coeficiente para todos los perfiles: "
                              "solo el compromiso y los recursos tienen uno por perfil")
            if columna == "Indice_Compromiso_Competencia":
                # La lectura, junto al p del cambio del índice: suelto, el modelo lo contaba como un aumento
                for bloque in (salida.get("coeficiente_comun"), salida.get("para_comparar_todos_los_perfiles")):
                    if bloque and "cambio_2012_2022" in bloque:
                        bloque["cambio_2012_2022"]["lectura"] = LECTURA_CAMBIO_INDICE

        avisos += [aviso for (var, estrato), (ediciones, aviso) in AVISOS_FRAGILES.items()
                   if var == columna and estrato in set(filas["estrato"]) and ediciones <= set(years)]
        pocos = sorted({_nombre_perfil(p) for p in filas.loc[filas["poco_representativo"] == True, "estrato"]
                        if p != "todos"})
        if pocos:
            avisos.append(f"Perfiles con menos de 30 centros en alguna edición (poco fiables): {', '.join(pocos)}")
        if avisos:
            salida["avisos"] = avisos
        return _json(salida)
    except Exception as e:
        return _error(f"Regresión: {e}")


# Variables del XGBoost: columna shap_<variable> -> nombre legible y columna de alumnos_modelo con su valor
NOMBRES_SHAP = {**VARIABLES_NUMERICAS, "sexo_femenino": "Sexo", "inmigrante": "Origen inmigrante",
                "perfil_categoria": "Perfil del centro", "pais_categoria": "País"}
COLUMNA_VALOR = {"sexo_femenino": "sexo", "inmigrante": "origen", "perfil_categoria": "perfil", "pais_categoria": "CNT"}
CLAVES_ALUMNO = ["CNT", "year", "CNTSCHID", "CNTSTUID"]
TIPOS_CLAVES = {"CNT": str, "year": "int64", "CNTSCHID": "int64", "CNTSTUID": "int64"}   # los de shap_alumnos
DIFERENCIA_NOTABLE = 2   # puntos de importancia entre grupos; por debajo, el patrón se considera igual
GRUPO_PEQUENO = 5000   # por debajo, el XGBoost apenas distingue patrones propios del grupo (03_xgboost_shap)
AVISO_GRUPO_PEQUENO = ("Grupo pequeño: el XGBoost no distingue bien patrones propios de grupos o perfiles pequeños; "
                       "para diferencias entre perfiles, usar la regresión")


def _alumnos_con_shap(year, CNT, **filtros) -> tuple[pd.DataFrame, dict, list[str], list[str]]:
    """Alumnos filtrados con sus SHAP. Devuelve (datos, filtros, avisos, variables del modelo)."""
    df, aplicados, avisos = filtrar_alumnos(year, CNT, **filtros)
    shap = cargar_shap()
    df = df.astype(TIPOS_CLAVES).merge(shap, on=CLAVES_ALUMNO, how="inner")
    variables = [c.removeprefix("shap_") for c in shap.columns if c.startswith("shap_")]
    if any(aplicados.get(c) for c in ("tipo_centro", "ubicacion", "tamano")) and len(df) < GRUPO_PEQUENO:
        avisos.append(AVISO_GRUPO_PEQUENO)
    return df, aplicados, avisos, variables


def _medias_shap(df: pd.DataFrame, variables: list[str], absoluto: bool) -> pd.Series:
    """Media ponderada con SENWT del SHAP (o de su valor absoluto) de cada variable."""
    valores = df[[f"shap_{v}" for v in variables]]
    valores = valores.abs() if absoluto else valores
    pesos = df["SENWT"].to_numpy(dtype=float)
    return pd.Series((valores.to_numpy(dtype=float) * pesos[:, None]).sum(axis=0) / pesos.sum(), index=variables)


def _valor_medio(df: pd.DataFrame, variable: str):
    """Valor típico del grupo en una variable, para el waterfall y la respuesta."""
    columna = COLUMNA_VALOR.get(variable, variable)
    if variable in ("sexo_femenino", "inmigrante"):
        categoria = "Chica" if variable == "sexo_femenino" else "Inmigrante"
        pesos = df["SENWT"]
        return f"{100 * pesos[df[columna] == categoria].sum() / pesos.sum():.0f} % {categoria.lower()}s"
    if variable in ("perfil_categoria", "pais_categoria"):
        unicos = df[columna].astype(str).unique()
        if len(unicos) > 1:
            return "varios"
        return _nombre_perfil(unicos[0]) if variable == "perfil_categoria" else NOMBRES_PAISES.get(unicos[0], unicos[0])
    return _r(media_ponderada(df, columna))


@tool
def get_xgboost_patterns(analisis: Literal["importancia", "contribucion", "forma"] = "importancia",
                         year: Union[int, str] = 2022, variable: str = "", CNT: str = "", tipo_centro: str = "",
                         ubicacion: str = "", tamano: str = "", compromiso: str = "", recursos: str = "",
                         sexo: str = "", origen: str = "", group_by: str = "") -> str:
    """
    What the XGBoost learned, from each student's SHAP values (math points).
    - importancia: points each variable moves the prediction (no direction); compare with group_by or year='all'.
    - contribucion: why a group's predicted score is above or below the edition average (shows a waterfall).
    - forma: mean SHAP per value of 'variable' (e.g. recursos, compromiso).
    - Filters and group_by as in get_math_stats.
    """
    try:
        if group_by and group_by not in GRUPOS:
            return _error(f"group_by '{group_by}' no válido. Usa: {', '.join(GRUPOS)}.")
        filtros = dict(tipo_centro=tipo_centro, ubicacion=ubicacion, tamano=tamano, nivel_compromiso=compromiso,
                       nivel_recursos=recursos, sexo=sexo, origen=origen)
        años = _years(year)
        solo_2022 = analisis == "contribucion" and len(años) > 1   # la contribución es de una sola edición
        if solo_2022:
            year, años = 2022, [2022]
        df, aplicados, avisos, variables = _alumnos_con_shap(year, CNT, **filtros)
        if df.empty:
            return _error("No hay alumnos con esos filtros.")
        salida = {"analisis": analisis, "ambito": aplicados.pop("pais", "panel de 12 países"),
                  "edicion": "2012-2022" if len(años) > 1 else años[0], "filtros": aplicados, "n_alumnos": len(df)}

        if analisis == "importancia":
            columna_grupo = GRUPOS[group_by] if group_by else ("year" if len(años) > 1 else None)
            orden = _medias_shap(df, variables, absoluto=True).sort_values(ascending=False)
            if columna_grupo is None:
                salida["puntos_que_mueve"] = {NOMBRES_SHAP[v]: _r(x) for v, x in orden.items()}
            else:
                grupos = {(int(c) if columna_grupo == "year" else str(c)): _medias_shap(g, variables, absoluto=True)
                          for c, g in df.groupby(columna_grupo, observed=True)}
                salida["puntos_que_mueve"] = {NOMBRES_SHAP[v]: {c: _r(m[v]) for c, m in grupos.items()}
                                              for v in orden.index}
                # Conclusión ya calculada: los modelos pequeños tienden a leer como diferencias unas décimas
                rango = pd.DataFrame(grupos).agg(lambda fila: fila.max() - fila.min(), axis=1)
                notables = [NOMBRES_SHAP[v] for v in orden.index if rango[v] >= DIFERENCIA_NOTABLE]
                salida["diferencias_de_mas_de_2_puntos"] = notables or "ninguna: el patrón es casi igual en todos los grupos"
            salida["nota"] = "media del valor absoluto del SHAP: cuánto mueve la predicción, sin dirección"

        elif analisis == "contribucion":
            medias = _medias_shap(df, variables, absoluto=False)
            medias = medias.reindex(medias.abs().sort_values(ascending=False).index)
            valor_base = float(df["valor_base"].iloc[0])
            salida.update({
                "prediccion_media_edicion": _r(valor_base), "prediccion_media_grupo": _r(media_ponderada(df, "prediccion")),
                "nota_media_grupo": _r(media_ponderada(df, "nota")),
                "contribuciones": {NOMBRES_SHAP[v]: {"puntos": _r(x), "valor_del_grupo": _valor_medio(df, v)}
                                   for v, x in medias.items()},
                "nota": "puntos que suma o resta cada variable frente a la predicción media de la edición",
            })
            from graficos import guardar_waterfall   # import diferido: shap y matplotlib solo cuando se dibuja
            titulo = ", ".join([str(salida["ambito"])] + [str(v) for v in aplicados.values()])
            salida["imagen"] = guardar_waterfall(
                medias.rename(NOMBRES_SHAP), pd.Series({NOMBRES_SHAP[v]: _valor_medio(df, v) for v in medias.index}),
                valor_base, f"{titulo[0].upper() + titulo[1:]} ({años[0]})", max_variables=len(medias))
            if solo_2022:
                avisos.append("La contribución se calcula con una sola edición: se usa 2022")

        else:   # forma
            if not variable:
                return _error("Para la forma de la relación hace falta una variable ('recursos', 'compromiso'...).")
            nombre = _variable_regresion(variable)
            if nombre not in variables:
                return _error(f"'{variable}' no tiene forma: elige una variable del alumno o del centro.")
            columna = COLUMNA_VALOR.get(nombre, nombre)
            valores = df[columna]
            if nombre == "recursos_estudio":   # respuestas parciales reescaladas (4,8...): al entero, como 03_xgboost_shap
                valores = np.floor(valores + 0.5)
            if pd.api.types.is_numeric_dtype(valores) and valores.nunique() > 13:   # continuas: quintiles
                tramos = pd.qcut(valores, 5, duplicates="drop")
                orden = tramos.cat.codes.astype(float)
                etiquetas = tramos.map(lambda t: f"de {_r(t.left)} a {_r(t.right)}", na_action="ignore").astype(object)
            elif nombre == "Indice_Compromiso_Competencia":                         # cada valor = unas conductas
                orden = valores.astype(float).round(2)
                etiquetas = (df["SKIPPING"].map({0: "no falta", 1: "falta a clase"}) + ", "
                             + df["TARDYSD"].map(TEXTO_IMPUNTUALIDAD) + ", "
                             + df["deberes_en_u"].map({1: "deberes en tiempo intermedio",
                                                       0: "deberes con poco o mucho tiempo"})).astype(object)
            elif pd.api.types.is_numeric_dtype(valores):                            # pocos valores: uno a uno
                orden = valores.astype(float).round(2)
                etiquetas = orden.map(lambda v: str(int(v)) if v.is_integer() else str(v)).astype(object)
            else:                                                                   # sexo, origen
                orden, etiquetas = pd.Series(0.0, index=df.index), valores.astype(str).astype(object)
            df = df.assign(_orden=orden.where(valores.notna(), np.inf),
                           _valor=etiquetas.where(valores.notna(), "sin dato"))

            def media(g: pd.DataFrame):
                return _r(float(np.average(g[f"shap_{nombre}"], weights=g["SENWT"])))

            forma = {}
            for (_, valor), g in df.groupby(["_orden", "_valor"], sort=True):
                forma[str(valor)] = ({int(y): media(gy) for y, gy in g.groupby("year")} if len(años) > 1
                                     else {"puntos": media(g), "alumnos": len(g)})
            salida["variable"] = NOMBRES_SHAP[nombre]
            salida["puntos_por_valor"] = forma
            salida["nota"] = "SHAP medio de los alumnos con cada valor: puntos que suma o resta tenerlo"
            if nombre == "Indice_Compromiso_Competencia":
                avisos.append("Cada fila es una combinación de las tres conductas, ordenadas de menor a mayor índice: "
                              "faltar a clase pesa más que los deberes, así que el orden del índice no es el de la nota")

        if avisos:
            salida["avisos"] = avisos
        return _json(salida)
    except Exception as e:
        return _error(f"XGBoost: {e}")


# ==============================================================================
# LISTA DE HERRAMIENTAS EXPORTADAS
# ==============================================================================
pisa_tools = [
    # a) Estadísticas descriptivas en vivo
    get_math_stats, calculate_correlation,
    # b) Contexto estático
    get_context_chart, get_methodology,
]
# c) Resultados de los modelos: cada herramienta entra si existe su tabla
if regresion_disponible():
    pisa_tools.append(get_regression_weights)
if shap_disponible():
    pisa_tools.append(get_xgboost_patterns)
