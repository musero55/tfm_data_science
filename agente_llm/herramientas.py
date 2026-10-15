import os
import sys
import time
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import shap
from langchain.tools import tool
import gradio as gr
from adjustText import adjust_text
import urllib.parse

import time
from typing import Literal, Union

matplotlib.use('Agg')


# ==========================================================
#  CARGA DE DATOS
# ==========================================================

try:
    # En un script, usa la carpeta donde está este archivo.
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    # En un notebook, __file__ no está definido.
    BASE_DIR = Path.cwd().resolve()

# Si el notebook se ejecuta desde una subcarpeta, busca la carpeta del proyecto
# en los directorios superiores.
if not (BASE_DIR / "data_llm").is_dir():
    BASE_DIR = next(
        (parent for parent in BASE_DIR.parents if (parent / "data_llm").is_dir()),
        BASE_DIR,
    )
DIR_IMAGENES = BASE_DIR / "temp_plots"
DIR_IMAGENES.mkdir(parents=True, exist_ok=True)

#Creamos las rutas de los modelos
MODELS_PATHS = {
    "math": BASE_DIR / "data_llm" / "modelo_xgb_math_opt.pkl",
    "read": BASE_DIR / "data_llm" / "modelo_xgb_read_opt.pkl",
    "scie": BASE_DIR / "data_llm" / "modelo_xgb_scie_opt.pkl"
}

SHAP_PATHS = {
    "math": BASE_DIR / "data_llm" / "shap_explainer_math.pkl",
    "read": BASE_DIR / "data_llm" / "shap_explainer_read.pkl",
    "scie": BASE_DIR / "data_llm" / "shap_explainer_scie.pkl"
}


df_estratos = joblib.load(BASE_DIR/'data_llm'/'dataset_pisa_estrato_llm.pkl')
df_paises= joblib.load(BASE_DIR/'data_llm'/'dataset_pisa_cnt_llm.pkl')
columnas_del_modelo= joblib.load(BASE_DIR/'data_llm'/'columnas_modelo.pkl')


#Carga los modelos en memoria
MODELS = {k: joblib.load(v) for k, v in MODELS_PATHS.items() if v.exists()}
EXPLAINERS = {k: joblib.load(v) for k, v in SHAP_PATHS.items() if v.exists()}

# ==========================================================
#  FUNCIONES AUXILIARES Y NORMALIZADORES
# ==========================================================

PISA_YEARS = [2012, 2015, 2018, 2022]

#Creamos un dicionario que serirá para normalizar la consulta del LLM
SUBJECT_MAP = {
    "math": "math", "matematicas": "math", "matemáticas": "math", "mates": "math", "mathematics": "math",
    "read": "read", "lectura": "read", "literatura": "read", "reading": "read",
    "scie": "scie", "ciencias": "scie", "science": "scie"
}
#Traduccion directa
SUBJECT_NAMES_ES = {
    "math": "Matemáticas",
    "read": "Lectura",
    "scie": "Ciencias"
}

def subject_normalizer(subject: str) -> str:
    """Normaliza el nombre de la asignatura a 'math', 'read' o 'scie'."""
    sub = str(subject).lower().strip()
    if sub not in SUBJECT_MAP:
        raise ValueError(
            f"Asignatura '{subject}' no válida. Usa: 'math', 'read' o 'scie'."
        )
    return SUBJECT_MAP[sub]

def clean_variable_name(variable: str) -> str:
    """
    Si el LLM incluye por error el sufijo del año en el nombre de la variable
    (ej. 'media_math_pais_2022'), lo elimina para evitar duplicar el año ('_2022_2022').
    """
    var_clean = str(variable).strip()
    for y in PISA_YEARS:
        suffix = f"_{y}"
        if var_clean.endswith(suffix):
            return var_clean[:-len(suffix)]
    return var_clean

def get_available_base_variables() -> list[str]:
    """Extrae la lista de nombres de variables base disponibles en df_paises (sin sufijo de año)."""
    base_vars = set()
    for col in df_paises.columns:
        if col in ("CNT", "pais"):
            continue
        base_vars.add(clean_variable_name(col))
    return sorted(base_vars)

def scale_derived_column(df_new: pd.DataFrame, df_old: pd.DataFrame, derived_col: str, num_col: str, den_col: str) -> None:
    """
    Recalcula una columna derivada del tipo (num_col / den_col) manteniendo
    su escala original (tanto si está en proporción 0-1 como en porcentaje 0-100).
    """
    if derived_col not in df_new.columns or num_col not in df_new.columns or den_col not in df_new.columns:
        return

    old_num = df_old[num_col].replace(0, np.nan)
    new_den = df_new[den_col].replace(0, np.nan)
    old_den = df_old[den_col]
    new_num = df_new[num_col]

    factor = (new_num / old_num) * (old_den / new_den)
    recalculated = df_old[derived_col] * factor
    df_new[derived_col] = recalculated.fillna(df_old[derived_col])

#TODO Comprobar si esto es verdad con los datos que tenemos de feaure engeneering 
def recalculate_derived_features(df_orig: pd.DataFrame, df_mod: pd.DataFrame, modified_cols: set[str]) -> list[str]:
    """
    Recalcula automáticamente los ratios e indicadores compuestos en df_mod
    cuando se alteran sus variables base durante una simulación What-If.
    Devuelve la lista de variables derivadas que se han actualizado.
    """
    recalculadas = []

    # 1. Presion_Academica_Neta = tiempo_deberes_totales_diarios / (1 + dispone_personal_apoyo)
    if {"tiempo_deberes_totales_diarios", "dispone_personal_apoyo"} & modified_cols:
        if "Presion_Academica_Neta" in df_mod.columns and "Presion_Academica_Neta" not in modified_cols:
            df_mod["Presion_Academica_Neta"] = (
                df_mod["tiempo_deberes_totales_diarios"] / (1.0 + df_mod["dispone_personal_apoyo"])
            )
            recalculadas.append("Presion_Academica_Neta")

    # 2. brecha_escs_local = escs_estrato - escs_media_pais
    if {"escs_estrato", "escs_media_pais"} & modified_cols:
        if "brecha_escs_local" in df_mod.columns and "brecha_escs_local" not in modified_cols:
            df_mod["brecha_escs_local"] = df_mod["escs_estrato"] - df_mod["escs_media_pais"]
            recalculadas.append("brecha_escs_local")

    # 3. brecha_deberes_local = tiempo_deberes_totales_diarios - media_pais(tiempo_deberes_totales_diarios)
    if "tiempo_deberes_totales_diarios" in modified_cols:
        if "brecha_deberes_local" in df_mod.columns and "brecha_deberes_local" not in modified_cols:
            deberes_media_pais = df_mod["tiempo_deberes_totales_diarios"].mean()
            df_mod["brecha_deberes_local"] = df_mod["tiempo_deberes_totales_diarios"] - deberes_media_pais
            recalculadas.append("brecha_deberes_local")

    # 4. coste_docente_por_alumno_usd_ppp = salario_legal_docente_15_años_experiencia_usd_ppp / alumnos_por_docente_secundaria_primera_etapa
    if {"salario_legal_docente_15_años_experiencia_usd_ppp", "alumnos_por_docente_secundaria_primera_etapa"} & modified_cols:
        col_coste = "coste_docente_por_alumno_usd_ppp"
        if col_coste in df_mod.columns and col_coste not in modified_cols:
            scale_derived_column(
                df_mod, df_orig, col_coste,
                "salario_legal_docente_15_años_experiencia_usd_ppp",
                "alumnos_por_docente_secundaria_primera_etapa"
            )
            recalculadas.append(col_coste)
            modified_cols.add(col_coste)

    # 5. coste_docente_por_alumno_pct_gasto_alumno_secundaria
    if {"coste_docente_por_alumno_usd_ppp", "gasto_alumno_secundaria_usd_ppp"} & modified_cols:
        col_pct_coste = "coste_docente_por_alumno_pct_gasto_alumno_secundaria"
        if col_pct_coste in df_mod.columns and col_pct_coste not in modified_cols:
            scale_derived_column(
                df_mod, df_orig, col_pct_coste,
                "coste_docente_por_alumno_usd_ppp",
                "gasto_alumno_secundaria_usd_ppp"
            )
            recalculadas.append(col_pct_coste)

    # 6. servicios_auxiliares_pct_gasto_alumno_secundaria
    if {"gasto_alumno_secundaria_servicios_auxiliares_usd_ppp", "gasto_alumno_secundaria_usd_ppp"} & modified_cols:
        col_serv = "servicios_auxiliares_pct_gasto_alumno_secundaria"
        if col_serv in df_mod.columns and col_serv not in modified_cols:
            scale_derived_column(
                df_mod, df_orig, col_serv,
                "gasto_alumno_secundaria_servicios_auxiliares_usd_ppp",
                "gasto_alumno_secundaria_usd_ppp"
            )
            recalculadas.append(col_serv)

    # 7. gasto_alumno_secundaria_pct_pib_per_capita
    if {"gasto_alumno_secundaria_usd_ppp", "pib_per_capita_usd_ppp"} & modified_cols:
        col_gasto_pib = "gasto_alumno_secundaria_pct_pib_per_capita"
        if col_gasto_pib in df_mod.columns and col_gasto_pib not in modified_cols:
            scale_derived_column(
                df_mod, df_orig, col_gasto_pib,
                "gasto_alumno_secundaria_usd_ppp",
                "pib_per_capita_usd_ppp"
            )
            recalculadas.append(col_gasto_pib)

    # 8. salario_legal_docente_15_años_experiencia_pct_pib_per_capita
    if {"salario_legal_docente_15_años_experiencia_usd_ppp", "pib_per_capita_usd_ppp"} & modified_cols:
        col_sal_pib = "salario_legal_docente_15_años_experiencia_pct_pib_per_capita"
        if col_sal_pib in df_mod.columns and col_sal_pib not in modified_cols:
            scale_derived_column(
                df_mod, df_orig, col_sal_pib,
                "salario_legal_docente_15_años_experiencia_usd_ppp",
                "pib_per_capita_usd_ppp"
            )
            recalculadas.append(col_sal_pib)

    # 9. ratio_gasto_preprimaria_sobre_secundaria
    if {"gasto_alumno_preprimaria_usd_ppp", "gasto_alumno_secundaria_usd_ppp"} & modified_cols:
        col_ratio_pre = "ratio_gasto_preprimaria_sobre_secundaria"
        if col_ratio_pre in df_mod.columns and col_ratio_pre not in modified_cols:
            scale_derived_column(
                df_mod, df_orig, col_ratio_pre,
                "gasto_alumno_preprimaria_usd_ppp",
                "gasto_alumno_secundaria_usd_ppp"
            )
            recalculadas.append(col_ratio_pre)

    return recalculadas

SYSTEM_PROMPT = """
You are an elite research assistant specializing in educational and socioeconomic data analysis for the PISA project and its associated Machine Learning models.

LANGUAGE REQUIREMENT (Strict):
- You must ALWAYS answer and interact with the user in Spanish, using formal, precise academic language.
- All final reports, explanations, and conclusions must be written entirely in Spanish.

CORE RESPONSIBILITIES:
- Analyze educational performance, socioeconomic factors, and ML model inferences.
- Accurately invoke available tools when historical data, cross-country rankings, correlations, future projections, simulations, or SHAP explanations are needed.

MACHINE LEARNING DOMAIN SPECIFICATIONS:
- You have access to predictive XGBoost models and SHAP explainability for 3 distinct subjects:
  1. 'math' -> Mathematics (default if unspecified).
  2. 'read' -> Reading and Literacy.
  3. 'scie' -> Scientific Literacy.
- When calling 'get_feature_importance', 'predict_and_explain_performance', or 'simulate_scenario', inspect the user's intent to set the 'subject' parameter accordingly.

AVAILABLE BASE VARIABLES (use without year suffix in tool parameters):
- Target scores (only in country dataset):
  'media_math_pais', 'media_read_pais', 'media_scie_pais'
- Socioeconomic & School Context:
  'tiempo_deberes_totales_diarios', 'dispone_personal_apoyo', 'escs_media_pais', 'escs_estrato',
  'Presion_Academica_Neta', 'brecha_escs_local', 'brecha_deberes_local', 'gini', 'pib_per_capita_usd_ppp'
- Expenditure & Educational Resources:
  'gasto_alumno_preprimaria_usd_ppp', 'gasto_alumno_primaria_usd_ppp', 'gasto_alumno_secundaria_usd_ppp',
  'gasto_alumno_secundaria_servicios_auxiliares_usd_ppp', 'gasto_pct_pib_primaria_a_postsecundaria',
  'pct_gasto_publico_en_primaria_a_postsecundaria', 'gasto_publico_total_pct_pib',
  'financiacion_privada_pct_gasto_secundaria', 'financiacion_exterior_pct_gasto_secundaria',
  'salario_legal_docente_15_años_experiencia_usd_ppp', 'alumnos_por_docente_secundaria_primera_etapa',
  'pct_gasto_secundaria_en_personal', 'pct_gasto_secundaria_en_capital',
  'servicios_auxiliares_pct_gasto_alumno_secundaria', 'gasto_alumno_secundaria_pct_pib_per_capita',
  'salario_legal_docente_15_años_experiencia_pct_pib_per_capita', 'coste_docente_por_alumno_usd_ppp',
  'coste_docente_por_alumno_pct_gasto_alumno_secundaria', 'ratio_gasto_preprimaria_sobre_secundaria'

TEMPORAL CONSTRAINTS:
- Historical database cycles: 2012, 2015, 2018, and 2022. All countries belong to the OECD.
- If the user does not specify a year for a query, default to 2022.
- For country trends across 2012-2022, use 'analyze_temporal_trend'.
- For future projections (e.g., PISA 2025), use 'predict_future_trend'.
- Always convert country names to their 3-letter ISO code (e.g., Spain -> 'ESP', Mexico -> 'MEX', Finland -> 'FIN').

GUARDRAILS & SCOPE (Strict):
- Only respond to queries directly related to the PISA study, educational policy, student socioeconomic context, or the project's ML models.
- If the user asks about unrelated topics (recipes, unrelated programming, sports, etc.), politely decline in Spanish:
  "Lo siento, únicamente estoy capacitado para responder consultas relacionadas con el estudio PISA y factores socioeducativos relativos a este proyecto."
"""

# ==============================================================================
# HERRAMIENTAS DE CONSULTA y ESTADISTICA
# ==============================================================================

@tool
def get_country_data(CNT: str, year: int) -> str:
    """
        Retrieve historical PISA educational and socioeconomic indicators for a specific country and edition.
        
        Parameters:
        - CNT: 3-letter ISO country code (e.g., 'ESP', 'MEX', 'FIN').
        - year: PISA cycle year (must be one of: 2012, 2015, 2018, 2022).
        """
    try:
        if year not in PISA_YEARS:
            return f"Error: El año '{year}' no es válido. Los ciclos disponibles son: {PISA_YEARS}."
        
        codigo_pais = CNT.upper().strip()
        registro = df_paises[df_paises['CNT'] == codigo_pais]
        
        if registro.empty:
            return f"No se han encontrado registros para {codigo_pais}."
            
        # Cogemos la unica fila y se conviete a una serie de pandas (El dato no se devuelve con el numero de fila)
        fila = registro.iloc[0]

        #Formatea la salida para tener 2 decimales y la unidad correspondiente de la variable
        def fmt(col_base: str, suffix: str = "") -> str:
            val = fila.get(f"{col_base}_{year}", np.nan)
            if pd.isna(val):
                return "No disponible"
            if isinstance(val, (int, float, np.number)):
                return f"{val:.2f}{suffix}"
            return f"{val}{suffix}"
        
        return (
                f"Resultados e indicadores de {codigo_pais} en PISA {year}:\n"
                f"- Puntuación media Matemáticas: {fmt('media_math_pais', ' pts')}\n"
                f"- Puntuación media Lectura: {fmt('media_read_pais', ' pts')}\n"
                f"- Puntuación media Ciencias: {fmt('media_scie_pais', ' pts')}\n"
                f"- Gasto por alumno en Secundaria (USD PPP): {fmt('gasto_alumno_secundaria_usd_ppp', ' USD PPP')}\n"
                f"- Gasto por alumno en Primaria (USD PPP): {fmt('gasto_alumno_primaria_usd_ppp', ' USD PPP')}\n"
                f"- Gasto público educativo (% del PIB): {fmt('gasto_publico_total_pct_pib', '%')}\n"
                f"- Salario docente (15 años exp, USD PPP): {fmt('salario_legal_docente_15_años_experiencia_usd_ppp', ' USD PPP')}\n"
                f"- Ratio alumnos por docente (Secundaria): {fmt('alumnos_por_docente_secundaria_primera_etapa')}\n"
                f"- Tiempo de deberes diarios: {fmt('tiempo_deberes_totales_diarios', ' h')}\n"
                f"- Dispone de personal de apoyo: {fmt('dispone_personal_apoyo')}\n"
                f"- Índice socioeconómico medio (ESCS): {fmt('escs_media_pais')}\n"
                f"- PIB per cápita (USD PPP): {fmt('pib_per_capita_usd_ppp', ' USD PPP')}\n"
                f"- Índice de Gini: {fmt('gini')}"
                )
    except Exception as e:
        return f"Error al recuperar datos históricos: {str(e)}"

@tool
def get_cross_country_stats( variable: str, year: Union[int, str] = 2022, operation: Literal["mean", "media", "top", "bottom"] = "mean", top_n: int = 5) -> str:
    """
    Computes comparative cross-country statistics and rankings using the country-level dataset (df_paises).
    
    Parameters:
    - column_name: Exact column name including the year suffix (e.g., 'media_math_pais_2022', 'Gasto_Alumno_PPP_2015', 'tiempo_deberes_totales_diarios_2018').
    - operation: Operation to perform. Options: 'mean' (global cross-country average), 'top' (highest scoring countries), 'bottom' (lowest scoring countries).
    - top_n: Number of countries to display when 'top' or 'bottom' is selected (defaults to 5).
    """

    try:
        var_base = clean_variable_name(variable)
        op = operation.lower().strip()

        # Si se especififica una comparativa global -> media histórica de todos los años ('all')
        if str(year).lower().strip() in ("all", "todos", "media", "mean"):
            cols_historicas = [f"{var_base}_{y}" for y in PISA_YEARS if f"{var_base}_{y}" in df_paises.columns]
            if not cols_historicas:
                vars_disponibles = ", ".join(get_available_base_variables())
                return (
                    f"Error: La variable '{var_base}' no existe en la base de datos.\n"
                    f"Variables disponibles: {vars_disponibles}"
                )
            df_temp = df_paises[["CNT"]].copy()
            df_temp["valor_agregado"] = df_paises[cols_historicas].mean(axis=1, skipna=True)
            df_valido = df_temp.dropna(subset=["valor_agregado"]).rename(
                columns={"valor_agregado": f"{var_base}_promedio_2012_2022"}
            )
            col_objetivo = f"{var_base}_promedio_2012_2022"
            etiqueta_temporal = "promedio histórico (2012-2022)"

        else: #Se especifica un año en concreto
            year_int = int(year)
            if year_int not in PISA_YEARS:
                return f"Error: El año '{year}' no es válido. Usa 2012, 2015, 2018, 2022 o 'all'."
            col_objetivo = f"{var_base}_{year_int}"
            if col_objetivo not in df_paises.columns:
                vars_disponibles = ", ".join(get_available_base_variables())
                return (
                    f"Error: La columna '{col_objetivo}' no existe en la base de datos.\n"
                    f"Variables base disponibles: {vars_disponibles}"
                )
            df_valido = df_paises[["CNT", col_objetivo]].dropna()
            etiqueta_temporal = f"edición {year_int}"

        #Hacemos el calculo solocitado 
        if df_valido.empty:
            return f"No hay datos válidos disponibles para '{var_base}' ({etiqueta_temporal})."

        if op in ("mean", "media"):
            valor = df_valido[col_objetivo].mean()
            return (
                f"La media de la OCDE (promedio entre {len(df_valido)} países con datos) "
                f"para '{var_base}' en la {etiqueta_temporal} es: {valor:.2f}"
            )
        elif op == "top":
            top_df = df_valido.nlargest(top_n, col_objetivo)
            return (
                f"Top {top_n} países para '{var_base}' ({etiqueta_temporal}):\n"
                f"{top_df.to_string(index=False)}"
            )
        elif op == "bottom":
            bottom_df = df_valido.nsmallest(top_n, col_objetivo)
            return (
                f"Bottom (últimos) {top_n} países para '{var_base}' ({etiqueta_temporal}):\n"
                f"{bottom_df.to_string(index=False)}"
            )
        else:
            return "Error: Operación no soportada. Usa 'mean', 'top' o 'bottom'."

    except Exception as e:
        return f"Error al ejecutar la consulta comparativa: {str(e)}"

@tool
def get_cross_variable_stats(filter_variable: str, operator: Literal[">", "<", ">=", "<=", "=="], filter_value: float, target_variable: str, year: int = 2022,
metric: Literal["mean", "median", "count"] = "mean") -> str:
    """
        Computes conditional statistics by filtering countries based on one variable and aggregating another in a specific PISA cycle.

        Parameters:
        - filter_variable: Base variable name to apply the filter condition on (e.g., 'gasto_alumno_secundaria_usd_ppp').
        - operator: Strict comparison operator. Must be one of: '>', '<', '>=', '<=', '=='.
        - filter_value: Numeric threshold for the condition (e.g., 12000).
        - target_variable: Base variable name to compute the final metric on (e.g., 'media_math_pais').
        - year: PISA cycle year (2012, 2015, 2018, 2022). Defaults to 2022.
        - metric: Mathematical aggregation on target column ('mean', 'median', 'count'). Defaults to 'mean'.
        """
 
    try:
        if year not in PISA_YEARS:
            return f"Error: El año '{year}' no es válido. Usa uno de: {PISA_YEARS}."
        
        f_base = clean_variable_name(filter_variable)
        t_base = clean_variable_name(target_variable)
        filter_col = f"{f_base}_{year}"
        target_col = f"{t_base}_{year}"

        # Validar existencia de columnas
        if filter_col not in df_paises.columns or target_col not in df_paises.columns:
            vars_disponibles = ", ".join(get_available_base_variables())
            return (
                f"Error: Verifica que ambas variables existan. Recibido: '{f_base}' y '{t_base}'.\n"
                f"Variables disponibles: {vars_disponibles}"
            )
        

        # Filtramos primero solo por la columna de condición para no perder países que cumplen el criterio
        df_con_filtro = df_paises.dropna(subset=[filter_col])

        ops = {
            ">": df_con_filtro[filter_col] > filter_value,
            "<": df_con_filtro[filter_col] < filter_value,
            ">=": df_con_filtro[filter_col] >= filter_value,
            "<=": df_con_filtro[filter_col] <= filter_value,
            "==": df_con_filtro[filter_col] == filter_value,
        }

        if operator not in ops:
            return f"Error: Operador '{operator}' no soportado. Usa >, <, >=, <= o ==."

        df_cumplen = df_con_filtro[ops[operator]]
        paises_cumplen = df_cumplen["CNT"].tolist()
        total_cumplen = len(paises_cumplen)

        if total_cumplen == 0:
            return f"Ningún país cumple la condición: {filter_col} {operator} {filter_value}."

        if metric == "count":
            return (
                f"Análisis condicional ({year}):\n"
                f"- Condición aplicada: {filter_col} {operator} {filter_value}\n"
                f"- Número de países que cumplen la condición: {total_cumplen} ({', '.join(paises_cumplen)})"
            )
        
        # Para 'mean' o 'median', comprobamos cuántos tienen dato válido en target_col
        df_objetivo_valido = df_cumplen.dropna(subset=[target_col])
        paises_validos = df_objetivo_valido["CNT"].tolist()
        excluidos = [p for p in paises_cumplen if p not in paises_validos]

        if df_objetivo_valido.empty:
            return (
                f"Los países {', '.join(paises_cumplen)} cumplen la condición {filter_col} {operator} {filter_value}, "
                f"pero ninguno dispone de datos válidos en '{target_col}'."
            )

        if metric == "mean":
            resultado = df_objetivo_valido[target_col].mean()
            nombre_metrica = "media"
        elif metric == "median":
            resultado = df_objetivo_valido[target_col].median()
            nombre_metrica = "mediana"
        else:
            return "Error: Métrica no soportada. Usa 'mean', 'median' o 'count'."

        nota_excluidos = (
            f"\n- Nota: Se excluyeron del cálculo de '{target_col}' por falta de dato: {', '.join(excluidos)}."
            if excluidos else ""
        )

        return (
            f"Análisis cruzado completado (PISA {year}):\n"
            f"- Condición aplicada: {filter_col} {operator} {filter_value}\n"
            f"- Países que cumplen la condición ({total_cumplen}): {', '.join(paises_cumplen)}\n"
            f"- Países con dato válido en '{target_col}' ({len(paises_validos)}): {', '.join(paises_validos)}"
            f"{nota_excluidos}\n"
            f"- La {nombre_metrica} de '{target_col}' para este grupo es: {resultado:.2f}"
        )

    except Exception as e:
        return f"Error al ejecutar la estadística cruzada: {str(e)}"

@tool
def calculate_correlation(variable_a: str, variable_b: str, year_a: int = 2022, year_b: int = 2022) -> str:
    """
    Calculates the Pearson correlation coefficient between two numeric indicators across OECD countries.
    Allows both same-year correlations (default, year_a = year_b = 2022) and lagged inter-temporal correlations.

    Parameters:
    - variable_a: First base variable name without year suffix (e.g., 'gasto_alumno_secundaria_usd_ppp').
    - variable_b: Second base variable name without year suffix (e.g., 'media_math_pais').
    - year_a: PISA cycle year for variable_a (2012, 2015, 2018, 2022). Defaults to 2022.
    - year_b: PISA cycle year for variable_b (2012, 2015, 2018, 2022). Defaults to 2022.
    """
    try:
        if year_a not in PISA_YEARS or year_b not in PISA_YEARS:
            return f"Error: Los años deben pertenecer a {PISA_YEARS}."

        col_a = f"{clean_variable_name(variable_a)}_{year_a}"
        col_b = f"{clean_variable_name(variable_b)}_{year_b}"

        if col_a not in df_paises.columns or col_b not in df_paises.columns:
            vars_disponibles = ", ".join(get_available_base_variables())
            return (
                f"Error: Verifica los nombres de las variables. Buscado: '{col_a}' y '{col_b}'.\n"
                f"Variables base disponibles: {vars_disponibles}"
            )

        df_valido = df_paises[["CNT", col_a, col_b]].dropna()
        paises_validos = len(df_valido)

        if paises_validos < 5:
            return f"Error: Solo hay {paises_validos} países con datos conjuntos; se requieren al menos 5 para una correlación fiable."

        corr = df_valido[col_a].corr(df_valido[col_b])

        fuerza = "fuerte" if abs(corr) >= 0.6 else "moderada" if abs(corr) >= 0.3 else "débil"
        direccion = "positiva (cuando una aumenta, la otra tiende a aumentar)" if corr > 0 else "negativa (cuando una aumenta, la otra tiende a disminuir)"
        tipo_corr = (
            f"contemporánea (edición {year_a})"
            if year_a == year_b
            else f"intertemporal con retardo ({year_a} vs {year_b})"
        )

        return (
            f"Análisis de Correlación de Pearson ({tipo_corr}):\n"
            f"- Variables analizadas: '{col_a}' y '{col_b}'\n"
            f"- Coeficiente de correlación (r): {corr:.3f}\n"
            f"- Interpretación: Relación lineal {fuerza} y {direccion}.\n"
            f"- Muestra utilizada: {paises_validos} países de la OCDE con datos válidos en ambas métricas."
        )
    except Exception as e:
        return f"Error al calcular la correlación: {str(e)}"

@tool
def analyze_temporal_trend(CNT: str, variable: str) -> str:
    """
    Extracts the historical evolution across all PISA cycles (2012, 2015, 2018, 2022) for a specific country.

    Parameters:
    - CNT: 3-letter ISO country code (e.g., 'ESP', 'MEX', 'FIN').
    - variable: Base metric name without the year suffix (e.g., 'media_math_pais', 'gasto_alumno_secundaria_usd_ppp', 'escs_media_pais').

    """
    try:
        codigo_pais = CNT.upper().strip()
        var_base = clean_variable_name(variable)
        registro = df_paises[df_paises["CNT"] == codigo_pais]

        if registro.empty:
            return f"Error: No se encontraron registros para el código de país '{codigo_pais}'."

        lineas = []
        valores_validos = []

        for y in PISA_YEARS:
            columna = f"{var_base}_{y}"
            if columna not in registro.columns:
                vars_disponibles = ", ".join(get_available_base_variables())
                return (
                    f"Error: La variable '{var_base}' no existe en la base de datos.\n"
                    f"Variables disponibles: {vars_disponibles}"
                )
            val = registro.iloc[0][columna]
            if pd.notna(val):
                val_str = f"{val:.2f}" if isinstance(val, (int, float, np.number)) else str(val)
                lineas.append(f"- Edición {y}: {val_str}")
                if isinstance(val, (int, float, np.number)):
                    valores_validos.append((y, float(val)))
            else:
                lineas.append(f"- Edición {y}: Dato no disponible")

        resumen_cambio = ""
        if len(valores_validos) >= 2:
            y_ini, v_ini = valores_validos[0]
            y_fin, v_fin = valores_validos[-1]
            diff_neta = v_fin - v_ini
            signo = "+" if diff_neta > 0 else ""
            resumen_cambio = f"\nVariación neta observada ({y_ini}-{y_fin}): {signo}{diff_neta:.2f}"

        return (
            f"Evolución histórica de '{var_base}' para {codigo_pais} (2012-2022):\n"
            + "\n".join(lineas)
            + resumen_cambio
        )
    except Exception as e:
        return f"Error al consultar la tendencia temporal: {str(e)}"    

#TODO Herramienta de prueba (OLS o utilizar el modelo para predecir ?)
@tool
def predict_future_trend(CNT: str, variable: str = "media_math_pais", target_year: int = 2025) -> str:
    """
    Projects the future value of an educational score or socioeconomic indicator for a specific country
    (defaulting to the next PISA cycle, 2025) using ordinary least squares (OLS) linear trend projection
    over the historical PISA series (2012, 2015, 2018, 2022).

    Parameters:
    - CNT: 3-letter ISO country code (e.g., 'ESP', 'FIN', 'MEX').
    - variable: Base variable name without year suffix (e.g., 'media_math_pais', 'media_read_pais', 'media_scie_pais', 'gasto_alumno_secundaria_usd_ppp'). Defaults to 'media_math_pais'.
    - target_year: Future year to project (must be > 2022, defaults to 2025 for the next PISA cycle).
    """
    try:
        if target_year <= 2022:
            return "Error: Para consultar años históricos (2012-2022) usa 'get_country_data' o 'analyze_temporal_trend'. El año objetivo debe ser mayor a 2022 (ej. 2025)."

        codigo_pais = CNT.upper().strip()
        var_base = clean_variable_name(variable)
        registro = df_paises[df_paises["CNT"] == codigo_pais]

        if registro.empty:
            return f"Error: No se encontraron registros para el código de país '{codigo_pais}'."

        x_years = []
        y_vals = []

        for y in PISA_YEARS:
            col = f"{var_base}_{y}"
            if col not in registro.columns:
                vars_disponibles = ", ".join(get_available_base_variables())
                return (
                    f"Error: La variable '{var_base}' no existe.\n"
                    f"Variables disponibles: {vars_disponibles}"
                )
            val = registro.iloc[0][col]
            if pd.notna(val) and isinstance(val, (int, float, np.number)):
                x_years.append(float(y))
                y_vals.append(float(val))

        if len(x_years) < 2:
            return (
                f"Error: '{codigo_pais}' solo dispone de {len(x_years)} dato(s) histórico(s) para '{var_base}'. "
                f"Se requieren al menos 2 ediciones para realizar una proyección lineal."
            )

        x_arr = np.array(x_years)
        y_arr = np.array(y_vals)

        # Ajuste lineal por mínimos cuadrados: y = pendiente * año + intercepto
        pendiente, intercepto = np.polyfit(x_arr, y_arr, 1)
        proyeccion = pendiente * target_year + intercepto

        # Cálculo de R^2 para informar de la fiabilidad de la tendencia lineal
        y_pred_hist = pendiente * x_arr + intercepto
        ss_res = np.sum((y_arr - y_pred_hist) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        ultimo_anio = int(x_arr[-1])
        ultimo_valor = y_arr[-1]
        cambio_proyectado = proyeccion - ultimo_valor
        signo_cambio = "+" if cambio_proyectado > 0 else ""
        signo_pendiente = "+" if pendiente > 0 else ""

        serie_texto = ", ".join([f"{int(a)}: {v:.2f}" for a, v in zip(x_arr, y_arr)])

        return (
            f"Proyección de tendencia lineal para {codigo_pais} — Variable '{var_base}' (Objetivo: PISA {target_year}):\n"
            f"- Serie histórica utilizada ({len(x_arr)} ciclos): {serie_texto}\n"
            f"- Último valor registrado ({ultimo_anio}): {ultimo_valor:.2f}\n"
            f"- Valor proyectado para {target_year}: {proyeccion:.2f} ({signo_cambio}{cambio_proyectado:.2f} respecto a {ultimo_anio})\n"
            f"- Ritmo de variación anual (pendiente): {signo_pendiente}{pendiente:.2f} por año ({signo_pendiente}{pendiente * 3:.2f} por ciclo trienal PISA)\n"
            f"- Bondad de ajuste lineal (R²): {r2:.3f}"
        )

    except Exception as e:
        return f"Error al calcular la proyección futura: {str(e)}"
    
# ==============================================================================
# HERRAMIENTAS DE MODELADO E INFERENCIA
# ==============================================================================
#
@tool
def get_feature_importance(subject:str = 'math') -> str:
    """
    Retrieves the global feature importance rankings from the trained ML model for a given subject.
    
    Parameters:
    - subject: Academic subject. Must be one of: 'math' (Mathematics), 'read' (Reading/Literature), 'scie' (Science). Defaults to 'math'.
    """

    try:
        sub = subject_normalizer(subject)
        modelo = MODELS.get(sub)
        if not modelo:
            return f"Error: El modelo para '{subject}' no está disponible actualmente."
        
        #Obtenemos las importancias
        importances = modelo.feature_importances_

        #Asignamos las importancias a las columnas 
        pesos = dict(zip(columnas_del_modelo, importances))

        #Ordenamos las claves, segun su valor en el diccionario ( nos quedamos con las 10 mas relevantes)
        top_cols = sorted(pesos, key=pesos.get, reverse=True)[:10]

        subject_names = {"math": "Matemáticas", "read": "Lectura", "scie": "Ciencias"}

        texto = f"Importancia global de variables en el modelo predictivo de {subject_names[sub]}:\n"

        #Generamos en un bucle la lista de importancias obtenidas para el modelo 
        for i, col in enumerate(top_cols, 1):
            texto += f"{i}. {col}: {pesos[col]:.4f}\n"

        return texto
    
    except Exception as e:
        return f"Error al obtener la importancia de variables: {str(e)}"


@tool
def predict_and_explain_performance(CNT: str, subject:str ='math') -> str:
    """
    Provides the baseline ML predicted subjet score and SHAP feature attribution for a country's CURRENT, REAL profile.
    Use this tool when the user asks:
    - Why a country achieves its current score.
    - Which factors are currently helping or penalizing a country.
    DO NOT use this tool if the user asks for hypothetical changes, policies, or what-if scenarios.

    Parameters:
    - CNT: 3-letter ISO code (e.g., 'ESP', 'FIN', 'MEX').
    - subject: Academic subject. Must be one of: 'math' (Mathematics), 'read' (Reading/Literature), 'scie' (Science). Defaults to 'math'.

    """
  
    try:
        sub = subject_normalizer(subject)
        codigo_pais = CNT.upper()
        
        modelo = MODELS.get(sub)
        explainer_sub = EXPLAINERS.get(sub)
        if not modelo or not explainer_sub:
            return f"Error: Modelo o explicador SHAP no disponible para la materia '{subject}'."
        
    
        # Nos quedamos con las variables del pais especifico 
        #TODO he metido filtrar por año, pero es lo que quiero ? no lo tengo en variable de entrada
        df_filtrado = df_estratos[(df_estratos['CNT'] == CNT) & (df_estratos['year'] == 2022)].copy()
        
        if df_filtrado.empty:
            return f"Error: No encontré datos para el código '{CNT}' en la base estratificada."


        df_modelo = df_filtrado[columnas_del_modelo]

        # Predicción y SHAP values del modelo
        prediccion_media = modelo.predict(df_modelo).mean()
        valores_shap_matriz = explainer_sub.shap_values(df_modelo)
        shap_medio_pais = valores_shap_matriz.mean(axis=0)

        #Valor base esperado -> Valor si el modelo no sabe nada de los datos
        valor_base = explainer_sub.expected_value

        #Chekeamos si el modelo viene en un formato tipo lista o numpy array, en cuyo caso nos quedamos con el primer valor
        if isinstance(valor_base, (list, np.ndarray)):

            valor_base = valor_base[0]

        #Convertimos los valores SHAP medios a una serie de pandas con las columnas como indices con sus respectivos valores
        s_impactos = pd.Series(shap_medio_pais, index=columnas_del_modelo)

        # Cogemos las 3 features que más suman/ restan (valores positivos/negativos más altos/bajos)
        variables_suman = [f"{col} (+{val:.2f})" for col, val in s_impactos[s_impactos > 0].nlargest(3).items()]
        variables_restan = [f"{col} ({val:.2f})" for col, val in s_impactos[s_impactos < 0].nsmallest(3).items()]

        return (
            f"Análisis Predictivo SHAP para {CNT}({sub}):\n"
            f"- Predicción media del modelo: {prediccion_media:.1f} puntos.\n"
            f"- Valor base de referencia : {valor_base:.1f} puntos.\n"
            f"- Top 3 Factores impulsores: {', '.join(variables_suman)}\n"
            f"- Top 3 Factores penalizadores: {', '.join(variables_restan)}\n"
        )
    except Exception as e:
        return f"Error al ejecutar el explicador: {str(e)}"


#TODO Cuando tenga los modelos para el ressto de asignaturas modificar el prompt y el código 
@tool
def simulate_scenario(CNT: str, modifications: dict, subject: str = "math") -> str:
    """
    Simulates a WHAT-IF scenario by altering specific variables for a country and estimating the impact on the selected subject.
    
    Parameters:
    - CNT: 3-letter ISO code (e.g., 'ESP', 'FIN').
    - modifications: Dictionary mapping exact feature names to their new hypothetical values.
      Example: {"Gasto_Alumno_PPP": 100000, "tiempo_deberes_totales_diarios": 2.0}
    - subject: Academic subject. Must be one of: 'math' (Mathematics), 'read' (Reading/Literature), 'scie' (Science). Defaults to 'math'.

    """
    try:
        sub = subject_normalizer(subject)
        codigo_pais = CNT.upper()
        modelo = MODELS.get(sub)
        if not modelo:
            return f"Error: El modelo para '{subject}' no está disponible."
        
        df_filtrado = df_estratos[(df_estratos['CNT'] == codigo_pais) & (df_estratos['year'] == 2022)].copy()
        if df_filtrado.empty:
            return f"Error: No se encontró el país '{codigo_pais}' en la base de 2022."

        #Calculamos la nota original con los datos sin cambiar
        nota_original = modelo.predict(df_filtrado[columnas_del_modelo]).mean()

        #Aplicamos los cambios indicados en modifications
        cambios_texto = []
        for col, nuevo_valor in modifications.items():
            if col in df_filtrado.columns:
                valor_antiguo = df_filtrado[col].mean() #Guardamos el valor antifuo
                df_filtrado[col] = nuevo_valor
                if isinstance(valor_antiguo, (int, float)):
                    cambios_texto.append(f"- {col}: pasó de {valor_antiguo:.2f} a {nuevo_valor}") #Generamos el prompt del cambio
                else:
                    cambios_texto.append(f"- {col}: pasó de '{valor_antiguo}' a '{nuevo_valor}'")
            else:
                return f"Error: La variable '{col}' no existe en la base del modelo. Revisa el nombre exacto."

        #Simulamos la nueva nota en funcion de los datos nuevos introducidos
        nota_simulada = modelo.predict(df_filtrado[columnas_del_modelo]).mean()
        diferencia = nota_simulada - nota_original
        signo = "+" if diferencia > 0 else ""

        #Traducimos al español 
        subject_names = {"math": "Matemáticas", "read": "Lectura", "scie": "Ciencias"}
        nombre_sub = subject_names.get(sub, sub)

        return (
            f"Simulación de escenario para {codigo_pais} ({nombre_sub}):\n\n"
            f"Modificaciones aplicadas:\n" + "\n".join(cambios_texto) + "\n\n"
            f"Resultados del modelo:\n"
            f"- Nota original estimada: {nota_original:.2f} puntos.\n"
            f"- Nota simulada estimada: {nota_simulada:.2f} puntos.\n"
            f"- Impacto neto provocado por el cambio: {signo}{diferencia:.2f} puntos."
        )
        
    except Exception as e:
        return f"Error en la simulación: {str(e)}"


#TODO Faltan las predicciones a futuro, pensar como se haría

# ==============================================================================
# HERRAMIENTAS DE VISUALIZACION
# ==============================================================================

# @tool
# def generate_pisa_plot(codigo_plot: str) -> str:
#     """
#     Genera y guarda una visualización estadística o gráfica explicativa de Machine Learning (SHAP).
#     El LLM debe elegir de forma autónoma el mejor gráfico según la consulta.

#     Entorno predefinido disponible dentro del código:
#     - 'df_pisa_cnt': DataFrame agregado. IMPORTANTE: La columna de países se llama 'CNT'.
#     - 'df_pisa_strat': DataFrame estratificado. IMPORTANTE: La columna de países se llama 'CNT'.
#     - 'columnas_esperadas': Lista con los nombres de las features de entrada del modelo.
#     - 'explainer': Objeto TreeExplainer de SHAP del modelo XGBoost.
#     - 'shap', 'plt', 'sns', 'pd', 'np': Librerías importadas listas para usar.
#     - 'adjust_text': Función de ajuste de etiquetas de texto (si está disponible, úsala para evitar solapamientos).

#     REGLAS ESTRICTAS PARA 'codigo_plot':
#     1. Debe ser código Python ejecutable que construya la figura.
#     2. Si usas funciones de 'shap.plots.*', incluye SIEMPRE el argumento 'show=False'.
#     3. NO uses plt.show().
#     4. NO uses plt.savefig() (el guardado se gestiona internamente).

#     BUENAS PRÁCTICAS DE DISEÑO Y COMPOSICIÓN:
#     - Márgenes dinámicos: Si colocas textos o etiquetas sobre puntos/barras, añade margen con 
#       plt.margins(x=0.15, y=0.15) para evitar que los textos se corten en los bordes.
#     - Etiquetas en puntos (scatter plots):
#       * No etiquetes todos los países si son muchos; selecciona el Top/Bottom o el país de interés.
#       * Si 'adjust_text' no es None, úsalo: 
#         texts = [plt.text(x, y, label) for x, y, label in zip(xs, ys, labels)]
#         adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))
#     - Eje X saturado: Si hay muchos países o categorías en el eje X, rota siempre las etiquetas:
#       plt.xticks(rotation=45, ha='right').
#     - Leyendas: Si la leyenda tapa datos, colócala fuera del gráfico:
#       plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left').
#     - Títulos y ejes: Incluye siempre plt.title(), plt.xlabel() y plt.ylabel() legibles.
#     """
#     try:
#         plt.clf()  # Limpieza del lienzo
#         sns.set_theme(style="whitegrid")  # Estilo limpio y consistente
#         plt.figure(figsize=(10, 5.5))

#         scope_local = {
#             "columnas_esperadas": columnas_esperadas,
#             "df_pisa_cnt": df_pisa_cnt,
#             "df_pisa_strat": df_pisa_strat,
#             "explainer": explainer,
#             "shap": shap,
#             "plt": plt,
#             "sns": sns,
#             "np": np,
#             "pd": pd,
#             "adjust_text": adjust_text
#         }

#         # Ejecución controlada del código generado por el LLM
#         exec(codigo_plot, {}, scope_local)

#         # Nombre único de archivo en disco
#         nombre_archivo = f"plot_{time.time_ns()}.png"
#         ruta_absoluta = DIR_IMAGENES / nombre_archivo
        
#         # Guardado asegurando que ningún elemento quede fuera del encuadre
#         plt.savefig(ruta_absoluta, dpi=180, bbox_inches='tight')
#         plt.close('all')

        
#         ruta_posix = ruta_absoluta.resolve().as_posix()

#         # 3. Codificar ÚNICAMENTE los espacios en blanco a '%20' para no alterar los dos puntos de 'C:'
#         ruta_url = ruta_posix.replace(" ", "%20")

#         # 4. En Windows, Gradio requiere una barra extra delante de la unidad (ej. /file=/C:/Users/...)
#         markdown_imagen = f"\n\n![Gráfico PISA](/gradio_api/file={ruta_posix})\n\n"


#         return (
#                     f"IMAGEN GUARDADA EXITOSAMENTE EN DISCO.\n"
#                     f"NO EJECUTES MÁS CÓDIGO NI LLAMES A NINGUNA HERRAMIENTA.\n"
#                     f"Muestra este Markdown al usuario tal cual en tu respuesta final:\n"
#                     f"{markdown_imagen}"
#                 )

#     except Exception as e:
#         plt.close('all')
#         return f"Error de Python al ejecutar tu código: {str(e)}. Corrige tu código e inténtalo de nuevo."
    
# ==============================================================================
# LISTA DE HERRAMIENTAS EXPORTADAS
# ==============================================================================
pisa_tools = [
    get_country_data, 
    get_cross_country_stats,
    get_cross_variable_stats,
    get_feature_importance,
    predict_and_explain_performance,
    simulate_scenario,
    calculate_correlation,
    analyze_temporal_trend
]
