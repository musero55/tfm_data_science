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

matplotlib.use('Agg')

#TODO los mensajes de error y lo que devuelven las herramientas tiene que estar en español o en ingles?
#En que punto se hace la traduccion final ?

# ==========================================================
#  CARGA DE DATOS
# ==========================================================
# Creamos 'temp_plots' dentro de tu entorno actual de trabajo

BASE_DIR = Path(__file__).resolve().parent  # Carpeta 'agente_llm'
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

df_estratos = joblib.load(BASE_DIR/'data_llm'/'df_strat_pisa_llm.pkl')
df_paises= joblib.load(BASE_DIR/'data_llm'/'df_cnt_pisa_llm.pkl')
columnas_del_modelo= joblib.load(BASE_DIR/'data_llm'/'models_columns.pkl')


#Carga los modelos en memoria
MODELS = {k: joblib.load(v) for k, v in MODELS_PATHS.items() if v.exists()}
EXPLAINERS = {k: joblib.load(v) for k, v in SHAP_PATHS.items() if v.exists()}


#TODO df_strat debe ser el dataframe con el que se entreno el modelo (ya codificado) junto con el año y el CNT para cada fila 

######################################################################

#Creamos un dicionario que serirá para normalizar la consulta del LLM
SUBJECT_MAP = {
    "math": "math", "matematicas": "math", "mates": "math", "mathematics": "math",
    "read": "read", "lectura": "read", "literatura": "read", "reading": "read",
    "scie": "scie", "ciencias": "scie", "science": "scie"
}

#Función para normalizar los posibles nombres que asigne el LLM a la asignatura
def subject_normalizer(subject: str) -> str:
    sub = subject.lower().strip() #minusculas y sin espacios
    if sub not in SUBJECT_MAP:
        raise ValueError(f" 'subjet {subject}' not valid . Allowed names: 'math', 'read', 'scie'.")
    return SUBJECT_MAP[sub]


SYSTEM_PROMPT = """
You are an elite research assistant specializing in educational and socioeconomic data analysis for the PISA project and its associated Machine Learning models.

LANGUAGE REQUIREMENT (Strict):
- You must ALWAYS answer and interact with the user in Spanish, using formal, precise academic language.
- All final reports, explanations, and conclusions must be written entirely in Spanish.

CORE RESPONSIBILITIES:
- Analyze educational performance, socioeconomic factors, and ML model inferences.
- Accurately invoke available tools when historical data, simulations, or explanations are needed.
  MACHINE LEARNING DOMAIN SPECIFICATIONS:
- You have access to predictive models and SHAP explainability for 3 distinct subjects:
  1. 'math' -> Mathematics (default if unspecified).
  2. 'read' -> Reading and Literacy.
  3. 'scie' -> Scientific Literacy.
- When calling 'get_feature_importance', 'predict_and_explain_country', or 'simulate_scenario', inspect the user's intent to set the 'subject' parameter accordingly.
- If the user asks for a general overview or mentions a specific subject, adjust the parameter to match.

TEMPORAL CONSTRAINTS:
- The database ONLY contains data for the cycles: 2012, 2015, 2018, and 2022. All countries belong to the OECD.
- For country trends or historical evolution, use 'analyze_temporal_trend'.
- Always convert country names to their 3-letter ISO code (e.g., Spain -> 'ESP').

GUARDRAILS & SCOPE (Strict):
- Only respond to queries directly related to the PISA study, educational policy, student socioeconomic context, or the project's ML models.
- If the user asks about unrelated topics (recipes, unrelated programming, sports, etc.), politely decline in Spanish:
  "Lo siento, únicamente estoy capacitado para responder consultas relacionadas con el estudio PISA y factores socioeducativos relativos a este proyecto."

"""

# ==============================================================================
# HERRAMIENTAS DE CONSULTA
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
        codigo_pais = CNT.upper().strip()
        registro = df_paises[df_paises['CNT'] == codigo_pais]
        
        if registro.empty:
            return f"No se han encontrado registros para {codigo_pais}."
            
        # Cogemos la unica fila y se conviete a una serie de pandas (El dato no se devuelve con el numero de fila)
        fila = registro.iloc[0]
        
        return (#filtramos por el sufijo del año para cada columna
            f"Resultados de {CNT} en PISA {year}:\n"
            f"- Puntuación media Matemáticas: {fila.get(f'media_math_pais_{year}', 'No disponible')}\n"
            f"- Puntuación media Lectura: {fila.get(f'media_read_pais_{year}', 'No disponible')}\n"
            f"- Puntuación media Ciencias: {fila.get(f'media_scie_pais_{year}', 'No disponible')}\n"
            f"- Gasto por alumno (PPP): {fila.get(f'Gasto_Alumno_PPP_{year}', 'No disponible')}\n"
            f"- Tiempo deberes diarios: {fila.get(f'tiempo_deberes_totales_diarios_{year}', 'No disponible')} h\n"
            f"- Índice ESCS medio: {fila.get(f'ESCS_media_pais_anual_{year}', 'No disponible')}\n"
            f"- Dispone personal de apoyo: {fila.get(f'dispone_personal_apoyo_{year}', 'No disponible')}"
        )
    except Exception as e:
        return f"Error al recuperar datos históricos: {str(e)}"

#TODO que pasa cuando no se le especifica el año ? usa 2022 o la media de todos los años ?
@tool
def get_cross_country_stats(column_name: str, operation: str = "mean",top_n: int = 5) -> str:
    """
    Computes comparative cross-country statistics and rankings using the country-level dataset (df_paises).
    
    Parameters:
    - column_name: Exact column name including the year suffix (e.g., 'media_math_pais_2022', 'Gasto_Alumno_PPP_2015', 'tiempo_deberes_totales_diarios_2018').
    - operation: Operation to perform. Options: 'media' (global cross-country average), 'top' (highest scoring countries), 'bottom' (lowest scoring countries).
    - top_n: Number of countries to display when 'top' or 'bottom' is selected (defaults to 5).
    """

    try:
        # Verificar que la columna proporcionada por el LLM exista en el dataset
        if column_name not in df_paises.columns:
            return (f"Error: La columna '{column_name}' no existe en la base de datos. "
                    f"Recuerda usar la nomenclatura correcta con el año al final (ej. '_2022').")
        
        # Filtramos los países que tengan datos nulos en esa columna para no sesgar
        df_valido = df_paises[['CNT', column_name]].dropna()
        
        if df_valido.empty:
            return f"No hay datos válidos para la columna {column_name}."

        if operation == "mean":
            valor = df_valido[column_name].mean()
            return f"La media global (promedio simple entre países) para {column_name} es: {valor:.2f}"
            
        elif operation == "top":
            top_df = df_valido.nlargest(top_n, column_name)
            # Retornar una tabla en formato texto legible para el LLM
            return f"Top {top_n} países para {column_name}:\n{top_df.to_string(index=False)}"
            
        elif operation == "bottom":
            bottom_df = df_valido.nsmallest(top_n, column_name)
            return f"Bottom (últimos) {top_n} países para {column_name}:\n{bottom_df.to_string(index=False)}"
            
        else:
            return "Operación no soportada. Usa 'media', 'top' o 'bottom'."
            
    except Exception as e:
        return f"Error al ejecutar la consulta analítica: {str(e)}"

@tool
def get_cross_variable_stats(filter_column: str, operator: str, filter_value: float, target_column: str, metric: str = "mean") -> str:
    """
    Computes conditional statistics by filtering countries based on one variable and aggregating another.
    
    Parameters:
    - filter_column: Column to apply the filter condition on (e.g., 'Gasto_Alumno_PPP_2022').
    - operator: Strict comparison operator. Must be one of: '>', '<', '>=', '<=', '=='.
    - filter_value: Numeric threshold for the condition (e.g., 50000).
    - target_column: Target column to compute the final metric on (e.g., 'media_math_pais_2022').
    - metric: Mathematical aggregation on target column ('mean', 'median', 'count'). Defaults to 'mean'.
    """
 
    try:
        # Validar existencia de columnas
        if filter_column not in df_paises.columns or target_column not in df_paises.columns:
            return (f"Error: Verifica que ambas columnas existan y terminen con el año correspondiente. "
                    f"Recibido: {filter_column} y {target_column}.")

        #TODO pero esto puede eliminar paises enteros !
        # Trabajar solo con filas donde ambas columnas tengan datos válidos
        df_limpio = df_paises.dropna(subset=[filter_column, target_column])
        
        # Mapeo del operator lógico
        if operator == '>':
            df_filtrado = df_limpio[df_limpio[filter_column] > filter_value]
        elif operator == '<':
            df_filtrado = df_limpio[df_limpio[filter_column] < filter_value]
        elif operator == '>=':
            df_filtrado = df_limpio[df_limpio[filter_column] >= filter_value]
        elif operator == '<=':
            df_filtrado = df_limpio[df_limpio[filter_column] <= filter_value]
        elif operator == '==':
            df_filtrado = df_limpio[df_limpio[filter_column] == filter_value]
        else:
            return f"Error: Operador '{operator}' no soportado. Usa >, <, >=, <=, o =="
        
        paises_encontrados = len(df_filtrado)
        if paises_encontrados == 0:
            return f"Ningún país cumple la condición: {filter_column} {operator} {filter_value}."
            
        # Cálculo de la métrica solicitada
        if metric == "mean":
            resultado = df_filtrado[target_column].mean()
        elif metric == "median":
            resultado = df_filtrado[target_column].median()
        elif metric == "count":
            resultado = paises_encontrados
        else:
            return "Error: Métrica no soportada. Usa 'mean', 'median' o 'count'."
            
        return (
            f"Análisis cruzado completado:\n"
            f"- Condición aplicada: {filter_column} {operator} {filter_value}\n"
            f"- Número de países en este grupo: {paises_encontrados}\n"
            f"- La {metric} de '{target_column}' para este grupo es: {resultado:.2f}"
        )
        
    except Exception as e:
        return f"Error al ejecutar la estadística cruzada: {str(e)}"

    
# ==============================================================================
# HERRAMIENTAS DE MODELADO E INFERENCIA (XGBoost Matemáticas)
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
        codigo_pais = codigo_pais.upper()
        
        modelo = MODELS.get(sub)
        explainer_sub = EXPLAINERS.get(sub)
        if not modelo or not explainer_sub:
            return f"Error: Modelo o explicador SHAP no disponible para la materia '{subject}'."
        
        # Nos quedamos con las variables del pais especifico 
        df_filtrado = df_estratos[df_paises['CNT'] == CNT]
        
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
            f"Análisis Predictivo SHAP para {CNT} (Matemáticas):\n"
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

#TODO Revisar esta funcion (Especificar que los años deben coincidir?)
#TODO Especificar los parámetros de entrada
@tool
def calculate_correlation(a_column: str, b_column: str) -> str:
    """
    Calculates the Pearson correlation coefficient between two numeric indicators across OECD countries.
    Both columns must exist in df_paises and include the cycle year suffix (e.g., 'Gasto_Alumno_PPP_2022').
    """
    try:
        if a_column not in df_paises.columns or b_column not in df_paises.columns:
            return (f"Error: Verifica que ambas columnas existan y tengan el año al final. "
                    f"Recibido: '{a_column}' y '{b_column}'.")
        
        # Eliminar nulos para calcular la correlación de forma segura
        df_valido = df_paises[[a_column, b_column]].dropna()
        paises_validos = len(df_valido)
        
        if paises_validos < 5:
            return "Error: No hay suficientes países con datos conjuntos para calcular una correlación fiable."
        
        corr = df_valido[a_column].corr(df_valido[b_column])
        
        # Interpretación automática para el agente
        fuerza = "fuerte" if abs(corr) >= 0.6 else "moderada" if abs(corr) >= 0.3 else "débil"
        direccion = "positiva (si una sube, la otra sube)" if corr > 0 else "negativa (si una sube, la otra baja)"
        
        return (
            f"La correlación entre {a_column} y {b_column} es: {corr:.3f}.\n"
            f"Interpretación: Existe una relación lineal {fuerza} y {direccion} entre estas variables.\n"
            f"Muestra utilizada: {paises_validos} países con datos válidos en ambas métricas."
        )
    except Exception as e:
        return f"Error al calcular la correlación: {str(e)}"

@tool
def analyze_temporal_trend(CNT: str, variable: str) -> str:
    """
    Extracts the historical evolution across all PISA cycles (2012, 2015, 2018, 2022) for a specific country.
    
    Parameters:
    - CNT: 3-letter ISO country code (e.g., 'ESP', 'MEX').
    - variable: Root metric name without the year suffix. Examples:
      'media_math_pais', 'media_read_pais', 'media_scie_pais', 
      'Gasto_Alumno_PPP', 'tiempo_deberes_totales_diarios', 'ESCS_media_pais_anual'.
    """
    try:
        codigo_pais = CNT.upper()
        registro = df_paises[df_paises['CNT'] == codigo_pais]
        
        if registro.empty:
            return f"Error: No se encontraron registros para el código de país '{codigo_pais}'."
        
        pisa_years = ['2012', '2015', '2018', '2022']
        lineas = []
        
        for year in pisa_years:
            columna = f"{variable}_{year}"
            if columna in registro.columns:
                val = registro.iloc[0][columna]
                if pd.notna(val):
                    val_str = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
                    lineas.append(f"- Edición {year}: {val_str}")
                else:
                    lineas.append(f"- Edición {year}: Dato no disponible")
            else:
                lineas.append(f"- Edición {year}: Métrica no registrada")
                
        return (
            f"Evolución histórica de '{variable}' para {codigo_pais} (2012-2022):\n" +
            "\n".join(lineas)
        )
    except Exception as e:
        return f"Error al consultar la tendencia temporal: {str(e)}"

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
