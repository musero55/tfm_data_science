

PROMPT_JUEZ_CASO_ESTUDIO = """Eres un evaluador riguroso de proyectos de analítica educativa y PISA.
Tu labor es comparar el informe/análisis generado por el agente frente a los hallazgos y conclusiones reales obtenidas en la investigación previa.

[PREGUNTA DEL USUARIO]:
{pregunta}

[HALLAZGOS Y DATOS REALES DE REFERENCIA]:
{datos_clave}

[CONCLUSIONES ESPERADAS DE LA INVESTIGACIÓN]:
{conclusiones_esperadas}

[RESPUESTA GENERADA POR EL AGENTE]:
{respuesta_agente}

CRITERIOS DE CALIFICACIÓN:
1. es_fiel_a_referencia: True si el agente identifica los factores clave y llega a las mismas conclusiones cualitativas que la referencia sin inventar datos. False si contradice las conclusiones o alucina cifras.
2. puntuacion_calidad: Puntuación del 1 al 5 en qué tan alineadas y rigurosas están sus conclusiones frente a la investigación.
3. completitud: Puntuación del 1 al 5 sobre si respondió a todas las partes planteadas en la pregunta.
4. motivo: Justificación detallada explicando por qué se considera acierto o fallo y si captó los matices de la investigación.
"""

#PLANTILLA
caso_estudio_x = {
    "id": "",
    "pregunta": (
        "España obtuvo 473 puntos en matemáticas en 2022. "
        "A partir de las variables que penalizan al país en el modelo, ¿qué factores explican esta brecha y qué simulación respalda una mejora?"
    ),
    "herramientas_imprescindibles": [],
    "referencia": {
        "datos_clave": [],
        "conclusiones_esperadas": []
    }
}


caso_estudio_deberes = {
    #Identificación del caso de prueba, id y pregunta pasada al agente_pisa
    "id": "CS1_Impacto_Deberes_Saturacion",
    "pregunta": (
        "Analiza el impacto del tiempo de deberes en la nota de matemáticas para España. "
        "¿Mandar más horas de deberes garantiza un incremento notable en el rendimiento según los modelos?"
    ),
    'herramientas_imprescindibles':[],
    # HALLAZGOS Y CONCLUSIONES
    "referencia": {
        "datos_clave": [
            "El tiempo medio de deberes en España ronda las 2-3 horas diarias.",
            "En el explicador SHAP, el tiempo de deberes aporta un impacto positivo moderado (+2 a +4 puntos).",
            "Variables de contexto (como el ESCS o el PIB) y la autoeficacia tienen un peso predictivo superior."
        ],
        "conclusiones_esperadas": [
            "Existe un umbral de saturación o rendimiento decreciente: aumentar drásticamente las horas no duplica el rendimiento.",
            "El modelo no respalda una relación lineal infinita (más horas != siempre mejor nota).",
            "El agente debe advertir que la política educativa debe priorizar la calidad/metacognición antes que la sobrecarga de tareas."
        ]
    }
}


#TODO Reformular estos casos con nuestros hallazgos

caso_alto_vs_bajo_rendimiento = {
    "id": "CS_03_Diferencias_Alto_Bajo_Rendimiento",
    "pregunta": (
        "Dime las diferencias clave entre países de alto y bajo rendimiento académico en matemáticas, "
        "y qué debería hacer un país de bajo rendimiento para poder igualar a uno de alto según los modelos."
    ),
    "herramientas_imprescindibles": [
        "get_country_stats", 
        "get_feature_importance", 
        "simulate_scenario"
    ],
    "referencia": {
        "datos_clave": [
            "En los rankings de PISA 2022, el top de matemáticas lo lideran países como Japón, Corea y Estonia, mientras que la cola queda por debajo de 400-430 puntos.",
            "En la importancia global de variables del modelo XGBoost, el contexto socioeconómico (ESCS / posesiones en el hogar) concentra el mayor peso predictivo.",
            "El gasto por alumno (Gasto_Alumno_PPP) presenta un efecto de rendimientos decrecientes superado cierto umbral económico.",
            "El tiempo de deberes no muestra una relación lineal simple: los países líderes no son necesariamente los que más horas de deberes asignan."
        ],
        "conclusiones_esperadas": [
            "El agente debe señalar que la brecha principal está condicionada por el entorno socioeconómico y la resiliencia del sistema escolar.",
            "Debe advertir que no existe una solución mágica de una sola variable (no basta solo con subir el gasto o saturar de horas de deberes).",
            "Debe proponer una simulación concreta con 'simulate_scenario' modificando variables de política pública (ej. elevar gasto base o apoyo docente) y cuantificar el salto estimado en puntos.",
            "Las recomendaciones deben orientarse a superar el umbral básico de inversión y reforzar la equidad y el personal de apoyo."
        ]
    }
}

caso_patrones_bajas_notas = {
    "id": "CS_04_Patrones_Paises_Baja_Puntuacion",
    "pregunta": "¿Qué tienen en común los países que tienen bajas notas en PISA?",
    "herramientas_imprescindibles": [
        "get_country_stats", 
        "get_cross_variable_stats"
    ],
    "referencia": {
        "datos_clave": [
            "Los países con menores puntuaciones en matemáticas coinciden con índices ESCS medios considerablemente inferiores.",
            "Presentan un gasto por alumno (Gasto_Alumno_PPP) habitualmente por debajo del umbral de eficiencia de la OCDE.",
            "Baja presencia de personal docente de apoyo escolar en comparación con los países de la OCDE.",
            "Paradoja de tiempo de estudio: pueden presentar tiempos altos de deberes sin que se traduzcan en mejoras de puntuación debido a la falta de soporte pedagógico."
        ],
        "conclusiones_esperadas": [
            "Identificar el grupo inferior de países mediante datos estadísticos y no mediante suposiciones.",
            "Explicar que el bajo rendimiento es de naturaleza multivariante, ligado a vulnerabilidad socioeconómica e infrafinanciación estructural.",
            "Destacar que mandar más deberes de forma aislada no compensa la falta de recursos estructurales en estos países."
        ]
    }
}


# IDEA : "¿El gasto por alumno en 2012 o 2015 se correlaciona con las notas de 2022?

"""Solución propuesta: Documentar bien a_column y b_column en el docstring indicando que por defecto deben llevar el mismo sufijo de año (ej. _2022), pero sin bloquear en el código que tengan años distintos. Si los años de a_column y b_column difieren, podemos añadir una nota automática en el return advirtiendo que es una correlación intertemporal, e incluir también el p-valor (scipy.stats.pearsonr) para que el agente sepa si la correlación es estadísticamente significativa o puro azar."""

#Unimos todos los casos de test para poder ser usados por el juez
CASOS_TEST_PISA = [caso_estudio_deberes,caso_estudio_x]