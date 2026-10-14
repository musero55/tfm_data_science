

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

#Unimos todos los casos de test para poder ser usados por el juez
CASOS_TEST_PISA = [caso_estudio_deberes,caso_estudio_x]