"""
Casos de prueba del agente PISA para ejecutar_bateria_evaluacion.

Cada caso es una pregunta suelta (el evaluador abre un hilo nuevo por caso):
    id, categoria, pregunta
    herramientas_imprescindibles: herramientas que debe llamar (lista vacía = no debe llamar a ninguna)
    min_llamadas_herramienta: opcional, por defecto tantas como herramientas imprescindibles
    referencia: datos_clave (cifras correctas) y conclusiones_esperadas, que lee el juez

Cifras de agente_pisa/artefactos (coeficientes_regresion.csv, conductas_compromiso.csv, shap_alumnos.parquet,
alumnos_modelo.parquet), agente_pisa/metodologia y data/processed/resultados_robustez.parquet.
"""


CASOS_PERFILES_AUSTRALIA = [
    {
        "id": "perfiles_diferencias_2022",
        "categoria": "Perfiles de centro",
        "pregunta": "¿Hay alguna diferencia significativa entre perfiles de centro?",
        "herramientas_imprescindibles": ["get_regression_weights"],
        "referencia": {
            "datos_clave": [
                "Referencia: público urbano mediano. Con alumnos iguales en todo lo demás, en 2022 se distinguen de "
                "cero: concertado urbano pequeño −11,6 (p = 0,003), privado urbano grande +10,1 (p = 0,036), privado "
                "urbano pequeño −17,9 (p = 0,009), público rural pequeño +5,27 (p = 0,032) y público urbano grande "
                "+7,48 (p = 0,002)",
                "El resto de perfiles no se distinguen de cero (por ejemplo, público rural mediano +3,21, p = 0,27)",
            ],
            "conclusiones_esperadas": [
                "Sí hay diferencias en cinco perfiles; el público urbano grande es la más precisa (p = 0,002)",
                "No llama frágil a la diferencia de +10 del privado urbano grande",
                "No atribuye las diferencias a países concretos: las cifras juntan los 12 países",
            ],
        },
    },
    {
        "id": "perfiles_diferencias_evolucion",
        "categoria": "Perfiles de centro",
        # Salida grande (14 perfiles x 4 ediciones): con ella el modelo de respaldo razonaba en bucle y devolvía
        # una respuesta vacía, y el principal copiaba mal cifras de una fila a otra
        "pregunta": "¿Qué perfiles de centro se distinguen del público urbano mediano en cada edición, de 2012 a 2022?",
        "herramientas_imprescindibles": ["get_regression_weights"],
        "referencia": {
            "datos_clave": [
                "2012: privado urbano grande +14,1 (p = 0,008), privado urbano pequeño −15,4 (p = 0,044), público "
                "rural pequeño +7,41 (p = 0,006)",
                "2015: concertado urbano mediano −8,83 (p = 0,027), privado urbano pequeño −16,2 (p = 0,045), público "
                "urbano pequeño −6,27 (p = 0,011)",
                "2018: privado urbano pequeño −12,7 (p = 0,014)",
                "2022: concertado urbano pequeño −11,6, privado urbano grande +10,1, privado urbano pequeño −17,9, "
                "público rural pequeño +5,27, público urbano grande +7,48",
                "El −12 del concertado urbano pequeño es de 2022 (en 2012 es −12,0 con p = 0,10, no significativo)",
            ],
            "conclusiones_esperadas": [
                "La respuesta no está vacía y da los perfiles de cada edición sin mezclar filas ni ediciones",
                "El privado urbano pequeño está por debajo en las cuatro ediciones",
                "Son diferencias de nivel en cada edición, no saltos de un año a otro",
            ],
        },
    },
    {
        "id": "fragilidad_privado_urbano_grande",
        "categoria": "Robustez",
        "pregunta": "¿Por qué es frágil el resultado de los privados urbanos grandes?",
        "herramientas_imprescindibles": ["get_methodology"],
        "referencia": {
            "datos_clave": [
                "Lo frágil es la subida del coeficiente de los recursos de estudio en ese perfil: de 3,4 en 2012 a "
                "17,6 en 2022 puntos por desviación típica (subida de unos 14, p = 0,001)",
                "Sin Australia la subida baja a 6,8 (p = 0,316) y el coeficiente de los recursos deja de diferir "
                "entre perfiles; el coeficiente no se anula",
                "En 2022, 80 de los 114 centros del perfil son australianos",
            ],
            "conclusiones_esperadas": [
                "Lo frágil es el coeficiente de los recursos, no la diferencia de nota de +10 frente al público urbano "
                "mediano, que no se comprobó quitando países",
                "Depende de Australia porque el perfil es sobre todo de centros australianos",
            ],
        },
    },
    {
        "id": "fragilidad_solo_australia",
        "categoria": "Robustez",
        "pregunta": "¿El resultado frágil de los privados urbanos grandes ocurre solo por Australia, o también por "
                    "otros países?",
        "herramientas_imprescindibles": ["get_methodology"],
        "referencia": {
            "datos_clave": [
                "Sin Australia la subida del coeficiente de los recursos del perfil baja de 14,3 a 6,8 (p = 0,316)",
                "Sin cualquiera de los otros 11 países la subida se mantiene, entre 13,9 y 17,0 (p ≤ 0,003); sin "
                "Reino Unido incluso sube, a 17,0",
                "La lista Australia, Reino Unido, Hungría, Italia, Eslovaquia y Eslovenia es la de otra cautela: la "
                "subida del compromiso de 2012 a 2022",
            ],
            "conclusiones_esperadas": [
                "Sí, solo depende de Australia",
                "No cita Reino Unido, Hungría, Italia, Eslovaquia ni Eslovenia como países de los que dependa",
            ],
        },
    },
    {
        "id": "perfiles_por_pais_no_documentado",
        "categoria": "Robustez",
        "pregunta": "Aparte del privado urbano grande, ¿cambian las diferencias de nota de los demás perfiles de "
                    "centro según los países que se incluyan?",
        "herramientas_imprescindibles": ["get_methodology"],
        "referencia": {
            "datos_clave": [
                "Las diferencias de nota entre perfiles no se repitieron quitando países ni con otras variantes",
                "No hay resultados por país: todas las cifras juntan los 12 países",
            ],
            "conclusiones_esperadas": [
                "No está documentado: no se puede decir si dependen de algún país",
                "No afirma que los demás perfiles sean estables ni que sean frágiles",
            ],
        },
    },
    {
        "id": "premisa_falsa_australia",
        "categoria": "Robustez",
        # Premisa falsa: el resultado sí depende solo de Australia
        "pregunta": "¿En qué cifras te basas para decir que el resultado de los privados urbanos grandes no depende "
                    "solo de Australia?",
        "herramientas_imprescindibles": ["get_methodology"],
        "referencia": {
            "datos_clave": [
                "Sin Australia la subida del coeficiente de los recursos del perfil baja de 14,3 a 6,8 (p = 0,316)",
                "Sin cualquiera de los otros 11 países se mantiene, entre 13,9 y 17,0 (p ≤ 0,003)",
            ],
            "conclusiones_esperadas": [
                "Corrige la premisa: el resultado sí depende solo de Australia",
                "No usa las diferencias de nota entre perfiles como prueba, porque juntan los 12 países y no dicen "
                "nada por país",
            ],
        },
    },
]


CASOS_MEMORIA = [
    {
        "id": "compromiso_vs_recursos",
        "categoria": "Regresión",
        # H1 (memoria 4.2, tabla 4.1)
        "pregunta": "¿Qué se relaciona más con la nota de matemáticas: el compromiso del alumno o sus recursos de "
                    "estudio en casa?",
        "herramientas_imprescindibles": ["get_regression_weights"],
        "referencia": {
            "datos_clave": [
                "2022: el compromiso se asocia a +12,1 puntos por desviación típica y los recursos de estudio a +5,9 "
                "(6,28 puntos por cada uno más de los seis recursos, por ejemplo tener también ordenador para estudiar)",
                "Pesan más que los dos algunos controles: la composición social del centro (+31,8 por desviación "
                "típica) y los libros en casa (+16,8)",
            ],
            "conclusiones_esperadas": [
                "El compromiso pesa aproximadamente el doble que los recursos de estudio",
                "Compara alumnos del mismo país, perfil de centro y origen social; es una asociación, no un efecto causal",
                "Explica el compromiso como conductas (no faltar, ser puntual, deberes) y los recursos como objetos "
                "concretos, no como puntos de un índice",
            ],
        },
    },
    {
        "id": "evolucion_compromiso_recursos",
        "categoria": "Evolución",
        # H2 (memoria 4.2 y 4.6)
        "pregunta": "¿Ha cambiado de 2012 a 2022 la relación del compromiso y de los recursos de estudio con la nota "
                    "de matemáticas?",
        "herramientas_imprescindibles": ["get_regression_weights"],
        "referencia": {
            "datos_clave": [
                "Compromiso, por desviación típica: 10,6 en 2012, 9,8 en 2015, 11,5 en 2018 y 12,1 en 2022; sube "
                "+1,5 de 2012 a 2022 (p = 0,029)",
                "Recursos de estudio, por desviación típica: 5,44 en 2012, 5,62 en 2015, 4,93 en 2018 y 5,90 en "
                "2022; el cambio (+0,46) no se distingue de cero (p = 0,514)",
                "En recursos, el único salto es la subida de su coeficiente en el privado urbano grande (unos 14 "
                "puntos por desviación típica de 2012 a 2022), que depende solo de Australia; en el compromiso no hay "
                "diferencias entre perfiles",
                "La subida del compromiso viene de los deberes (de −0,4 a +5,7 puntos) y deja de distinguirse de "
                "cero al quitar algunos países; faltar a clase y llegar tarde pesan igual en toda la década",
            ],
            "conclusiones_esperadas": [
                "Empieza por la conclusión: en el conjunto de los 12 países, la relación del compromiso y de los "
                "recursos con la nota apenas cambia de 2012 a 2022",
                "No presenta la subida del compromiso como un cambio significativo, un aumento ni un refuerzo: la "
                "atribuye a los deberes, cuya pregunta cambia entre ediciones, y dice que depende de algunos países",
                "La relación de los recursos con la nota es estable en la década",
                "Avisa de que en 2018 los deberes vienen de un cuestionario opcional (64 % de cobertura, 28 % en "
                "Reino Unido)",
            ],
        },
    },
    {
        "id": "faltar_a_clase_causa",
        "categoria": "Conductas",
        # H4, asimetría de conductas (memoria 4.5 a, tabla 4.5) y limitación (1) de 4.6
        "pregunta": "¿Faltar a clase hace que un alumno saque peor nota en matemáticas?",
        "herramientas_imprescindibles": ["get_regression_weights"],
        "referencia": {
            "datos_clave": [
                "Haber faltado a clase en las dos últimas semanas se asocia a unos 17,8 puntos menos en 2012 y 16,9 "
                "en 2022, frente a no haber faltado; el cambio no se distingue de cero (p = 0,508)",
                "Cada paso de impuntualidad, unos 12,2 puntos menos en 2012 y 10,2 en 2022; los deberes en tiempo "
                "intermedio, −0,4 en 2012 y +5,7 en 2022",
                "En 2022 el 34 % de los alumnos ha faltado a clase en las dos últimas semanas",
            ],
            "conclusiones_esperadas": [
                "No afirma que faltar cause la bajada: los datos son de un solo momento y la relación es una "
                "asociación",
                "Faltar a clase es la conducta del compromiso que más se asocia con la nota, y de forma estable",
                "Usa la cifra de la conducta, no el coeficiente del índice de compromiso",
            ],
        },
    },
    {
        "id": "compromiso_por_titularidad",
        "categoria": "Perfiles de centro",
        # H3, primera parte (memoria 4.3, figura 20)
        "pregunta": "¿Rinde igual el compromiso del alumno en un colegio público que en uno concertado o privado?",
        "herramientas_imprescindibles": ["get_regression_weights"],
        "referencia": {
            "datos_clave": [
                "2022, por desviación típica: coeficiente común 12,1; públicos entre 10,9 y 13,9; concertados entre "
                "8,6 (urbano pequeño) y 15,9 (rural grande); privados urbanos entre 11,9 y 15,2",
            ],
            "conclusiones_esperadas": [
                "Sí: el compromiso se asocia a la nota de forma parecida en todos los perfiles de centro",
                "No saca diferencias comparando perfiles de uno en uno, con intervalos que se solapan",
                "No mezcla esta respuesta con la cautela de Australia, que es sobre los recursos del privado urbano "
                "grande",
            ],
        },
    },
    {
        "id": "efecto_techo_recursos",
        "categoria": "XGBoost",
        # H4, efecto techo (memoria 4.4, tabla 4.3)
        "pregunta": "¿La nota sube igual con cada recurso de estudio que se añade en casa, o hay un punto a partir "
                    "del cual apenas cambia?",
        "herramientas_imprescindibles": ["get_xgboost_patterns"],
        "referencia": {
            "datos_clave": [
                "2022, puntos frente a la predicción media: 3 recursos −14,5; 4 recursos −6,0; 5 recursos +0,2; "
                "los 6 recursos +2,3",
                "La mayoría ya tiene 5 o 6 de los seis recursos (en 2022, el 57 % los seis y el 27 % cinco)",
            ],
            "conclusiones_esperadas": [
                "Hay efecto techo: la ganancia está en llegar a 4 o 5 recursos y el sexto añade solo unos 2 puntos",
                "La recta de la regresión (unos 6 puntos por recurso) exagera el beneficio del último recurso",
            ],
        },
    },
    {
        "id": "importancia_xgboost_2022",
        "categoria": "XGBoost",
        # Memoria 4.4, tabla 4.2 y figura 22
        "pregunta": "En el modelo de aprendizaje automático, ¿qué variables pesan más en la predicción de la nota "
                    "en 2022?",
        "herramientas_imprescindibles": ["get_xgboost_patterns"],
        "referencia": {
            "datos_clave": [
                "Puntos que mueve cada variable en 2022: composición social del centro 21,7; país 15,2; libros en "
                "casa 14,3; sexo 9,6; compromiso 9,5; ocupación de los padres 7,8",
                "Recursos de estudio 3,0, perfil del centro 3,1 y dotación del centro 3,5",
            ],
            "conclusiones_esperadas": [
                "El compromiso mueve unas tres veces más que los recursos de estudio",
                "La importancia dice cuántos puntos mueve una variable, sin dirección: no es un coeficiente",
            ],
        },
    },
    {
        "id": "xgboost_vs_regresion",
        "categoria": "XGBoost",
        # Conclusión 2 de la memoria (1.4 y 4.4). El R² solo está en la nota de modelos
        "pregunta": "¿Predice mucho mejor la nota el XGBoost que la regresión lineal?",
        "herramientas_imprescindibles": ["get_methodology"],
        "referencia": {
            "datos_clave": [
                "R² con centros no vistos al entrenar: 0,31 la regresión y 0,34 el XGBoost en 2022 (0,35 y 0,39 en "
                "2012); la mejora es de 0,02 a 0,03 en todas las ediciones",
                "El error medio por alumno del XGBoost es de unos 59 o 60 puntos; los dos modelos dejan sin explicar "
                "unos dos tercios de las diferencias de nota",
            ],
            "conclusiones_esperadas": [
                "No: apenas mejora, la relación es casi lineal y la regresión recoge casi toda la señal",
                "Lo que añade el XGBoost es la forma: efecto techo de los recursos y más peso de faltar a clase que "
                "de los deberes",
            ],
        },
    },
    {
        "id": "riqueza_vs_libros",
        "categoria": "Regresión",
        # Memoria 1.2 a) y 4.2, resultado (2) de los controles
        "pregunta": "En 2022, ¿se relaciona más con la nota la riqueza de la familia, como los coches o los baños de "
                    "la casa, o los libros que hay en casa?",
        "herramientas_imprescindibles": ["get_regression_weights"],
        "referencia": {
            "datos_clave": [
                "Riqueza del hogar: +0,59 puntos por desviación típica (0,48 por cada coche o baño más), p = 0,199: "
                "no se distingue de cero",
                "Libros en casa: +16,8 puntos por desviación típica (11,8 por cada tramo más de libros), p < 0,001",
            ],
            "conclusiones_esperadas": [
                "Los libros en casa se relacionan mucho más con la nota; la riqueza, prácticamente nada una vez "
                "tenidos en cuenta el origen social, los libros y los recursos de estudio",
                "Es una asociación entre alumnos del mismo país, no un efecto causal",
            ],
        },
    },
    {
        "id": "por_que_no_escs",
        "categoria": "Metodología",
        # Memoria 3.5, desglose del origen social
        "pregunta": "¿Por qué el estudio no usa el índice socioeconómico oficial de la OCDE como control?",
        "herramientas_imprescindibles": ["get_methodology"],
        "referencia": {
            "datos_clave": [
                "El índice de la OCDE combina la ocupación y los estudios de los padres con las posesiones del hogar, "
                "que incluyen los propios recursos de estudio (habitación propia, ordenador, internet, libros)",
                "En su lugar se usan sus piezas por separado: ocupación y estudios de los padres, riqueza (coches y "
                "baños) y libros en casa; la composición social del centro se calcula solo con ocupación y estudios",
            ],
            "conclusiones_esperadas": [
                "Controlar por ese índice sería controlar en parte por la propia variable de interés y le restaría "
                "efecto (sobrecontrol)",
                "No dice que el índice de la OCDE sea erróneo en general",
            ],
        },
    },
    {
        "id": "compromiso_ha_empeorado",
        "categoria": "Evolución",
        # Memoria 4.1 c) (figura 12): la bajada de 2012 a 2015 refleja en parte más faltas y retrasos (nota de
        # compromiso, «Limitaciones»); la de después, sobre todo cómo se pregunta por los deberes
        "pregunta": "¿Ha empeorado el compromiso de los alumnos entre 2012 y 2022?",
        "herramientas_imprescindibles": ["get_math_stats", "get_methodology"],
        "referencia": {
            "datos_clave": [
                "Media del índice de compromiso: +0,21 en 2012, +0,02 en 2015, −0,06 en 2018 y −0,09 en 2022",
                "El grupo central de deberes pasa del 36 % de los alumnos en 2012 al 23 % en 2022, porque en 2022 "
                "las respuestas de tiempo tienen pocos valores distintos",
                "De 2012 a 2015, los alumnos que faltaron a clase en las dos últimas semanas pasan del 27,5 % al "
                "36,2 %, y los que llegaron tarde alguna vez, del 34,1 % al 41,1 %",
            ],
            "conclusiones_esperadas": [
                "La bajada desde 2015 se debe sobre todo a cómo se pregunta por los deberes, no a un cambio real de "
                "comportamiento; no afirma sin más que el compromiso haya empeorado",
                "La bajada de 2012 a 2015 sí refleja en parte un cambio de conducta: más faltas y retrasos",
                "La evolución en el tiempo se lee mejor a través de la asistencia (faltar a clase y llegar tarde)",
                "Avisa de la cautela de 2018 (deberes de un cuestionario opcional)",
            ],
        },
    },
]

CASOS_TEST_PISA = CASOS_PERFILES_AUSTRALIA + CASOS_MEMORIA
