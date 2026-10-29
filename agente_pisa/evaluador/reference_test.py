
## =======================Relativo al NOTEBOOK 01_EDA_equidad========================== ##
caso_equidad_espana_2022 = {
    "id": "CS_Equidad_Social_Recursos_Espana_2022",
    "pregunta": (
        "¿Cuál es la situación de España en 2022 respecto a la desigualdad social y la dotación de recursos "
        "en centros desfavorecidos en comparación con el resto de países analizados?"
    ),
    "herramientas_imprescindibles": [],
    "referencia": {
        "datos_clave": [
            "Gini de la renta en España (2022): 0,316 (puesto 15 de 31 países, muy próximo a la mediana de 0,309).",
            "Dispersión del ESCS dentro del país: 1,009 (puesto 5 de 33 países, frente a una mediana de 0,920).",
            "Segregación social entre centros: 23,288% (puesto 23 de 33 países, por debajo de la mediana de 26,106%).",
            "Brecha de alumnos por profesor entre centros desfavorecidos y favorecidos: -4,536 (puesto 30 de 31, frente a una mediana de -1,247)."
        ],
        "conclusiones_esperadas": [
            "España presenta un contraste importante en equidad social: aunque su desigualdad de ingresos (Gini) es media y tiene baja segregación escolar entre centros, sufre una de las mayores dispersiones socioeconómicas internas entre familias (ESCS) de los países estudiados.",
            "En cuanto a recursos, la brecha negativa (-4,54) en alumnos por profesor es un indicador positivo de equidad: demuestra que en España los centros más desfavorecidos cuentan con ratios más reducidos (están mejor dotados en personal) que los centros favorecidos, situándose entre los países con mayor compensación o menor saturación en entornos vulnerables."
        ]
    }
}

caso_impacto_desigualdad_rendimiento = {
    "id": "CS_Impacto_Desigualdad_y_Brecha_Recursos_Matematicas",
    "pregunta": (
        "¿Cómo se relacionan la desigualdad social y las brechas de recursos entre centros con el rendimiento en matemáticas en PISA? "
        "¿Están sistemáticamente peor dotados los colegios desfavorecidos?"
    ),
    "herramientas_imprescindibles": [],
    "referencia": {
        "datos_clave": [
            "Correlaciones negativas con matemáticas en variables sociales (estandarizadas por edición): dispersión del ESCS (r = -0,70), índice de Gini (r = -0,59) y segregación social entre centros (r = -0,57). Las tres métricas se solapan moderadamente entre sí (r entre 0,42 y 0,58).",
            "Correlaciones negativas en brechas de recursos: escasez de material (r = -0,55, datos desde 2015), alumnos por profesor (r = -0,47) y escasez de personal (r = -0,41). El tamaño de clase (r = -0,31) y el profesorado certificado (r = -0,24) muestran relaciones más débiles.",
            "En 2022, en 22 de 31 países los centros desfavorecidos tienen menos alumnos por profesor (brecha negativa) y en 29 de 31 tienen clases más pequeñas.",
            "Excepciones en 2022 con brecha positiva (peor dotación en desfavorecidos): 9 países en ratio alumnos/profesor (CHE, CZE, DNK, FRA, GBR, MEX, PRT, TUR, USA) y 2 países en tamaño de clase (GBR, POL)."
        ],
        "conclusiones_esperadas": [
            "Tanto la desigualdad socioeconómica como la peor dotación relativa en centros vulnerables se asocian de forma consistente con un menor rendimiento medio nacional en matemáticas, siendo la dispersión interna del ESCS el factor con la relación negativa más fuerte.",
            "Los centros desfavorecidos no están siempre peor dotados en términos cuantitativos: en la gran mayoría de los países cuentan con clases más pequeñas y menos alumnos por docente (lo cual puede responder a políticas compensatorias o a la presencia de escuelas rurales pequeñas).",
            "El agente debe matizar que la proporción de profesorado certificado arroja un signo contrario al esperado y poco concluyente, y advertir que estas correlaciones entre países son muy estables en el tiempo pero no implican causalidad directa."
        ]
    }
}

## =======================Relativo al NOTEBOOK 01_EDA_factores_macroeconomicos========================== ##

caso_impacto_macroeconomia_rendimiento = {
    "id": "CS_Factores_Macroeconomicos_Rendimiento_Matematicas",
    "pregunta": (
        "¿Qué indicadores macroeconómicos y de desarrollo de un país se asocian más con un buen o mal "
        "rendimiento en matemáticas en PISA? ¿Influye la coyuntura económica de un año concreto?"
    ),
    "herramientas_imprescindibles": [],
    "referencia": {
        "datos_clave": [
            "Indicadores asociados positivamente con la nota en matemáticas (estandarizados por edición): conectividad/usuarios de internet (r = 0,55), inversión en I+D sobre el PIB (r = 0,48), envejecimiento demográfico/población de 65+ años (r = 0,45) y PIB per cápita (r = 0,39).",
            "Indicadores asociados negativamente con el rendimiento: proporción de población infantil de 0-14 años (r = -0,62), peso de la agricultura en el PIB (r = -0,57), inflación anual (r = -0,38) y desempleo juvenil/total (r = -0,30 / -0,29).",
            "Variables de ciclo económico anual como el crecimiento del PIB (r = -0,05) o el comercio exterior tienen una relación prácticamente nula con las notas.",
            "Casi todos los indicadores macro son estructurales y muy estables en el tiempo (correlación entre 2015 y 2022 >= 0,85), salvo la inflación y el crecimiento anual del PIB."
        ],
        "conclusiones_esperadas": [
            "El rendimiento académico nacional está fuertemente ligado al grado de desarrollo estructural y modernización del país (digitalización, inversión científica en I+D y riqueza), mientras que las economías con mayor peso agrario, alta natalidad e inestabilidad de precios obtienen peores resultados medios.",
            "La coyuntura económica puntual (que el PIB crezca más o menos el año del examen) no afecta a las notas de PISA; lo que importa es la estructura socioeconómica de fondo acumulada durante años.",
            "El agente debe advertir que estas relaciones macroeconómicas no son causales por sí solas, sino que reflejan diferencias en el nivel de renta y desarrollo general de los países."
        ]
    }
}

caso_perfiles_paises_mercado_laboral = {
    "id": "CS_Perfiles_Macro_Desempleo_y_Estructura_Productiva",
    "pregunta": (
        "¿Cómo se relacionan el mercado laboral, la demografía y el tipo de economía de los países de la OCDE "
        "con su entorno educativo y gasto público?"
    ),
    "herramientas_imprescindibles": [],
    "referencia": {
        "datos_clave": [
            "El desempleo total y el desempleo juvenil van de la mano (r = 0,95) y se asocian con una menor participación laboral activa y un rendimiento ligeramente inferior en matemáticas (r ~ -0,30).",
            "Existe una marcada oposición entre economías basadas en el sector servicios frente a las industriales (r = -0,79), y entre poblaciones jóvenes (0-14 años) frente a sociedades envejecidas de 65+ años (r = -0,81).",
            "Los países con mayor peso del sector servicios y población más envejecida destinan un porcentaje significativamente mayor de su PIB al gasto sanitario (r = 0,70) y al gasto público total (r = 0,65).",
            "La penetración de internet en la población está estrechamente ligada tanto al PIB per cápita (r = 0,72) como a una mayor participación en el mercado laboral (r = 0,67)."
        ],
        "conclusiones_esperadas": [
            "Los países analizados se dividen en perfiles socioeconómicos muy definidos: por un lado, economías terciarizadas, digitalizadas y demográficamente envejecidas que sostienen un Estado del bienestar amplio (alto gasto público y sanitario) y logran mejores notas en PISA; por otro, economías en transición, más jóvenes, con mayor peso agrario/industrial y mayor inflación, que obtienen peores puntuaciones.",
            "Además del nivel de riqueza, el mercado laboral marca una diferencia clave entre países desarrollados: aquellos que sufren alto paro estructural (especialmente juvenil) presentan un freno adicional frente a los países con alta inserción laboral."
        ]
    }
}

## =======================Relativo al NOTEBOOK 01_EDA_presupuestos========================== ##
caso_presupuesto_renta_y_simulacion_espana = {
    "id": "CS_Presupuesto_Renta_y_Simulacion_Espana",
    "pregunta": (
        "¿Existe una relación directa entre gastar más por alumno en secundaria y obtener mejores notas en matemáticas en PISA? "
        "Analiza la situación de inversión acumulada y salarial de España en 2022 y simula qué impacto tendría incrementar un 15% "
        "el gasto por alumno en secundaria."
    ),
    "herramientas_imprescindibles": [
        "calculate_correlation",
        "get_country_data",
        "simulate_scenario"
    ],
    "referencia": {
        "datos_clave": [
            "Existe una correlación positiva entre el gasto por alumno en secundaria y la nota en matemáticas, pero está fuertemente confundida por la riqueza del país (PIB per cápita), ya que los países más ricos gastan más y rinden más.",
            "En España (2022), el gasto acumulado estimado entre los 6 y los 15 años ronda los 95.400 USD PPP constantes de 2020 (~51.293 USD PPP en primaria 2013-2018 más ~44.106 USD PPP en secundaria 1.ª etapa 2019-2022).",
            "El salario legal docente en España (2022) con 15 años de experiencia equivale a unos 63.287 USD PPP a precios constantes de 2020 (41.197 € corrientes).",
            "Resultado cuantitativo de la herramienta 'simulate_scenario' comparando la puntuación base predicha para España frente a la puntuación simulada con +15% de gasto."
        ],
        "conclusiones_esperadas": [
            "El agente debe advertir que la correlación entre gasto por alumno y rendimiento entre países es exploratoria y no implica un efecto causal directo, pues refleja diferencias estructurales de renta (PIB per cápita) y desarrollo.",
            "Al presentar la simulación del +15% en gasto educativo para España, debe mostrar que el impacto marginal aislado es moderado o presenta rendimientos decrecientes una vez cubierto un umbral básico de financiación.",
            "Debe concluir que cómo se asignan los recursos (composición del gasto y equidad) es tan relevante como el volumen total invertido."
        ]
    }
}

caso_estabilidad_temporal_presupuesto = {
    "id": "CS_Estabilidad_Temporal_y_Varianza_Presupuesto",
    "pregunta": (
        "Al analizar la evolución entre 2012 y 2022 del gasto educativo por alumno, los salarios docentes y las notas de PISA, "
        "¿los países cambian significativamente su inversión de una edición a otra o las diferencias son principalmente estructurales entre países?"
    ),
    "herramientas_imprescindibles": [
        "analyze_temporal_trend",
        "get_cross_country_stats"
    ],
    "referencia": {
        "datos_clave": [
            "De las 19 variables económicas y de rendimiento analizadas en las 4 ediciones (132 filas), una mediana del 91% de la varianza total se debe a diferencias entre países y solo un ~9% a cambios temporales dentro del mismo país.",
            "La correlación intraclase (ICC) mediana es de 0,88 (y la correlación entre los valores de 2015 y 2022 es también de 0,88), lo que reduce las ~122-132 filas disponibles a una muestra efectiva de apenas ~35 observaciones independientes.",
            "Las variables más rígidas en el tiempo son los salarios legales docentes (inicial y con 15 años de experiencia), donde el 99% de la varianza es entre países (ICC = 0,99 y cambio típico de solo 0,09-0,10 desviaciones típicas).",
            "El gasto por alumno en secundaria (89,7% de varianza entre países, ICC = 0,86) y la nota en matemáticas (90,4% entre países, ICC = 0,88) muestran trayectorias casi paralelas con pocos cruces entre países a lo largo de la década."
        ],
        "conclusiones_esperadas": [
            "El presupuesto educativo, los salarios docentes y el rendimiento medio son rasgos estructurales muy estables en cada país: el ranking entre países apenas se altera entre 2012 y 2022.",
            "El agente debe explicar la implicación metodológica clave: dado que las 4 ediciones de un país son casi observaciones repetidas, comparar países mezcla el presupuesto con factores culturales y de renta, mientras que analizar cambios dentro de un mismo país tiene muy poca variación temporal con la que trabajar."
        ]
    }
}

caso_inversion_global_esfuerzo_y_factores_clave = {
    "id": "CS_Inversion_Global_Esfuerzo_PIB_y_Factores_Rendimiento",
    "pregunta": (
        "A nivel global en la OCDE, ¿qué se relaciona más con un alto rendimiento en matemáticas en PISA: "
        "el gasto real por alumno (desde preprimaria hasta secundaria) o el esfuerzo presupuestario en porcentaje del PIB? "
        "Además, según el modelo global, ¿qué peso real tienen el presupuesto y los salarios docentes frente al contexto socioeconómico?"
    ),
    "herramientas_imprescindibles": [
        "get_cross_country_stats",
        "get_cross_variable_stats",
        "calculate_correlation",
        "get_feature_importance"
    ],
    "referencia": {
        "datos_clave": [
            "El gasto real por alumno en paridad de poder adquisitivo (USD PPP constantes de 2020, tanto en secundaria como el acumulado de los 6 a los 15 años y en etapas previas) muestra una correlación positiva clara con las notas de matemáticas en todas las ediciones (2012-2022), muy pareja a la del PIB per cápita.",
            "El esfuerzo relativo (% del PIB dedicado a educación de primaria a postsecundaria o % sobre el gasto público total) varía más entre ediciones (ICC = 0,65) pero por sí solo no garantiza mejores notas si el PIB del país es bajo, ya que lo determinante en recursos es el poder de compra real por estudiante.",
            "En las variables económicas de los países también intervienen el nivel salarial docente (inicial y a los 15 años de experiencia en USD PPP), la ratio de alumnos por profesor y el peso de la financiación privada.",
            "En la importancia global de variables del modelo ('get_feature_importance'), el estatus socioeconómico y cultural (ESCS) y factores del alumnado lideran el peso predictivo por encima de las variables macro de presupuesto."
        ],
        "conclusiones_esperadas": [
            "El agente debe explicar que a escala global importa más la inversión efectiva por alumno (USD PPP) acumulada a lo largo de la escolarización que el mero porcentaje del PIB, aunque esta relación presenta rendimientos decrecientes a partir de cierto umbral de riqueza.",
            "Debe señalar que el gasto educativo está fuertemente ligado al PIB per cápita (los países más ricos invierten más por alumno y pagan mayores salarios docentes en términos reales), por lo que no debe interpretarse como causa única del éxito sin controlar por la riqueza y equidad del país.",
            "Debe concluir, apoyándose en el modelo predictivo, que aunque una financiación sólida y buenos salarios docentes crean la base estructural necesaria, el entorno socioeconómico (ESCS) y lo que ocurre dentro del aula siguen siendo los mayores determinantes del rendimiento en PISA."
        ]
    }
}