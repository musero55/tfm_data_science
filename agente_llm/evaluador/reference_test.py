
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

## =======================Relativo al NOTEBOOK 02_EDA_factores_macroeconomicos========================== ##

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
