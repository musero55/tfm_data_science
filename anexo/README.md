# Anexo

Trabajo que ya no forma parte de la línea principal del TFM, pero que se conserva porque explica decisiones tomadas por el camino y puede citarse en la memoria. La línea principal está en `notebooks/fase_1_data_ingestion_eda/`, `src/` y `data/intermediate/` (ver `data/intermediate/README.md`).

Cada carpeta guarda sus notebooks junto con los datos y figuras que generan. Los notebooks están a la misma profundidad que los de `notebooks/`, así que `../../` sigue siendo la raíz del proyecto: leen `data/raw/` y `src/` como antes, y sus propias tablas y figuras en `datos/`, `modelos/` y `figuras/`.

## `exploratorio_33_paises/`: el bloque exploratorio

Se hizo con los 33 países de la OCDE, antes de elegir el tema. Su conclusión llevó al enfoque actual: las variables de país (presupuesto y macroeconomía) cambian poco entre ediciones y la mayor parte de la variación de la nota está dentro de cada país, así que el TFM estudia variables del alumno.

| Notebook | Qué estudia | Genera |
|---|---|---|
| `01_EDA_por_edicion` | Calidad de las tablas maestras y nulos por edición; comparación de los índices originales con los reescalados (`_TREND`) | `figuras/01_comparativa_original_Vs_Trend.png` |
| `01_EDA_presupuestos` | Gasto educativo, salarios y renta por país y edición, y su relación con la nota; perfiles de gasto con K-means | `datos/presupuesto_pais_edicion.parquet`, `datos/presupuesto_serie_anual.parquet` y siete figuras |
| `01_EDA_factores_macroeconomicos` | Indicadores macroeconómicos del Banco Mundial y su relación con la nota; perfiles con K-means | `datos/macro_pais_edicion.parquet` y cuatro figuras |

Usan `src/presupuesto_pais_edicion.py` y `src/perfiles_kmeans.py`, que siguen en `src/`: el primero lo usa también `01_EDA_equidad`, de la línea principal.

Dos figuras se han renombrado para corregir erratas: `01_cambios_variables_pais_edicion.png` (antes `vairbles`) y `01_componentes_principales_macroeconomicos.png` (antes `pricipales`). Los notebooks ya usan los nombres nuevos.

`01_EDA_notas_asignatura` y `01_EDA_equidad` también formaban parte de este bloque, pero siguen en la línea principal porque generan tablas que usa el análisis.

## `modelo_por_estrato/`: el primer enfoque de modelado

Una fila por estrato (país, edición y estrato de PISA), con variables de país y medias de los alumnos del estrato, y XGBoost con SHAP para matemáticas, lectura y ciencias. Se sustituyó por el modelo actual, por alumno y edición con el perfil del centro, cuando el TFM pasó a estudiar variables del alumno (la conclusión del bloque exploratorio).

| Notebook | Qué hace | Genera |
|---|---|---|
| `02_data_panel_modelos` | Construye la tabla por estrato | `datos/datos_modelo_estrato.parquet` y los `dataset_pisa_*_llm.pkl` del agente |
| `03_estudio_factores_estratos` | Primer XGBoost de matemáticas | `modelos/X_test_preparado.pkl`, `modelos/modelo_xgb_math_opt.pkl` |
| `03_modelado` | XGBoost de las tres asignaturas, con SHAP | `modelos/modelo_xgb_{math,read,scie}_opt.pkl` y los explicadores SHAP del agente |
| `04_inferencia_matematicas` | Interpretación del modelo de matemáticas | Explicador SHAP del agente |

`datos/pisa_medias_asignaturas_estrato_anual.parquet` son las notas medias por estrato que lee `02_data_panel_modelos`; ya no la genera ningún notebook.

**Estos notebooks no se pueden volver a ejecutar enteros.** Se conservan con sus salidas como registro:

* `02_data_panel_modelos` lee `data/processed/deberes_cnt.parquet` y `deberes_strat.parquet`, que ya no existen (las horas de deberes salen ahora de `01_EDA_indice_de_compromiso_competencia`).
* Los cuatro escriben o leen en `agente_llm/data_llm/` (`columnas_modelo.pkl`, explicadores SHAP), carpeta que ya no existe.
