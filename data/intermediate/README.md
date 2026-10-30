# Tablas intermedias

Tablas que guardan los notebooks de `notebooks/fase_1_data_ingestion_eda/` y que lee el análisis, casi todas a través de `src/tabla_alumno.py` (`tabla_por_alumno`). Las tablas maestras por edición no están aquí, sino en `data/raw/datos_anuales/` (las genera `00_pipeline_multiedicion_PISA`).

Las tablas del bloque exploratorio y del primer modelo por estrato están en `anexo/`, y las que ya no usa nadie, en `obsoleto/`.

| Carpeta o fichero | La genera | Contenido | La lee |
|---|---|---|---|
| `diccionario_columnas.csv` | `00_pipeline_multiedicion_PISA` | Columnas de las tablas maestras de cada edición y su descripción | `01_EDA_recursos_alumno`, `01_EDA_recursos_centro` |
| `estratos_fijos/pisa{edición}_estratos_fijos.parquet` | `00_Pipeline_generacion_estratos` | Perfil de cada centro (titularidad, ubicación y tamaño) | `tabla_alumno`, `01_EDA_indice_de_compromiso_competencia`, `01_EDA_variables_modelo` |
| `medias_asignaturas_anual/pisa_medias_asignaturas_{centro,pais}_anual.parquet` | `01_EDA_notas_asignatura` | Nota media de matemáticas, lectura y ciencias por centro y por país, con todos los valores plausibles | `01_EDA_equidad`, `01_EDA_indice_de_compromiso_competencia` |
| `escs_anual/pisa_escs_{centro,pais}_anual.parquet` | `01_EDA_equidad` | `ESCS` medio por centro y por país | `tabla_alumno` (composición social del centro) |
| `recursos_anual/pisa_recursos_{centro,pais}_anual.parquet` | `01_EDA_equidad` | Recursos del centro (profesorado, ordenadores, escasez...) por centro y por país | `src/recursos_centro.py`, que calcula los índices de dotación y escasez |
| `recursos_alumno_anual/pisa_recursos_alumno_anual.parquet` | `01_EDA_recursos_alumno` | Subíndices de recursos del hogar por alumno, entre ellos `recursos_estudio` | `tabla_alumno` |
| `compromiso_competencia/pisa{edición}_compromiso_competencia.parquet` | `01_EDA_indice_de_compromiso_competencia` | Índice de compromiso y competencia por alumno, con su tercil de deberes y sus componentes, en el panel común (12 países, cuatro ediciones) | `tabla_alumno` (`panel='común'`) |
| `compromiso_competencia_panel_ampliado/pisa{edición}_compromiso_competencia.parquet` | `01_EDA_indice_de_compromiso_competencia` (sección 5.7.6) | El mismo índice calculado con los 25 países del panel ampliado, sin 2018 | `tabla_alumno` (`panel='ampliado'`) |
