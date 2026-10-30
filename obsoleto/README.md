# Obsoleto

Ficheros que ya no lee ni genera ningún notebook ni módulo del proyecto. Se apartan aquí para revisarlos antes de borrarlos. Cada uno conserva bajo `obsoleto/` la ruta que tenía, así que restaurar uno es moverlo de vuelta a esa ruta.

Si tras revisarlo nada hace falta, se puede borrar la carpeta entera.

## Horas de deberes antiguas

| Fichero | Qué era | Por qué se puede eliminar |
|---|---|---|
| `data/intermediate/pisa_esfuerzo_alumno_anual.parquet` | Horas de deberes por alumno y edición | Lo generaba `01_EDA_factores_esfuerzo_alumno` (y su copia), que ya no existe. Las horas de deberes salen ahora de `01_EDA_indice_de_compromiso_competencia` (`tiempo_deberes_totales_diarios`, en `data/intermediate/compromiso_competencia/`) y `src/tabla_alumno.py` ya no lo lee |
| `data/intermediate/pisa_esfuerzo_colegio_anual.parquet` | Las mismas horas, medias por centro | Mismo notebook de origen; nadie lo lee |
| `data/intermediate/pisa_esfuerzo_pais_anual.parquet` | Las mismas horas, medias por país | Mismo notebook de origen; nadie lo lee |

## Submuestra de 2022 del inicio del proyecto

Antes de trabajar con las cuatro ediciones completas se hizo una submuestra de centros de 2022, estratificada y con los pesos ajustados. Hoy se usa la muestra completa de cada edición con los pesos originales.

| Fichero | Qué era | Por qué se puede eliminar |
|---|---|---|
| `data/intermediate/pisa_submuestra_estudiantes.parquet` | Alumnos de la submuestra de 2022 con los índices del cuestionario | Ningún notebook la genera ni la lee |
| `data/intermediate/pisa_sub_dataframe_global_nulos.parquet` | La submuestra unida con índices del cuestionario de centro, con nulos | Ningún notebook la genera ni la lee |
| `data/intermediate/escuelas_seleccionadas.pkl` | Lista de centros de la submuestra | Ningún notebook la genera ni la lee |
| `data/intermediate/pesos_senado_alumno_mapeados.pkl` | Pesos de senado ajustados a la submuestra | Ningún notebook la genera ni la lee. `SENWT` se calcula ahora sobre la muestra completa en `src/pisa_pipeline.py` |
| `data/intermediate/README.md` | Documentación de la submuestra y de tablas derivadas que ya no existen | Sustituido por un `data/intermediate/README.md` nuevo con las tablas actuales |

Las funciones de submuestreo de `src/pisa_pipeline.py` (`submuestreo_estratificado`, `leer_estudiantes_submuestra`, `calcular_senwt_adj`...) no se han tocado: `procesar_edicion` las usa cuando se le pasa `fraccion`.

## Otros

| Fichero | Qué era | Por qué se puede eliminar |
|---|---|---|
| `data/intermediate/recursos_centro_anual/pisa_recursos_centro_indices_anual.parquet` | Índices de recursos del centro por centro y edición, del panel común | Lo guardaba la sección 11 de `01_EDA_recursos_centro`, que se ha quitado: nadie lo leía, porque `src/tabla_alumno.py` recalcula los índices con `src/recursos_centro.py` |
| `src/README.txt.txt` | Nota de que `src/` es un paquete de Python | Doble extensión y sin información útil |
| `data/processed/README.txt.txt` | Nota de que la carpeta guarda los datos finales | Doble extensión y sin información útil |
