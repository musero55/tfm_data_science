# Análisis de los Informes PISA - Trabajo Fin de Máster (TFM)

Este repositorio contiene el entorno de desarrollo y los análisis estadísticos del informe PISA utilizando Python, Jupyter Notebooks y VS Code. El proyecto sigue el ciclo de vida estándar de un proyecto de Ciencia de Datos para garantizar un desarrollo limpio, colaborativo y reproducible.

---

## 🚀 Guía de Inicio Rápido

1. **Clonar el repositorio** y abrir la carpeta raíz en VS Code.
2. **Verificar versión de Python**: Debido a las versiones actualizadas de las dependencias (`pandas 3.x` y `numpy 2.5+`), es obligatorio utilizar **Python >= 3.12**. :
   ```bash
   python --version
   ```
3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Descargar los datos crudos**: Debido al gran tamaño de los microdatos de PISA (`.SAS7BDAT`), estos están excluidos del control de versiones (Git). Descarga los ficheros raw desde nuestra carpeta compartida de **Google Drive** y guárdalos localmente en la siguiente ruta exacta:
   `data/raw/CY08MSP_STU_QQQ.SAS7BDAT`

---

## 📁 Estructura del Proyecto

*   `data/raw/`: Datos originales de PISA en formato SAS (NUNCA se modifican ni se suben a Git).
*   `data/intermediate/`: Resultados parciales de cada competencia (en formato de alto rendimiento `.parquet`).
*   `data/processed/`: Dataset o Matriz Maestra final unificada lista para el modelado.
*   `notebooks/`: Flujo de trabajo modular organizado por las fases metodológicas de la Ciencia de Datos.
*   `src/`: Funciones y lógica común reutilizable (ej. cálculo de errores de imputación y medias ponderadas) para evitar código duplicado.

---

## 📐 Ciclo de Vida del Dato y Estructura de Fases (`notebooks/`)

Para mantener el hilo conductor del estudio, los cuadernos se organizan estrictamente en subcarpetas numeradas por fase:

### 📂 `fase_1_data_ingestion_eda/`
*(Ingesta de Datos y Análisis Exploratorio)*
Dedicada a la carga del archivo masivo de PISA, el tratamiento de valores nulos, filtros iniciales y cálculo de variables latentes (variables conceptuales) por competencia.

*   `1.0_comprension_codebook.ipynb`
*   `1.1_eda_y_limpieza_lectura.ipynb`
*   `1.2_eda_y_limpieza_matematicas.ipynb`
*   `1.3_eda_y_limpieza_ciencias.ipynb`

### 📂 `fase_2_feature_engineering_unification/`
*(Ingeniería de Características y Consolidación)*
Fase destinada a la creación de nuevos índices contextuales y a la unificación (*merge*) de las tablas procesadas de la Fase 1 en un único dataset maestro.
*   `2.1_construccion_indices_contexto.ipynb`
*   `2.2_unificacion_matriz_maestra_pisa.ipynb`

### 📂 `fase_3_advanced_analytics_modeling/`
*(Modelado Predictivo y Analítica Avanzada)*
Fase experimental donde se aplican algoritmos de Machine Learning y estadística avanzada (Clustering, Random Forest, etc.) trabajando sobre el dataset ligero de la Fase 2. Cada integrante gestiona su propio estudio de forma independiente.
*   `3.1_clustering_jerarquico_rendimiento.ipynb`
*   `3.2_importancia_variables_random_forest.ipynb`
*   `3.3_clustering_socioeconomico_contexto.ipynb`

### 📂 `fase_4_insights_storytelling/`
*(Visualización, Conclusiones y Reporte)*
Fase final orientada a la generación de gráficos conclusivos de alta calidad (Plotly) y tablas que se incluirán directamente en la memoria escrita del TFM.
*   `4.1_generacion_graficos_memoria.ipynb`
*   `4.2_tablas_anexos_tfm.ipynb`

---

## 📐 Estándares de Nomenclatura (Obligatorios)

### 1. Archivos Jupyter Notebooks
Se utilizará una numeración secuencial dentro de su respectiva fase, seguida de una descripción corta en minúsculas y separada por guiones bajos (*snake_case*).
*   *Ejemplo:* `2.2_unificacion_matriz_maestra_pisa.ipynb`

### 2. Archivos de Datos Intermedios (`data/intermediate/`)
Cualquier tabla intermedia exportada en formato Parquet debe seguir el formato: `pisa_[competencia_o_estudio]_consolidado.parquet`.
*   *Ejemplo:* `pisa_lectura_consolidado.parquet`

### 3. Nombres de Variables y Columnas en Pandas
*   **Códigos Oficiales PISA:** Mantendremos estrictamente las columnas clave de PISA en **mayúsculas** tal cual vienen del archivo original para evitar errores en los `merge` (ej. `CNT` para el país, `W_FSTUWT` para el peso, y los `PV1READ` al `PV10READ`).
*   **Columnas Calculadas:** Cualquier columna propia creada o calculada por el equipo se escribirá obligatoriamente en **minúsculas** y *snake_case*.
    *   *Correcto:* `media_lectura_pisa`, `varianza_imputacion`, `nombre_pais`.
    *   *Incorrecto:* `MediaLectura`, `Error_Imputacion`, `NombrePais`.

---

## 🛠️ Buenas Prácticas para el Trabajo en Equipo

*   **⚠️ Rutas de los Notebooks (`../../`):** Dado que los cuadernos ahora están dentro de subcarpetas, para acceder a las carpetas `data/` o `src/` se deben subir **dos niveles** en el directorio usando la sintaxis de Python: `../../data/...` o `sys.path.append("../../")`.
*   **No modificar notebooks ajenos:** Cada integrante tiene asignado su propio hilo de estudio en la Fase 3. Si necesitas cambios en el procesamiento de un compañero de la Fase 1, comunícaselo previamente o modulariza el cambio mediante una función en la carpeta `src/`.
*   **Subir solo código limpio:** Antes de hacer un `git commit` y subir cambios a GitHub, asegúrate de limpiar la salida de las celdas del notebook (*Kernel -> Clear All Outputs*). Esto evita que el archivo `.ipynb` pese demasiado y previene conflictos de Git por metadatos de ejecución.
*   **Precisión de Datos:** Usar siempre formato `.parquet` para exportar datos intermedios y procesados. Este formato conserva los tipos de datos estrictos y los decimales (`float64`) con precisión exacta.
