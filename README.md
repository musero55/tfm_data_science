# Análisis de los Informes PISA - Trabajo Fin de Máster (TFM)

 Aquí dentro está montado todo el entorno de desarrollo y los análisis estadísticos de los informes PISA. Propongo la siguiente estructura incial para el desarollo del proyecto y que sea algo coherente entre todos

---

## 🚀 Guía de Inicio Rápido

1. **Clonar el repositorio** y abrir la carpeta raíz en VS Code.
2. **Verificar versión de Python**: Trabajo con las últimas versiones de las dependencias (**pandas 3.x** y **numpy 2.5+**), asi que tenemos que usar **Python >= 3.12**. Si no se actualizan estos paquetes, habrá problemas serios de compatibilidad y dependencias con el resto de librerías del proyecto.
   ```bash
   python --version
   ```
3. **Instalar y actualizar dependencias**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```
4. **Descargar los datos crudos**: Como los microdatos de PISA pesan un montón (`.SAS7BDAT`), están fuera del control de versiones (Git). Hay que bajarse los ficheros raw desde la carpeta compartida de **Google Drive** y guardarlos localmente en esta ruta exacta:
   `data/raw/CY08MSP_STU_QQQ.SAS7BDAT`
   
   **Esta la opcion de transformar los datos SAS7BDAT a parquet para que pesen menos**

---

## 📁 Estructura del Proyecto

*   `data/raw/`: Datos originales de PISA en formato SAS (¡OJO! NUNCA se tocan ni se suben a Git).
*   `data/intermediate/`: Resultados parciales, tablas, variables de interes (ej:semilla), dataframes para estudios en comun que lo requieran (en formato `.parquet` por eficiencia de tamaño).
*   `data/processed/`: Datasets finales utilizados para el modelado
*   `notebooks/`: Notebooks particulares, separados en fases de trabajo para que sea modular.
*   `src/`: Funciones y lógica común que podemos reutilizar todos (ej. cálculo de errores de imputación y calculo de medias con PV values) para no duplicar código de forma innecesaria.

---

## 📐 Ciclo de Vida del Dato y Estructura de Fases (`notebooks/`)

para garantizar la modularidad y no perder el hilo, los cuadernos se organizarán en subcarpetas numeradas por fase (Ejemplo de estructura de subcarpeta **A Debatir**):

### 📂 `fase_1_data_ingestion_eda/`
*(Ingesta de Datos y Análisis Exploratorio)*
Aquí se hace la carga del archivo masivo de PISA, el tratamiento de valores nulos, los filtros iniciales y el cálculo de variables latentes (las conceptuales que consideremos) por cada competencia.

*   `1.0_comprension_codebook.ipynb`
*   `1.1_eda_y_limpieza_lectura.ipynb`
*   `1.2_eda_y_limpieza_matematicas.ipynb`
*   `1.3_eda_y_limpieza_ciencias.ipynb`

### 📂 `fase_2_feature_engineering_unification/`
*(Ingeniería de Características y Consolidación)*
Fase pensada para crear los nuevos índices contextuales y hacer el *merge* de las tablas procesadas de la Fase 1 en un único dataset maestro.
*   `2.1_construccion_indices_contexto.ipynb`
*   `2.2_unificacion_matriz_maestra_pisa.ipynb`

### 📂 `fase_3_advanced_analytics_modeling/`
*(Modelado Predictivo y Analítica Avanzada)*
Experimentaciones. Aquí se aplican algoritmos de Machine Learning y estadística trabajando sobre el dataset generado en la Fase 2. Cada integrante gestiona su propio estudio de forma independiente.
*   `3.1_clustering_jerarquico_rendimiento.ipynb`
*   `3.2_importancia_variables_random_forest.ipynb`
*   `3.3_clustering_socioeconomico_contexto.ipynb`

### 📂 `fase_4_insights_storytelling/`
*(Visualización, Conclusiones y Reporte)*
Fase final orientada a sacar los gráficos conclusivos bien pulidos (con Plotly) y las tablas finales que irán directas a la memoria escrita del TFM.
*   `4.1_generacion_graficos_memoria.ipynb`
*   `4.2_tablas_anexos_tfm.ipynb`

---

## 📐 Estándares de Nomenclatura (Obligatorios)

### 1. Archivos Jupyter Notebooks
Se utilizará una numeración secuencial según la fase en la que se esté, seguida de una descripción corta en minúsculas y separada por guiones bajos (*snake_case*).
*   *Ejemplo:* `2.2_unificacion_matriz_maestra_pisa.ipynb`

### 2. Archivos de Datos Intermedios (`data/intermediate/`)
Cualquier tabla intermedia que se exporte en formato Parquet debe seguir este patrón: `pisa_[competencia_o_estudio]_consolidado.parquet`.
*   *Ejemplo:* `pisa_lectura_consolidado.parquet`

### 3. Nombres de Variables y Columnas en Pandas
*   **Códigos Oficiales PISA:** Se mantendrán estrictamente las columnas clave de PISA en **mayúsculas** tal cual vienen del archivo original para no romper los `merge` (ej. `CNT` para el país, `W_FSTUWT` para el peso, y los `PV1READ` al `PV10READ`).
*   **Columnas Calculadas:** Cualquier columna nueva creada o calculada por el equipo se escribirá obligatoriamente en **minúsculas** y *snake_case*.
    *   *Correcto:* `media_lectura_pisa`, `varianza_imputacion`, `nombre_pais`.
    *   *Incorrecto:* `MediaLectura`, `Error_Imputacion`, `NombrePais`.

---

## 🛠️ Buenas Prácticas para el Trabajo en Equipo

*   **⚠️ Rutas de los Notebooks (`../../`):** Como los cuadernos ahora están metidos en subcarpetas, para acceder a `data/` o `src/` se deben subir **dos niveles** en el directorio usando la sintaxis de Python: `../../data/...` o `sys.path.append("../../")`.
*   **No tocar notebooks de otros:** Cada uno tiene asignado su propio hilo de estudio en la Fase 3. Si se necesitan cambios en el procesamiento que hizo un compañero en la Fase 1, se le avisa previamente o se modulariza el cambio mediante una función en la carpeta `src/`.
*   **Subir solo código limpio (¡OJO CON LOS GRÁFICOS!):** Antes de hacer un `git commit` y subir cambios a GitHub, hay que asegurarse de limpiar la salida de las celdas del notebook (*Kernel -> Clear All Outputs*). 
    *   Al hacer esto, **se borrarán todos los textos, tablas e imágenes generadas**.  Los gráficos se guardan internamente como bloques de texto gigantescos  que hacen que el archivo `.ipynb` pesen muchisimo y que Git se vuelva loco provocando conflictos imposibles de solucionar al hacer los *merges*.

    *   *Salvar los gráficos en memoria* El proyecto es reproducible, así que cualquiera puede volver a correr el notebook para verlos de nuevo. Para los gráficos finales, se incluirá código en el notebook para exportarlos directamente como archivos de imagen (`.png`, `.svg` o `.html` para Plotly) en una carpeta local dedicada, evitando así guardarlos dentro del propio cuaderno.
