


**Enfoque central**"¿Hasta qué punto las recetas educativas tradicionales (más inversión económica, mayor carga de deberes, digitalización masiva y gestión autonómica aislada) impactan realmente en el éxito académico, o son meros mitos frente a factores críticos ocultos como la gestión de la atención y la brecha digital?"

1.  El Mito de la Inversión (Fase 2): ¿Más gasto por alumno garantiza mejores resultados en matemáticas o existe un umbral de eficiencia económica?

2. La Paradoja de la Carga Lectiva (Fase 3): ¿Dedicar más horas a los deberes a nivel global mejora el rendimiento o genera saturación?

3.  La Disparidad Territorial (Fase 4): ¿Cómo influyen los factores psicosociales (bullying, sentido de pertenencia) y de qué manera varían según la gestión de cada Comunidad Autónoma?

4. El Factor Crítico y la Distracción Digital (Fase 5): ¿Es la riqueza material y la hiperconectividad (índices HOMEPOS / ICTRES) un recurso educativo o el verdadero saboteador de la concentración y el rendimiento del alumnado?


## 1. Fase 1: Puesta en Contexto (El Panorama General)
* **Objetivo:** Establecer la línea base y la fotografía actual del panorama educativo global para dar sentido al resto de los análisis.
* **Enfoque de Negocio:** Proporcionar al Ministerio una visión clara de dónde se sitúa España frente al resto del mundo antes de profundizar en variables específicas.
* **Plan de Trabajo y Seguimiento:**
  - [ ] **EDA (Análisis Exploratorio):** Análisis descriptivo general de las puntuaciones principales.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho
  - [ ] **Visualización:** Creación de un dashboard o gráficos resumen de situación general.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho

## 2. Fase 2: Inversión vs. Rendimiento (Análisis Económico)
* **Hipótesis de trabajo:** *"Una mayor inversión por alumno no se traduce linealmente en mejores resultados matemáticos; existe un umbral de eficiencia y queremos ver cómo se posiciona España en su evolución histórica."*

* **Plan de Trabajo y Seguimiento:**
  - [ ] **Data Prep:** Búsqueda y cruce de datos históricos de inversión por alumno (este año y anteriores).
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho
  - [ ] **Análisis Comparativo:** Evolución de la inversión de España vs. media internacional.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho
  - [ ] **ML (Árbol de Regresión):** Modelado para medir cuánto influye realmente la inversión económica en los resultados de Matemáticas.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho

## 3. Fase 3: Eficiencia del Estudio (El Impacto de los Deberes)
* **Hipótesis de trabajo:** *"Mandar más horas de deberes a nivel global no garantiza mejores resultados en matemáticas, pudiendo existir una saturación que no aporta valor al aprendizaje."*

* **Plan de Trabajo y Seguimiento:**
  - [ ] **Clustering / Agrupación:** Clasificar a los países según el volumen (alto, medio, bajo) de horas dedicadas a deberes.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho
  - [ ] **EDA:** Comparativa global de horas de deberes vs. puntuación media en PISA.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho
  - [ ] **ML (Regresión):** Análisis predictivo para ver el peso real de la variable "número de deberes" sobre la nota de Matemáticas.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho

## 4. Fase 4: Bienestar y Entorno Social (Análisis Regional en España)
* **Hipótesis de trabajo:** *"El rendimiento académico está fuertemente condicionado por factores psicosociales (bullying, sentido de pertenencia), y su impacto varía significativamente dependiendo de la Comunidad Autónoma."*

* **Plan de Trabajo y Seguimiento:**
  - [ ] **Data Prep:** Filtrar datos exclusivos de España y segmentar por Comunidad Autónoma.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho
  - [ ] **EDA Regional:** Comparativa de los índices de bullying, factores psicológicos y sentido de pertenencia al centro.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho
  - [ ] **Análisis de Impacto:** Identificar qué factor social penaliza más el rendimiento en cada Comunidad Autónoma.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho

## 5. Fase 5: El Factor Crítico (La Paradoja de la Riqueza Material y la Brecha Digital)
* **Hipótesis de trabajo:** *"El índice HOMEPOS no solo mide estatus socioeconómico tradicional, sino que actúa como un proxy de hiperconectividad y hábitos digitales desregulados que saturan la capacidad de concentración y minan el rendimiento."*

* **Plan de Trabajo y Seguimiento:**
  - [ ] **Revisión del Árbol de Regresión (HOMEPOS):** Analizar el modelo base donde HOMEPOS destaca (78.9% de importancia) para interpretar su impacto.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho
  - [ ] **Desglose mediante Regresión Lasso:** Desarmar la multicolinealidad del modelo para ver qué elementos específicos dentro de HOMEPOS están impulsando esta tendencia.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho
  - [ ] **Reincorporación y Análisis de ICTRES:** Recuperar la columna de recursos TIC en el hogar (ICTRES) en el dataframe para analizar su efecto específico por países.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho
  - [ ] **Conclusión Política Final:** Redactar el informe de impacto sobre la gestión de la atención y la distracción digital como culpables del descenso del rendimiento.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho

---

## 📈 Resumen para el equipo (Discord / Tablón de Anuncios)

### royecto Capstone PISA
hilo conductor:

**Los 4 Bloques de Trabajo:**
1. 🌍 **Contexto General:** ¿Dónde estamos parados a nivel global?
2. 💰 **Inversión Económica:** ¿Gastamos bien en España? (Comparativa histórica + Árbol de Regresión sobre resultados de Mates).
3. 📚 **Carga de Deberes:** ¿Más horas en casa = más nota en Mates? (Agrupación de países + Regresión).
4. 🫂 **Factores Sociales en España:** ¿Cómo afectan el bullying y la salud mental según la Comunidad Autónoma?
5. 📱 **El Factor Crítico (Brecha Digital):** ¿Es la hiperconectividad material (HOMEPOS/ICTRES) la verdadera causante de la falta de atención y caída de rendimiento?

👉 **Por favor, revisad el documento principal y cambiad los estados de los checkboxes según vayáis avanzando:**
* `⬜ Pendiente` -> Aún no se ha tocado.
* `⏳ En proceso` -> Estoy trabajando en el código/análisis.
* `✅ Hecho` -> Subido y validado.