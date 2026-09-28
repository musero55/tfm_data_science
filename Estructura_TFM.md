

**Enfoque Central:** A través del análisis de los datos del informe, buscamos entender qué factores (económicos, académicos y sociales) determinan realmente el éxito educativo. Pasamos de las suposiciones a la validación de hipótesis concretas para guiar la toma de decisiones.

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
* **Enfoque de Negocio:** Evaluar la eficiencia del gasto público educativo.
* **Plan de Trabajo y Seguimiento:**
  - [ ] **Data Prep:** Búsqueda y cruce de datos históricos de inversión por alumno (este año y anteriores).
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho
  - [ ] **Análisis Comparativo:** Evolución de la inversión de España vs. media internacional.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho
  - [ ] **ML (Árbol de Regresión):** Modelado para medir cuánto influye realmente la inversión económica en los resultados de Matemáticas.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho

## 3. Fase 3: Eficiencia del Estudio (El Impacto de los Deberes)
* **Hipótesis de trabajo:** *"Mandar más horas de deberes a nivel global no garantiza mejores resultados en matemáticas, pudiendo existir una saturación que no aporta valor al aprendizaje."*
* **Enfoque de Negocio:** Orientar las políticas sobre la carga lectiva fuera del horario escolar.
* **Plan de Trabajo y Seguimiento:**
  - [ ] **Clustering / Agrupación:** Clasificar a los países según el volumen (alto, medio, bajo) de horas dedicadas a deberes.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho
  - [ ] **EDA:** Comparativa global de horas de deberes vs. puntuación media en PISA.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho
  - [ ] **ML (Regresión):** Análisis predictivo para ver el peso real de la variable "número de deberes" sobre la nota de Matemáticas.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho

## 4. Fase 4: Bienestar y Entorno Social (Análisis Regional en España)
* **Hipótesis de trabajo:** *"El rendimiento académico está fuertemente condicionado por factores psicosociales (bullying, sentido de pertenencia), y su impacto varía significativamente dependiendo de la Comunidad Autónoma."*
* **Enfoque de Negocio:** Focalizar las políticas de bienestar emocional y cohesión social de forma territorializada.
* **Plan de Trabajo y Seguimiento:**
  - [ ] **Data Prep:** Filtrar datos exclusivos de España y segmentar por Comunidad Autónoma.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho
  - [ ] **EDA Regional:** Comparativa de los índices de bullying, factores psicológicos y sentido de pertenencia al centro.
    * 📌 Estado: ⬜ Pendiente / ⏳ En proceso / ✅ Hecho
  - [ ] **Análisis de Impacto:** Identificar qué factor social penaliza más el rendimiento en cada Comunidad Autónoma.
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

👉 **Por favor, revisad el documento principal y cambiad los estados de los checkboxes según vayáis avanzando:**
* `⬜ Pendiente` -> Aún no se ha tocado.
* `⏳ En proceso` -> Estoy trabajando en el código/análisis.
* `✅ Hecho` -> Subido y validado.