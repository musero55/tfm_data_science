# Arquitectura técnica del agente PISA

Agente analítico del TFM sobre la nota en matemáticas de PISA 2012-2022 en los 12 países del panel común (AUS, DNK, ESP, GBR, GRC, HUN, ISL, ITA, KOR, POL, SVK y SVN). Sirve para entender el proyecto y analizar sus resultados: calcula estadísticas descriptivas, explica las decisiones metodológicas y consulta los resultados de la regresión y del XGBoost. Combina una interfaz de exploración manual con un chat con un agente LangGraph que usa modelos de Groq con herramientas (*Tool Calling*).

## 1. Puesta en marcha

El agente no necesita los microdatos de PISA ni volver a entrenar nada: lee solo `artefactos/`, que se entrega ya generado. Todo se lanza desde la raíz del repositorio.

1. **Entorno.** El mismo del resto del proyecto: el `.venv` creado con el `requirements.txt` de la raíz y registrado como kernel `tfm_env` (ver «Puesta en marcha» en el `README.md` principal). Incluye Streamlit, LangGraph, Groq y Jupyter.
2. **Clave de Groq.** Copiar `agente_pisa/.env.example` como `agente_pisa/.env` y poner la clave en `GROQ_API_KEY=` (se crea gratis en [console.groq.com/keys](https://console.groq.com/keys)).
3. **App:**
   ```bash
   .venv/Scripts/python -m streamlit run agente_pisa/app.py
   ```
   Se abre en `http://localhost:8501`. En Linux o macOS la ruta es `.venv/bin/python`. El primer waterfall de cada sesión tarda unos 14 s porque importa `shap`; los siguientes, menos de medio segundo.
4. **Evaluación:** abrir `agente_pisa/evaluador/evaluator.ipynb` con el kernel `tfm_env` y ejecutar todas las celdas. Pregunta los 16 casos de `reference_test.py` con una pausa de 80 s entre casos por la cuota gratuita de Groq, así que tarda unos 25 minutos. Cada ejecución se guarda en `evaluador/resultados/evaluacion_<fecha>_<hora>.csv`, y el notebook se entrega con los resultados de la última.

`artefactos/` solo hay que regenerarlo si cambian los datos, los modelos o las figuras. Lee `data/processed/`, `models/` y `reports/`, que también se entregan:

```bash
.venv/Scripts/python agente_pisa/preparar_artefactos.py
```

## 2. Estructura de ficheros

```
agente_pisa/
├── app.py                    Interfaz Streamlit (streamlit run agente_pisa/app.py)
├── agente.py                 Grafo LangGraph, selección de modelos Groq, ventana de memoria, reintentos y respaldo
├── herramientas.py           Las 6 herramientas del agente, agrupadas en 3 vías
├── datos.py                  Carga de artefactos con st.cache_data
├── graficos.py               Waterfall SHAP de un grupo de alumnos, dibujado en el momento
├── config.py                 Rutas, países, avisos, modelos de Groq y etiquetas legibles
├── preparar_artefactos.py    Genera artefactos/ a partir de data/, models/ y reports/
├── system_prompt.md          Instrucciones del agente (en inglés, respuestas en español)
├── metodologia/              Notas de metodología, una por tema (ver sección 5)
├── .env.example              Plantilla del .env con la clave de Groq
├── artefactos/               Datos que consume la app (se entregan ya generados)
│   ├── alumnos_modelo.parquet       Alumnos en_modelo (298.782), legibles, con la nota de matemáticas
│   ├── coeficientes_regresion.csv   Regresión por edición, traducida al formato del agente (sección 8)
│   ├── shap_alumnos.parquet         SHAP del XGBoost de cada alumno, medio entre valores plausibles
│   ├── metricas_modelos.csv         R², RMSE y MAE de la regresión y del XGBoost por edición
│   ├── conductas_compromiso.csv     Coeficiente de cada conducta del compromiso (2012 y 2022), de data/processed/resultados_robustez.parquet
│   ├── catalogo_graficos.csv        Los 33 gráficos que el agente y la galería pueden mostrar, con su descripción
│   └── img/                         PNG copiados de reports/ (análisis exploratorio y modelado)
├── evaluador/                Juez LLM, 16 casos de referencia y notebook de evaluación (evaluator.ipynb)
│   └── resultados/                  Un CSV por ejecución de la evaluación
└── README_ARQUITECTURA.md    Este documento
```

## 3. Cómo se conectan los ficheros

```
data/processed/tabla_modelo_alumno.parquet + models/ (resultados del modelado) + reports/ (PNG)
        │
        ▼  python agente_pisa/preparar_artefactos.py   (solo si cambian los datos o los modelos)
artefactos/
        │
        ▼  datos.py  (lectura cacheada)
herramientas.py  ◄── metodologia/*.md  (get_methodology)
        │        ──► graficos.py       (waterfall de get_xgboost_patterns)
        ▼
agente.py  (LangGraph: asistente <-> herramientas, con system_prompt.md)
        │
        ├──► app.py              (Streamlit: chat + panel visual)
        └──► evaluador/          (batería de evaluación)
```

`config.py` lo importan todos los módulos. Las herramientas devuelven JSON. La app lee esos resultados en los mensajes de herramienta para mostrar imágenes (clave `imagen`: gráficos del catálogo y waterfalls) y para dibujar gráficas de las estadísticas (salidas de `get_math_stats`), sin que el LLM tenga que repetir nada.

## 4. Decisiones de arquitectura

**Datos de alumno legibles y sin codificar.** `alumnos_modelo.parquet` son los alumnos que entran en el modelo (`en_modelo`), con las variables en sus unidades originales, la nota de matemáticas (media de los valores plausibles) y los `estratos fijos` desglosados en tipo, ubicación y tamaño del centro. Las herramientas filtran con los mismos valores con los que pregunta el usuario («privados», «rurales», «chicas»).

**Los mismos alumnos que el modelo.** Las estadísticas descriptivas, la regresión y el XGBoost usan los mismos 298.782 alumnos, así que las cifras del chat son coherentes entre sí.

**Ponderación con el peso de senado.** Todas las medias usan `SENWT`. Dentro de un país y edición es proporcional a `W_FSTUWT`, así que da la media representativa del país. En el panel cada país pesa casi lo mismo: el `SENWT` suma igual en cada país y edición con todos los alumnos de PISA, pero no todos entran en el modelo y cada país pierde una proporción distinta, así que pesa entre el 6 y el 12 % (en vez del 8,3 %). Darles exactamente el mismo peso cambiaría las medias menos de 1 punto. Se deja así porque es el mismo peso que usan la regresión y el XGBoost de `models/`, y las cifras del agente son coherentes con ellos.

**Resultados de los modelos, no los  modelos.** Hay una regresión y un XGBoost por edición y por valor plausible (más de 40 modelos). La app no carga ninguno: lee tres tablas de resultados que se dejan en `models/` (formato en `models/README.md`). La regresión llega combinada con las reglas de Rubin y los SHAP, promediados entre valores plausibles. Así la app arranca rápido y no depende de la versión de XGBoost.

**La metodología, fuera del System Prompt.** Cómo se construye cada índice, por qué estos países, cómo se leen los modelos o qué resultados son frágiles está en `metodologia/`. `get_methodology` devuelve una nota solo cuando se pregunta por ella, y el System Prompt obliga a contestar con su texto, no con conocimiento general.

**Cautelas dentro de la salida de las herramientas.** Groq salta a menudo al modelo de respaldo (`gpt-oss-20b`), que a veces se salta las notas de metodología y lee mal tablas largas. Por eso las cautelas importantes van en el propio JSON:
- el aviso de 2018 y el de perfiles con menos de 30 centros;
- la conclusión del cambio de 2012 a 2022, la primera de la salida cuando se piden el compromiso o los recursos con 2012 y 2022: en el conjunto de los 12 países la relación apenas cambia y es parecida en todos los perfiles; lo que cambia a primera vista se debe a un país o a un cambio en la pregunta. La subida del compromiso, uno de los dos resultados que no aguantan las comprobaciones de robustez, lleva además su lectura junto al p del cambio, porque suelto el modelo lo contaba como un aumento;
- el otro resultado frágil: el privado urbano grande de 2022, que depende de Australia;
- el aviso de que el XGBoost no ve patrones de grupos pequeños;
- conclusiones ya calculadas, como qué variables difieren más de 2 puntos entre grupos.

**Aviso de 2018.** En 2018 las horas de deberes proceden de un cuestionario opcional (64 % de los alumnos con índice de compromiso; 28 % en GBR). El agente lo recuerda siempre que una respuesta incluye esa edición: las herramientas añaden el aviso y el System Prompt obliga a mencionarlo.

**Caché de datos.** Las tablas se leen con `st.cache_data`, que devuelve una copia por llamada. La app precarga todas las cachés en el hilo principal (`precargar_artefactos`) porque LangGraph ejecuta las herramientas en hilos sin contexto de Streamlit. Por la misma razón, `graficos.py` dibuja los waterfalls de uno en uno con un candado (`pyplot` no es seguro entre hilos).

**Selección dinámica de modelos de Groq.** `listar_modelos_groq()` consulta `client.models.list()` y cruza la respuesta con `MODELOS_GROQ_PRIORIDAD` (modelos verificados con Tool Calling: `gpt-oss-120b` como principal y `gpt-oss-20b` como respaldo). Si Groq retira un modelo, desaparece del selector sin tocar el código.

**Protección de tokens en la capa gratuita de Groq.** El plan gratuito permite 8.000 tokens por minuto y 200.000 al día por modelo (`gpt-oss-120b` y `gpt-oss-20b`). Una pregunta suele hacer dos llamadas al LLM (elegir herramientas y redactar la respuesta), y cada una envía el System Prompt (unos 940 tokens) y los esquemas de las 6 herramientas (unos 1.525): unos 2.460 fijos por llamada. Antes eran 4.200 y una pregunta superaba el límite por minuto; desde el recorte, una pregunta normal gasta entre 4.800 y 6.000 tokens (unas 7.000 si lee una nota de metodología). Para no gastar más:
- el System Prompt guarda solo las reglas de interpretación y el enrutado; cómo llamar a cada herramienta está en su descripción, sin repetirlo;
- los filtros de las herramientas son texto con valor vacío por defecto, que ocupa menos de la mitad que «texto o nada»;
- ventana deslizante de 4 mensajes, que empieza siempre en una pregunta del usuario y no separa una llamada a herramienta de su resultado;
- salidas de herramientas en JSON compacto con etiquetas legibles: de unos 50 a 500 tokens (las notas de metodología, de 800 a 2.800);
- las gráficas de las respuestas las dibuja la app a partir del JSON, sin pedírselas al LLM;
- la exploración manual y la galería se calculan con Python puro y consumen 0 tokens.

**Límite de cuota (429) con reintentos y respaldo.** `ChatGroq` se crea con `max_retries=0` y la llamada se envuelve con `tenacity` (`@retry` con backoff exponencial de 2 a 10 s, 2 intentos por modelo). Si el modelo sigue sin cuota, `invocar_con_respaldo` salta al siguiente modelo validado.

## 5. Herramientas: qué hace cada una y cuándo se llama

El agente elige las herramientas leyendo su descripción (el *docstring*) y el enrutado del System Prompt: la vía más barata que responda la pregunta, todas las llamadas independientes en paralelo y como mucho 2 rondas de llamadas por pregunta. La edición por defecto es 2022.

| Vía | Herramienta | Qué hace | Cuándo la llama el agente |
|---|---|---|---|
| a) Estadísticas descriptivas | `get_math_stats` | Media ponderada de la nota (o de otra variable del alumno) en cualquier subgrupo: país, edición, `estratos fijos`, nivel de compromiso, nivel de recursos, sexo u origen. Con `group_by` compara grupos o hace el ranking de países. La app dibuja su resultado | «¿Qué nota sacan los centros rurales?», «ranking de países en 2022», «¿cómo ha evolucionado la nota?» |
| | `calculate_correlation` | Correlación ponderada entre dos variables del alumno: la media de las correlaciones dentro de cada país y edición (como en el EDA) y la global | «¿Se relacionan los recursos con la nota?» |
| b) Contexto | `get_context_chart` | Devuelve el PNG del catálogo que mejor encaja con las palabras clave (descriptivos, regresión, robustez, Australia, XGBoost…) y los 5 más relacionados. El agente no ve la imagen: la explica con su descripción | Cuando el usuario quiere ver un gráfico del análisis |
| | `get_methodology` | Devuelve una nota de `metodologia/` | Preguntas de «cómo» o «por qué» sobre el diseño del estudio, y «¿es fiable este resultado?» |
| c) Modelos | `get_regression_weights` | Resultados de la regresión por edición (ver abajo) | Peso de una variable, qué pesa más, diferencias entre perfiles con lo demás igual, simulaciones y cambios de 2012 a 2022 |
| | `get_xgboost_patterns` | Patrones del XGBoost a partir de los SHAP de cada alumno (ver abajo) | Qué mueve más la predicción, por qué un grupo saca más o menos y la forma de una relación |

**Temas de `get_methodology`** (uno por fichero):

| Tema | Contenido |
|---|---|
| `muestra_y_limitaciones` | Qué se estudia, por qué estos 12 países y estos alumnos, nota y pesos, limitaciones |
| `compromiso` | Cómo se construye el índice de compromiso y competencia, niveles y cautelas (la subida viene de los deberes) |
| `recursos_estudio` | Cómo se construye el índice de recursos de estudio y cómo se igualan las ediciones |
| `controles` | Variables de control, por qué no se usa el índice socioeconómico de la OCDE, respuestas que faltan |
| `perfiles_centro` | Los `estratos fijos`, perfiles con pocos centros y la cautela de Australia |
| `modelos` | Cómo se lee un coeficiente, unidades, diferencias entre perfiles, XGBoost y SHAP, calidad de los modelos |
| `robustez` | Qué resultados aguantan las comprobaciones y cuáles hay que contar con cautela |

**`get_regression_weights(variable, year, tipo_centro, ubicacion, tamano, cambio)`**

| Llamada | Devuelve | Pregunta típica |
|---|---|---|
| Sin `variable` | El peso de todas las variables en una edición, ordenadas por puntos por desviación típica (la única forma justa de compararlas) | «¿Qué pesa más en la nota?» |
| `variable='recursos'` | Coeficiente común con su unidad (por cada uno más de los seis recursos, que se nombran), intervalo, p y cuántos recursos tiene el alumnado (media y % con los seis) | «¿Cuánto se asocian los recursos con la nota?» |
| `variable='compromiso'` (o `'faltar'`, `'llegar tarde'`, `'deberes'`) | Además del coeficiente del índice, el de cada conducta por separado (2012 y 2022, de las comprobaciones de robustez) y qué parte del alumnado la tiene. El índice da el mismo peso a las tres, pero no pesan igual | «¿Y si deja de faltar a clase?», «¿importan los deberes?» |
| Con filtros de perfil | El coeficiente dentro de esos perfiles, y detrás el común para comparar. Solo compromiso y recursos tienen uno por perfil | «¿Y en los privados urbanos grandes?» |
| `variable='perfil'` | Diferencia de nota de cada perfil frente al público urbano mediano, con alumnos iguales en todo lo demás | «Con alumnos parecidos, ¿se saca más en los privados?» |
| `cambio`, `desde` | Simulación lineal (coeficiente × cambio) con su intervalo y el aviso del efecto techo. Con `desde`, comprueba que el resultado cabe en la escala (los recursos van de 0 a 6) | «¿Y si un alumno con 3 recursos tuviera el doble?», «¿y si le quitamos el ordenador?» |
| `year='all'` | Las cuatro ediciones y el cambio de 2012 a 2022 con su p | «¿Ha cambiado el peso del compromiso?» |

**`get_xgboost_patterns(analisis, year, variable, filtros, group_by)`**

| `analisis` | Devuelve | Pregunta típica |
|---|---|---|
| `importancia` | Puntos que mueve cada variable de media (media del valor absoluto del SHAP), en un grupo, entre grupos o entre ediciones. Indica qué variables difieren más de 2 puntos entre grupos | «¿Qué pesa más en los centros rurales frente a los urbanos?» |
| `contribucion` | Por qué la predicción de un grupo está por encima o por debajo de la media de la edición: el SHAP medio de cada variable y el valor típico del grupo. Dibuja un waterfall | «¿Por qué los alumnos de privados sacan más?» |
| `forma` | SHAP medio para cada valor de una variable (las continuas, por quintiles). En el compromiso, cada valor del índice aparece como su combinación de conductas | «¿Se aplana la relación de los recursos con la nota?» |

Los niveles descriptivos usan cortes fijos, iguales en todas las ediciones:
- **Compromiso:** bajo por debajo de −1 desviación típica, medio entre −1 y +1, alto por encima de +1 (los umbrales del EDA del índice).
- **Recursos de estudio:** hasta 4, 5 de 6 o los 6.

## 6. Interfaz

- **Barra lateral:** selector del modelo principal (el resto queda como respaldo), nueva conversación y controles de exploración manual: edición, país, dimensión de comparación y filtros de `estratos fijos`.
- **Columna izquierda:** chat. Mientras el agente trabaja, indica qué herramienta consulta. Si una respuesta tiene gráfica, el botón «Ver gráfico en el panel» la vuelve a mostrar.
- **Columna derecha:** panel con el gráfico arriba y la explicación debajo, en tres vistas:
  - *Exploración manual:* barras por grupo o líneas por edición, con un texto generado en Python y los avisos.
  - *Galería de gráficos:* PNG del catálogo.
  - *Respuesta del agente:* las gráficas, las imágenes y los waterfalls del último turno, con su explicación.

## 7. System Prompt

`system_prompt.md` fija el alcance (solo matemáticas, 12 países, alumnos del modelo, aviso de 2018), el estilo, los guardarraíles y el enrutado en 5 rutas:

1. Estadísticas descriptivas: `get_math_stats` y `calculate_correlation`.
2. Gráficos del análisis: `get_context_chart`.
3. Metodología: `get_methodology`, contestando solo con su texto.
4. Regresión: `get_regression_weights`, diciendo siempre la unidad, aclarando que la diferencia entre perfiles es «con lo demás igual» y que una simulación no predice la nota de un alumno concreto. Ante «¿es fiable?», consultar `robustez`.
5. XGBoost: `get_xgboost_patterns`. La importancia no es un coeficiente, el SHAP no ve grupos pequeños y las diferencias de uno o dos puntos se cuentan como «patrón parecido».

El marco teórico no va en el System Prompt, que se envía en cada llamada. Encaja mejor como una nota más (`metodologia/marco_teorico.md`), que el agente solo lee cuando se pregunta por él.

## 8. Traducción de la entrega del modelado

La tabla de la regresión de `models/` tiene un formato distinto del que usa el agente. `preparar_artefactos.py` la traduce sin tocar el original:

| `regresion` | `estrato` | De dónde sale |
|---|---|---|
| `comun` | `todos` | Especificación `pendiente_comun`, filas `principal`: un coeficiente por variable para todos los perfiles |
| `comun` | el perfil | Especificación `pendiente_comun`, filas `efecto_fijo_perfil`, con `variable = efecto_perfil`: diferencia frente a `PUB_URB_MED` con lo demás igual |
| `por_perfil` | el perfil | Especificación `pendiente_por_perfil`, filas `pendiente_perfil`: el coeficiente de compromiso y recursos en cada uno de los 15 perfiles |

- **Unidades:** se dejan las originales, las mismas de `alumnos_modelo.parquet`. El coeficiente tipificado pasa a la columna `coeficiente_dt`, vacía en las variables de 0 o 1 y en los efectos de perfil.
- **Descartes:**
  - la constante, porque sin los efectos fijos de país no corresponde a ningún país;
  - los efectos fijos de perfil de la regresión por perfil, porque allí la distancia entre perfiles depende de los recursos del alumno;
  - los controles de la regresión por perfil, que son casi iguales a los de la común.

- **Métricas:** el modelo `regresion` pasa a llamarse `regresion_por_perfil`, que es la que se evaluó.
- `shap_alumnos.parquet` se copia tal cual, solo con los alumnos del agente, tras comprobar que en cada alumno el valor base más los SHAP da la predicción.
