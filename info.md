# Comparabilidad de variables PISA entre ediciones

Para un análisis longitudinal de PISA, **no se puede asumir que todas las variables sean directamente comparables entre ediciones**. Hay que distinguir entre distintos tipos de variables.

## 1. Puntuaciones de rendimiento (PV)

* Las puntuaciones de rendimiento en Matemáticas, Lectura y Ciencias **sí están diseñadas para ser comparables entre ciclos**.
* La OECD utiliza *trend items* y procedimientos de *fixed parameter linking* para mantener una escala común.
* Al analizar diferencias entre años, debe considerarse también el **error de enlace (*link error*)**, además del error muestral.

## 2. ESCS (ISEC)

* El `ESCS` original de cada edición **no debe compararse directamente entre años**.
* En cada ciclo se vuelve a estandarizar (`media OECD = 0`, `SD = 1`) y algunos de sus componentes pueden cambiar.
* Para análisis longitudinales debe utilizarse el **Trend ESCS**, recalculado por la OECD utilizando una metodología común.

Por ejemplo, un `ESCS = 0.5` en 2015 no necesariamente representa exactamente el mismo nivel socioeconómico que un `ESCS = 0.5` en 2022.

## 3. Índices de cuestionarios

Hay que distinguir entre:

* **Trend Scales:** están diseñadas específicamente para medir cambios a lo largo del tiempo y utilizan procedimientos de enlace para mantener una métrica común.
* **Índices nuevos o específicos de un ciclo:** aunque estén estandarizados (`mean = 0`, `SD = 1`), **no son necesariamente comparables entre años**.

Por tanto, no basta con que una variable tenga el mismo nombre en dos ediciones.

# Implicaciones para Machine Learning

Si se combinan PISA 2015, 2018 y 2022 para entrenar un modelo de ML:

1. **Comprobar la definición de cada variable en cada ciclo.**
2. Para `ESCS`, utilizar **Trend ESCS** en lugar de los valores originales de cada edición.
3. Comprobar qué variables son **Trend Scales**.
4. Armonizar las variables categóricas cuya clasificación haya cambiado entre ciclos.
5. Solo después de esta armonización aplicar transformaciones como **One-Hot Encoding**.
6. Comprobar que los tipos de datos y las distribuciones sean coherentes entre ciclos.

## Armonización de variables categóricas

Las clasificaciones internacionales pueden cambiar entre ediciones. Por ejemplo:

* `ISCO-88` → `ISCO-08` para ocupaciones.
* `ISCED-97` → `ISCED-11` para educación.

Por ello, **no se deben introducir directamente los códigos numéricos de ISCO o ISCED en el modelo**, ya que son categorías y sus valores numéricos no representan una magnitud continua.

Antes del One-Hot Encoding hay que utilizar las **tablas oficiales de correspondencia (*mapping tables*)** para garantizar que las categorías representan conceptos equivalentes.

# Pipeline recomendado

```text
Microdatos de cada ciclo
        ↓
Comprobar definición de las variables
        ↓
Identificar Trend Scales
        ↓
Sustituir ESCS por Trend ESCS
        ↓
Armonizar categorías y codificaciones
        ↓
Comprobar tipos y distribuciones
        ↓
Concatenar los ciclos
        ↓
Aplicar preprocessing / One-Hot Encoding
        ↓
Entrenar el modelo de ML
```

## Errores que se deben evitar

* **Asumir comparabilidad por el nombre de la variable.**
* Comparar directamente el `ESCS` original de distintos ciclos.
* Utilizar un índice estandarizado de un único ciclo como si fuera una escala longitudinal.
* Introducir códigos ISCO/ISCED como variables numéricas sin armonización.
* Hacer One-Hot Encoding antes de comprobar que las categorías tienen la misma **equivalencia semántica** entre ciclos.
* Ignorar el **link error** al analizar diferencias temporales en las puntuaciones de rendimiento.

> **Idea fundamental:** para combinar PISA longitudinalmente, hay que armonizar tanto la **semántica** como la **métrica** de las variables antes de entrenar el modelo. Que dos variables tengan el mismo nombre no garantiza que midan exactamente lo mismo entre ediciones.