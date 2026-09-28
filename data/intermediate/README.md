
# Informacion de las tablas contenidas en la carpeta intermediate #

## `pisa_submuestra_estudiantes.parquet`
    
    Tabla que contiene la submuestra creada a partir de los datos raw de `CY08MSP_STU_QQQ.FORMAT`.
    Contiene todos los indices agregados contenidos en el archivo STU_QQQ pero de la submuestra realizada del 20% de los datos.

    * Numero de alumnos originales: 613744
    * Numero Alumnos seleccionados: 146500
    * Porcentaje de la muestra sobre el total: 23.87%

    Contiene los siguientes indices:

| Categoría | Variables |
|---|---|
| IDs y pesos base | `CNT`, `STRATUM`, `CNTSCHID`, `CNTSTUID`, `W_FSTUWT` |
| Contexto socioeconómico y familiar | `ESCS`, `HISEI`, `PAREDINT`, `HOMEPOS`, `WEALTH`, `CULTPOSS`, `HEDRES`, `ICTRES` |
| Actitud y psicología | `ANXMAT`, `MATHEFF`, `MATHINT`, `MATHBEH` |
| Clima escolar (alumno) | `BELONG`, `BULLY`, `DISCLIMA`, `TEACHSUP` |
    



## `pisa_sub_dataframe_global_nulos.parquet`
    
    Tabla que contiene los datos unidos de *pisa_submuestra_estudiantes.parquet* y los indces de los datos raw de ¨CY08MSP_TCH_QQQ.FORMAT¨, solamente contiene
    los datos de los colegios elegidos en la submuestra anterior. La tabla presenta nulos debido a paises de los que no existe información de dichos indices, por lo que la información no ha sido tratada 

    Contiene los indices de pisa_submuestra_estudiantes.parquet + los indices especificos del colegio :

| Categoría | Variables |
|---|---|
| IDs y pesos base | `CNT`, `STRATUM`, `CNTSCHID` |
| Cuestionario de centro | `STRATIO`, `PROPMATH`, `SCHLCLI`, `EDUSHORT`, `STAFFSHORT` |


# `pisa22_submuestra_global_limpio.parquet`

    Tabla que parte del pisa_sub_dataframe_global_nulos.parquet de el que se ha realziado el EDA . Aspectos importantes

* ⚠️ Se ha eliminado la columna `ICTRES` por faltar datos en 27 de los 81 paises registrados
* ⚠️ La imputación de nulos es *multivariante* resolviendo el valor del nulo mediante la observacion deotras variables presentes que tengan correlacion

# `pisa22_submuestra_paises.parquet`

    Muestra de cada pais los indices derivados (que han sido ponderados para ajustarse a al submuestra).
    Es decir, cada fila es un único pais y cada columna un índice derivado

    Parte de la tabla pisa22_submuestra_global_limpio.parquet, pero se han eliminado las variables identificativas de estratio y de colegio, así como las ponderaciones de los alumnos y de senado ajustadas.
    
    Como se ha indicado al principio, cada indice derivado presente en la tabla es el resultado de hacer la media ponderada (ajustada a la submuestra) del indice que se presentaba para cada colegio:
    
        indice : np.average(x[indice],weights=x['W_FSTUWT_adj']) 

| Categoría | Variables |
|---|---|
| Identificador | `CNT` |
| Índices | `PAREDINT`, `HISEI`, `HOMEPOS`, `ESCS`, `BELONG`, `TEACHSUP`, `MATHEFF`, `ANXMAT`, `STRATIO`, `PROPMATH`, `STAFFSHORT`, `EDUSHORT` |