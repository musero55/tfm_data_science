# Modelos: regresión y XGBoost

## Qué preguntan
- Cuánto se asocian el compromiso del alumno y sus recursos de estudio con su nota de matemáticas, comparando alumnos
  de igual origen social; si esa relación cambia según el perfil del centro, y si ha cambiado de 2012 a 2022.
- La regresión es el modelo principal: da un coeficiente con su intervalo. El XGBoost sirve para comprobar si la
  regresión deja fuera curvas o interacciones importantes.
- Se estima un modelo por edición, con los mismos 298.782 alumnos de los 12 países. Cada país pesa casi lo mismo
  (ver la nota de muestra y limitaciones).
- PISA da varios valores posibles de la nota de cada alumno (5 en 2012, 10 en las demás ediciones). Cada modelo se
  repite con cada uno y los resultados se combinan; usar uno solo daría errores demasiado pequeños.

## La regresión
La nota de un alumno se explica con estas piezas, en cada edición por separado:
- **País (efecto fijo):** cada país tiene su propio punto de partida. Así los coeficientes solo comparan alumnos del
  mismo país, sin mezclar diferencias de sistema educativo o de forma de responder.
- **Perfil de centro (efecto fijo):** cada perfil tiene también su propio punto de partida. En 2022, con lo demás
  igual, el privado urbano grande parte de unos 10 puntos más que el público urbano mediano (el de referencia) y el
  privado urbano pequeño, de unos 18 menos.
- **Compromiso y recursos de estudio:** las variables de interés.
- **Controles:** ocupación y estudios de los padres, riqueza del hogar, libros en casa, sexo, inmigración,
  composición social del centro y dotación del centro (ver la nota de controles). Tienen un solo coeficiente para
  todos los perfiles. Si a un alumno le falta un control, se le pone la media y un indicador de «sin dato».

Hay dos versiones:
- **Coeficiente común:** un solo coeficiente de compromiso y otro de recursos para todos los perfiles.
- **Coeficiente por perfil:** compromiso y recursos tienen un coeficiente distinto en cada uno de los 15 perfiles.
  Los controles siguen siendo comunes.

## Cómo se lee un coeficiente
- Coeficiente común de los recursos en 2022, unos 6 puntos: «entre dos alumnos del mismo país, del mismo perfil de
  centro y de igual origen social, el que tiene uno más de los seis recursos de estudio (por ejemplo, también
  ordenador para estudiar) saca de media unos 6 puntos más».
- No es la relación con todos los alumnos mezclados. Sin el efecto fijo de perfil, parte de la diferencia de nota
  entre tipos de centro se atribuiría a los recursos, porque los centros con mejores notas también tienen alumnos con
  más recursos. El efecto fijo separa las dos cosas: el nivel de cada perfil va por un lado y el coeficiente por otro.
- El coeficiente por perfil responde a lo mismo, pero solo dentro de ese perfil. Si los coeficientes por perfil se
  parecen entre sí, el coeficiente común resume bien a todos.
- Variables de sí o no (chica, inmigrante): puntos por pasar de «no» a «sí».

## Diferencia de nota entre perfiles, con lo demás igual
- El punto de partida de cada perfil responde a «con alumnos iguales, ¿cuánto más o menos se saca en este perfil?».
  Se mide frente al público urbano mediano y sale de la regresión de coeficiente común: como ahí todos los perfiles
  tienen los mismos coeficientes, la diferencia entre dos perfiles es la misma para cualquier alumno.
- No es la diferencia de nota media. En 2022 los alumnos del privado urbano grande sacan de media unos 59 puntos más
  que los del público urbano mediano; con alumnos iguales (mismo país, compromiso, recursos y origen social, y centro
  de igual composición social), unos 10 más. El resto se explica por quiénes son sus alumnos.
- En 2022 se distinguen de cero el privado urbano grande (+10), el público urbano grande (+7,5) y el público rural
  pequeño (+5) por arriba, y el concertado urbano pequeño (−12) y el privado urbano pequeño (−18) por abajo; en los
  demás la diferencia es pequeña o incierta. Que un perfil saque más con lo demás igual no prueba que el centro sea la causa: puede haber
  diferencias entre alumnos que el modelo no mide.
- Estas diferencias juntan los 12 países y no se repitieron quitando países. La cautela de Australia es sobre el
  coeficiente de los recursos del privado urbano grande, no sobre su +10 (ver la nota de robustez).
- En la regresión por perfil también hay un punto de partida, pero allí cada perfil tiene su propia inclinación y la
  distancia entre perfiles cambia según los recursos del alumno; por eso no se usa.

## Unidades
- La regresión se estima con las variables tipificadas (puntos por desviación típica, calculada con las cuatro
  ediciones juntas). Sirve para comparar variables entre sí: «el compromiso pesa el doble que los recursos».
- La aplicación usa unidades originales: en los recursos, puntos por cada uno más de los seis objetos que se
  cuentan (habitación propia, ordenador, internet, libros de ayuda, libros técnicos y diccionario); en el compromiso,
  por punto del índice. Es el mismo coeficiente dividido por la desviación típica; no es otro modelo. Ejemplo:
  recursos en 2012, 5,44 por desviación típica = 5,80 por cada recurso más. Privado urbano grande en 2022: 17,65 por
  desviación típica = 18,79 por cada recurso más.
- Al hablar del compromiso, decir qué conductas hay detrás, no «un punto del índice». El índice da el mismo peso a
  las tres, pero no pesan igual. En un modelo con las tres por separado (mismos controles; solo 2012 y 2022):
  haber faltado a clase en las dos últimas semanas se asocia a unos 17 o 18 puntos menos; cada paso de impuntualidad
  (de ninguna vez a 1 o 2, o de 1 o 2 a 3 o más), a unos 12 menos en 2012 y 10 en 2022; y dedicar a los deberes un
  tiempo intermedio, a nada en 2012 (−0,4) y a unos 6 puntos más en 2022. Para una conducta concreta hay que usar
  estas cifras: con el índice, «dejar de faltar» daría unos 13 puntos y «deberes en tiempo intermedio», unos 14.
- Al hablar de recursos, decir qué objetos son, no «un punto» de un índice. La mayoría de los alumnos tiene ya 5 o 6
  de los seis (en 2022, el 84 %; con 3 o menos, el 5 %): «tener el doble» solo es posible para muy pocos, y para el
  resto el margen es de uno como mucho.
- En el compromiso apenas cambia, porque su índice ya tiene desviación típica ≈ 1.
- Las notas de robustez y de perfiles de centro citan cifras por desviación típica. Al dar una cifra, decir siempre
  en qué unidad está y no comparar cifras de unidades distintas.

## Fiabilidad de los coeficientes
- El error típico tiene en cuenta que los alumnos de un mismo centro se parecen y la diferencia entre los valores
  posibles de la nota. Se comprobó que coincide casi del todo con el método oficial de PISA (pesos replicados).
- Un perfil con menos de 30 centros en una edición se marca como poco representativo: su coeficiente es poco fiable.
- Para decidir si los perfiles difieren, cuenta el contraste conjunto de los 15 coeficientes, no comparar perfiles de
  uno en uno (con tantos perfiles, alguno sale distinto por azar).

## Resultados principales de la regresión
- Compromiso: unos 10 a 12 puntos por punto del índice en todas las ediciones; sin diferencias detectables entre
  perfiles. Por conductas, lo que más pesa es la asistencia (faltar a clase y llegar tarde); los deberes, poco.
- Recursos de estudio: unos 5 a 6 puntos por cada uno más de los seis, estables de 2012 a 2022 para el conjunto de
  los 12 países; sin diferencias entre perfiles salvo en 2022, por el privado urbano grande, una diferencia que no
  resiste quitar Australia ni el efecto fijo país × perfil (ver la nota de perfiles de centro y la de robustez).
- De 2012 a 2022, la relación del compromiso y de los recursos con la nota apenas cambia en el conjunto de los 12
  países. La subida aparente del compromiso (+1,5 puntos) viene de los deberes, cuya pregunta cambia, y es frágil
  (ver la nota de robustez).
- Pesan más algunos controles que las variables de interés: la composición social del centro y los libros en casa.

## XGBoost y SHAP
- **XGBoost:** suma cientos de árboles de decisión pequeños. Puede recoger curvas e interacciones sin que nadie se
  las indique. Usa los mismos alumnos y variables que la regresión; el país y el perfil entran como variables.
- **Valor SHAP:** para cada alumno y variable, cuántos puntos suma o resta esa variable a su predicción frente a la
  predicción media de la edición. En cada alumno, predicción media + suma de los SHAP = su predicción.
- **Importancia SHAP:** la media del valor absoluto del SHAP, es decir, cuántos puntos mueve la variable de media.
  No es un coeficiente: no dice la dirección ni «cuánto por unidad». Por eso que la importancia del compromiso no
  cambie entre ediciones no contradice que su coeficiente suba algo.
- **Contribución de un grupo:** la media de los SHAP de sus alumnos. Dice por qué la predicción de ese grupo está
  por encima o por debajo de la media: por ejemplo, en 2022 los alumnos de centros privados tienen una predicción
  unos 47 puntos más alta, y unos 33 vienen de la composición social de sus centros.
- Los SHAP se calculan con todos los alumnos con los que se entrenó el modelo: describen lo que ha aprendido, no
  son predicciones para alumnos nuevos.
- Cuando dos variables van juntas (recursos y origen social), SHAP reparte el mérito entre ellas de una forma que
  depende del modelo.

## Qué añade el XGBoost
- **Apenas predice mejor:** el R² sube solo entre 0,02 y 0,03 sobre la regresión. La relación es casi lineal y la
  regresión recoge casi todo lo que estas variables dicen de la nota.
- **Los recursos tienen efecto techo:** de 3 a 4 recursos se ganan unos 5 a 10 puntos; de 5 a 6, solo entre 1 y 3. La recta
  de la regresión exagera algo el último recurso.
- **El índice de compromiso no se comporta como una escala ordenada:** faltar a clase pesa mucho más que los
  deberes, aunque el índice les da el mismo peso (ver la nota de compromiso).
- **No ve diferencias entre perfiles que la regresión no haya visto**, y tampoco el salto del privado urbano grande
  (menos de 1.500 alumnos). Los SHAP no sirven para estudiar perfiles pequeños.
- El perfil del centro pesa poco una vez conocidos el país y la composición social del centro.

## Calidad de los modelos
Medida con alumnos de centros que el modelo no vio al entrenarse (validación cruzada en 5 grupos de centros):

| Edición | R² regresión | R² XGBoost | Error medio por alumno (XGBoost) |
|---|---|---|---|
| 2012 | 0,35 | 0,39 | 59 puntos |
| 2015 | 0,28 | 0,31 | 60 puntos |
| 2018 | 0,29 | 0,32 | 60 puntos |
| 2022 | 0,31 | 0,34 | 59 puntos |

- Los dos modelos dejan sin explicar unos dos tercios de las diferencias de nota: la nota depende de mucho más.
- Sirven para describir relaciones medias, no para predecir la nota de un alumno concreto.
- Las métricas de la regresión son las de la versión por perfil.

## Cómo usar esta nota
- Asociación, no causa: no decir que tener más recursos o más compromiso sube la nota.
- Cada edición tiene su modelo: se puede comparar un coeficiente entre ediciones, pero no simular «este alumno en
  otra edición».
- Una simulación con la regresión es una cuenta lineal (coeficiente × cambio) y solo vale dentro de los valores
  reales: los recursos van de 0 a 6 y cerca de 6 la ganancia real es menor.
- Avisar si un perfil es poco representativo y, si preguntan si un resultado es fiable, usar la nota de robustez.
- Origen: `notebooks/fase_3_modelado/03_regresion_por_edicion.ipynb` y `03_xgboost_shap.ipynb`.
