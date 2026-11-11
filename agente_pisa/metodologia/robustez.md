# Comprobaciones de robustez: qué resultados aguantan

## Conclusión sobre el cambio de 2012 a 2022
- En el conjunto de los 12 países, la relación del compromiso y de los recursos con la nota apenas cambia de 2012 a
  2022 y es parecida en todos los perfiles de centro.
- Los cambios que aparecen a primera vista se deben a un país o a un cambio en la pregunta:
  - la subida del coeficiente del compromiso viene solo de los deberes, cuya pregunta cambia en cada edición;
  - el salto del coeficiente de los recursos en el privado urbano grande en 2022 depende solo de Australia.
- El detalle de cada uno está en «Resultados que hay que contar con cautela».

## Qué se hizo
- El modelo de regresión se repitió cambiando una decisión cada vez, para ver si los resultados dependen de ella:
  - cómo se mide el compromiso (sus tres componentes por separado, o solo absentismo e impuntualidad);
  - los recursos de estudio por tramos en lugar de la escala de 0 a 6;
  - otros controles del centro (su nivel socioeconómico medio, la escasez de recursos según el director);
  - quitar cada uno de los 12 países, de uno en uno;
  - un panel ampliado de 25 países (sin la edición 2018), también quitando cada país;
  - una pendiente por país en lugar de la común;
  - un efecto fijo por celda país × perfil en lugar de los de país y perfil por separado, en los dos paneles (solo
    compara alumnos del mismo país y perfil).
- La aplicación usa siempre el modelo de partida (12 países). Estas comprobaciones solo sirven para decir cuánto
  fiarse de cada resultado.
- Los coeficientes están en puntos de matemáticas por desviación típica de la variable. «Cambio» es el coeficiente
  de 2022 menos el de 2012.

## Qué significa «frágil»
- Un resultado es **frágil** si sale en el modelo de partida pero deja de distinguirse de cero (p ≥ 0,05) al
  cambiar una sola de las decisiones de arriba. Es **firme** si aguanta todas.
- Frágil no es falso: con los 12 países el resultado existe, pero no puede presentarse como algo general.
- Frágil no es poco representativo: eso es otra etiqueta, la de un perfil con menos de 30 centros en una edición.
- Solo es firme lo que se comprobó: lo que no aparece en esta nota no es ni frágil ni firme.

## Lo que no se comprobó
- **Las diferencias de nota entre perfiles** (cuánto más o menos saca cada perfil que el público urbano mediano con
  lo demás igual) no se repitieron quitando países ni con las otras variantes. No hay cifras para decir si dependen
  de algún país: ni frágiles ni firmes, sin comprobar.
- **La aplicación no tiene resultados por país.** Todas sus cifras juntan los 12 países (cada alumno se compara con
  los de su país). La pendiente por país solo está en el notebook de robustez (sección 10), con lo que resume esta
  nota en «Resultados que hay que contar con cautela».

## Resultados que aguantan todas las comprobaciones
- **El compromiso se asocia a más nota en todas las ediciones**: entre unos 9 y 13 puntos por desviación típica,
  sea cual sea la especificación. Es la variable de interés con más relación con la nota.
- **La asistencia pesa igual en toda la década.** Haber faltado a clase se asocia a unos 17 o 18 puntos menos y
  cada nivel de impuntualidad a unos 10 a 13 puntos menos, en 2012 y en 2022. Se mantiene con más alumnos (sin
  exigir la pregunta de deberes) y con 25 países.
- **Los recursos de estudio se asocian a más nota, de forma estable**: unos 5 o 6 puntos por desviación típica
  (aproximadamente, por tener uno más de los seis recursos; entre 4,5 y 6,5 según la especificación). El cambio de 2012 a 2022 nunca
  se distingue de cero con los 12 países (va de 0,0 a 0,8 puntos), salvo al quitar Eslovaquia (1,5, p = 0,041). Por tramos, tener todos menos uno o los
  seis sigue asociado a más nota, y tener los seis añade poco sobre tener cinco.
- **Las conclusiones no dependen de los controles del centro.** Con la escasez de recursos los coeficientes no se
  mueven. Con el nivel socioeconómico medio del centro, el de los recursos baja algo (de 5,4-5,9 a 4,5-5,2), pero
  las conclusiones son las mismas.
- **Los coeficientes comunes no cambian al comparar solo alumnos del mismo país y perfil.** Con el efecto fijo país
  × perfil, el del compromiso en 2022 pasa de 12,1 a 11,8 y el de los recursos, de 5,9 a 5,8.

## Resultados que hay que contar con cautela
- **La subida del compromiso de 2012 a 2022 es pequeña y frágil.** En el modelo de partida sube 1,5 puntos
  (p = 0,029). Es positiva en todas las comprobaciones (de 1,0 a 2,7 puntos), pero deja de distinguirse de cero al
  quitar Australia, Reino Unido, Hungría, Italia, Eslovaquia o Eslovenia, con el nivel socioeconómico del centro o
  con el efecto fijo país × perfil (1,3 puntos, p = 0,052).
  Además, viene del componente de deberes, el que peor se compara entre ediciones (ver la nota de compromiso).
  Por eso la conclusión es que la relación del compromiso con la nota apenas cambia: no se puede decir que haya
  aumentado ni que se haya reforzado, ni llamar significativa a esa subida.
- **El privado urbano grande en 2022 depende de Australia.** Esto no es su diferencia de nota de +10 frente al
  público urbano mediano: esa diferencia no se comprobó quitando países y no se deduce de lo que sigue. Lo frágil es el coeficiente de los recursos de estudio en ese perfil. Con los 12 países, ese coeficiente sube unos 14 puntos de 2012 a 2022 (de 3,4 a 17,6), y por esa subida el
  coeficiente de los recursos difiere entre perfiles en 2022. Aguanta las comprobaciones por tramos y con otros
  controles, pero no dos: sin Australia, la subida baja a 6,8 puntos (p = 0,316) y el coeficiente de los recursos
  deja de diferir entre perfiles; con el efecto fijo país × perfil, los perfiles dejan de diferir en 2022 (p = 0,111)
  y en el cambio (p = 0,544). Al quitar cualquiera de los otros 11 países la subida se mantiene (de 13,9 a 17,0
  puntos, p ≤ 0,003): Australia es el único país del que depende. Con 12 países, ninguna diferencia entre perfiles
  resiste las comprobaciones. Ver la nota de perfiles de centro.
- **Con 25 países la imagen de los recursos es algo distinta.** Su coeficiente sí sube de 2012 a 2022 (de 4,5 a 6,1,
  p = 0,002) y en 2022 los privados urbanos grandes y medianos quedan por encima de los públicos, sin depender de
  ningún país (tampoco de Australia). Con el efecto fijo país × perfil esa diferencia de 2022 se mantiene (p < 0,001;
  privados urbanos grandes y medianos 13,2-15,2 frente a públicos 4,1-7,8), pero su cambio desde 2012 deja de
  diferir entre perfiles (p = 0,100): es una diferencia de nivel en 2022, no un cambio distinto por perfil. Es un
  indicio de un patrón más general, pero ese panel tiene otros países, otro índice de compromiso y no tiene 2018: no
  es el modelo de la aplicación.
- **La estabilidad es la del conjunto de los 12 países, no la de cada uno.** Con una pendiente por país, las
  pendientes difieren entre países en todas las ediciones, y su cambio de 2012 a 2022 también (p = 0,019 en el
  compromiso y p < 0,001 en los recursos). En el compromiso sube en Italia y Eslovaquia y baja en Polonia; en los
  recursos sube en Australia y Eslovenia y baja en Eslovaquia y Polonia. En el coeficiente común esos cambios se
  compensan.

## Lo que no cambia entre perfiles
- **En el compromiso no se detectan diferencias entre perfiles de centro**, ni en 2022 ni en su cambio de 2012 a
  2022. Ningún país quitado ni el efecto fijo país × perfil cambian esta conclusión con los 12 países. Que no se
  detecten diferencias no demuestra que sean iguales.

## Cómo usar esta nota
- Si el usuario pregunta si un resultado es fiable, contestar con lo que aguanta y lo que no, en lenguaje sencillo.
- Si pregunta si la relación ha cambiado de 2012 a 2022, empezar por la conclusión del principio de esta nota y
  explicar después de dónde salen los cambios aparentes.
- No presentar como hallazgo firme la subida del compromiso ni la del privado urbano grande.
- Si preguntan por qué es frágil «el resultado del privado urbano grande», empezar aclarando que lo frágil es la
  subida del coeficiente de sus recursos (de 3,4 a 17,6 puntos por desviación típica), no los +10 puntos de nota que
  saca frente al público urbano mediano, que no se comprobaron.
- De lo que no se comprobó, decir que no está documentado; no deducir que es estable.
- Origen: `notebooks/fase_3_modelado/03_robustez_regresion.ipynb`; las cifras de las comprobaciones en
  `data/processed/resultados_robustez.parquet`, salvo la pendiente por país (sección 10) y el efecto fijo país ×
  perfil (sección 11), que solo están en el notebook.
