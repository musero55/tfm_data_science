"""
preparacion_modelo.py
=====================

Preparación de la tabla por alumno (`tabla_alumno.tabla_por_alumno`) para los modelos, en dos pasos que se
mantienen separados:

    * `marcar_universo`: la tabla para guardar, en sus unidades originales, sin imputar ni codificar, con cuatro
      marcas por alumno: si su perfil queda fuera (`perfil_excluido`), si su perfil-edición tiene pocos centros
      (`perfil_poco_representativo`), si tiene las dos variables de interés (`con_variables_interes`) y si entra en
      el modelo (`en_modelo`). Es la que leen los notebooks de modelado y el agente.
    * `parametros_escala` / `matriz_regresion`: la matriz de la regresión, que se reconstruye siempre desde la
      tabla guardada. Las variables continuas se tipifican con `SENWT` y las cuatro ediciones juntas (la escasez,
      edición a edición), y los huecos de los controles y comprobaciones se rellenan con la media y un indicador
      `sin_<variable>`. Las variables de interés no se imputan: los alumnos sin ellas no entran.
    * `matriz_xgboost`: las mismas variables en sus unidades originales y con sus huecos, que XGBoost trata solo.
    * `DESCRIPCION_COLUMNAS` / `diccionario_columnas`: qué es cada columna de la tabla guardada y en qué unidades.

Los perfiles de centro son los de `00_Pipeline_generacion_estratos`; el umbral de centros, el de
`01_EDA_variables_modelo`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pisa_pipeline import DOMINIOS
from src.tabla_alumno import VARIABLES_MODELO

MINIMO_CENTROS = 30   # centros de un perfil-edición por debajo de los cuales sus resultados serían poco representativos
DE_INTERES = VARIABLES_MODELO['compromiso'] + VARIABLES_MODELO['recursos del alumno']
CONTROLES = VARIABLES_MODELO['control: origen social'] + VARIABLES_MODELO['control: alumno'] + VARIABLES_MODELO['control: centro']
COMPROBACIONES = VARIABLES_MODELO['comprobación']
# Variables de 0 o 1 (o pocos valores ordenados), que se dejan en su escala al tipificar las demás
SIN_TIPIFICAR = ['sexo_femenino', 'inmigrante', 'deberes_en_u', 'SKIPPING', 'TARDYSD']
# Cada edición mide la escasez con su propia escala: se tipifica dentro de cada edición
TIPIFICAR_POR_EDICION = ['escasez_centro']
MARCAS = ['perfil_excluido', 'perfil_poco_representativo', 'con_variables_interes', 'en_modelo']
IDENTIFICADORES = ['CNT', 'CNTSCHID', 'CNTSTUID', 'year', 'pais_edicion', 'centro', 'perfil', 'W_FSTUWT', 'SENWT']


def centros_por_perfil(datos: pd.DataFrame) -> pd.DataFrame:
    """Centros de cada perfil (filas) y edición (columnas) en `datos`."""
    return datos.groupby(['perfil', 'year'])['centro'].nunique().unstack('year').fillna(0).astype(int)


def marcar_universo(datos: pd.DataFrame, minimo_centros: int = MINIMO_CENTROS) -> pd.DataFrame:
    """`datos` con las columnas de `MARCAS`, sin tocar ninguna otra:

    * `perfil_poco_representativo`: el perfil tiene menos de `minimo_centros` centros en esa edición.
    * `perfil_excluido`: el perfil no llega a `minimo_centros` en ninguna edición; sus alumnos no entran en el modelo.
    * `con_variables_interes`: el alumno tiene el índice de compromiso y los recursos de estudio.
    * `en_modelo`: `con_variables_interes` y perfil no excluido."""
    centros = centros_por_perfil(datos)
    pocos = (centros < minimo_centros).stack().rename('perfil_poco_representativo').reset_index()
    excluidos = set(centros.index[(centros < minimo_centros).all(axis=1)])
    datos = datos.merge(pocos, on=['perfil', 'year'], how='left', validate='many_to_one')
    datos['perfil_excluido'] = datos['perfil'].isin(excluidos)
    datos['con_variables_interes'] = datos[DE_INTERES].notna().all(axis=1)
    datos['en_modelo'] = datos['con_variables_interes'] & ~datos['perfil_excluido']
    return datos


def _media_y_desviacion(valores: pd.Series, pesos: pd.Series) -> tuple[float, float]:
    con_dato = valores.notna()
    media = np.average(valores[con_dato], weights=pesos[con_dato])
    return media, np.sqrt(np.average((valores[con_dato] - media) ** 2, weights=pesos[con_dato]))


def parametros_escala(datos: pd.DataFrame, peso_col: str = 'SENWT') -> pd.DataFrame:
    """Media y desviación típica ponderadas con `peso_col` de cada variable continua del modelo, calculadas con los
    alumnos `en_modelo`: una fila por variable con `edicion` = 'todas' o, en `TIPIFICAR_POR_EDICION`, una por
    edición. Sirven para tipificar (`matriz_regresion`) y para pasar un coeficiente por desviación típica a las
    unidades originales."""
    universo = datos[datos['en_modelo']]
    filas = []
    for variable in [variable for variable in DE_INTERES + CONTROLES + COMPROBACIONES if variable not in SIN_TIPIFICAR]:
        grupos = universo.groupby('year') if variable in TIPIFICAR_POR_EDICION else [('todas', universo)]
        for edicion, grupo in grupos:
            media, desviacion = _media_y_desviacion(grupo[variable], grupo[peso_col])
            filas.append({'variable': variable, 'edicion': str(edicion), 'media': media, 'desviacion_tipica': desviacion})
    return pd.DataFrame(filas)


def matriz_regresion(datos: pd.DataFrame, parametros: pd.DataFrame) -> pd.DataFrame:
    """Alumnos `en_modelo` con `IDENTIFICADORES`, los valores plausibles, las variables de interés tipificadas y los
    controles y comprobaciones tipificados (salvo `SIN_TIPIFICAR`) y sin huecos: cada hueco se rellena con 0 (la
    media, en las tipificadas) y se añade el indicador `sin_<variable>` (1 = sin dato) si la variable tiene alguno.
    El perfil y el país quedan como texto: los efectos fijos y las interacciones se construyen al estimar."""
    universo = datos[datos['en_modelo']].reset_index(drop=True)
    valores_plausibles = [columna for columna in universo.columns if columna.startswith('PV') and columna.endswith(tuple(DOMINIOS))]
    matriz = universo[IDENTIFICADORES + valores_plausibles].copy()
    for variable in DE_INTERES + CONTROLES + COMPROBACIONES:
        valores = universo[variable].astype(float)
        if variable not in SIN_TIPIFICAR:
            escala = parametros[parametros['variable'] == variable].set_index('edicion')
            clave = universo['year'].astype(str) if variable in TIPIFICAR_POR_EDICION else pd.Series('todas', index=universo.index)
            valores = (valores - clave.map(escala['media'])) / clave.map(escala['desviacion_tipica'])
        if variable not in DE_INTERES and valores.isna().any():
            matriz[f'sin_{variable}'] = valores.isna().astype(float)
            valores = valores.fillna(0.0)
        matriz[variable] = valores
    return matriz


def matriz_xgboost(datos: pd.DataFrame) -> pd.DataFrame:
    """Alumnos `en_modelo` con `IDENTIFICADORES`, los valores plausibles y las variables del modelo en sus unidades
    originales, con sus huecos (XGBoost los reparte en cada división). El perfil y el país pasan a categoría, que
    XGBoost usa como variable categórica (`enable_categorical=True`) sin crear columnas nuevas."""
    universo = datos[datos['en_modelo']].reset_index(drop=True)
    valores_plausibles = [columna for columna in universo.columns if columna.startswith('PV') and columna.endswith(tuple(DOMINIOS))]
    matriz = universo[IDENTIFICADORES + valores_plausibles + DE_INTERES + CONTROLES + COMPROBACIONES].copy()
    matriz[['perfil_categoria', 'pais_categoria']] = universo[['perfil', 'CNT']].astype('category')
    return matriz


# Qué es cada columna de la tabla guardada (`marcar_universo`) y en qué unidades
DESCRIPCION_COLUMNAS = {
    'CNT': ('identificador', 'País, código ISO de tres letras'),
    'CNTSCHID': ('identificador', 'Centro (único dentro del país y la edición)'),
    'CNTSTUID': ('identificador', 'Alumno (único dentro del país y la edición)'),
    'year': ('identificador', 'Edición de PISA: 2012, 2015, 2018 o 2022'),
    'pais_edicion': ('identificador', 'País y edición, por ejemplo ESP_2022 (grupo del efecto fijo de país)'),
    'centro': ('identificador', 'Centro y edición, país_edición_CNTSCHID (grupo de los errores agrupados)'),
    'perfil': ('grupo', 'Perfil del centro: titularidad (PUB público, CON concertado, PRI privado), ubicación (RUR rural, URB urbano) y tamaño (PEQ pequeño, MED mediano, MACRO grande)'),
    'W_FSTUWT': ('peso', 'Peso final del alumno: a cuántos alumnos de 15 años de su país representa'),
    'SENWT': ('peso', 'Peso de senado: proporcional a W_FSTUWT dentro de cada país-edición, con todos los países-edición pesando lo mismo. Es el peso del modelo'),
    'Indice_Compromiso_Competencia': ('interés', 'Índice de compromiso y competencia (01_EDA_indice_de_compromiso_competencia): media con pesos iguales de las horas de deberes en U invertida, el absentismo y la impuntualidad. Media 0 y desviación típica 1 en el panel con SENWT; más = más compromiso. Vacío si falta algún componente'),
    'recursos_estudio': ('interés', 'Recursos de estudio del hogar (01_EDA_recursos_alumno): cuántos tiene de seis (habitación propia, ordenador, internet, libros de ayuda, libros técnicos y diccionario), de 0 a 6. Con 4 o 5 respondidos se prorratea, por eso hay valores no enteros'),
    'HISEI_TREND': ('control: origen social', 'Ocupación más alta de los padres, índice ISEI reescalado por la OCDE para comparar ediciones (de 11 a 89; más = ocupación de más estatus)'),
    'PAREDINT_TREND': ('control: origen social', 'Años de estudios del padre o la madre con más estudios, reescalado por la OCDE (de 3 a 16)'),
    'riqueza_hogar': ('control: origen social', 'Riqueza del hogar: coches (0 a 3) más baños (0 a 3), de 0 a 6'),
    'libros': ('control: origen social', 'Libros en casa por tramos: 1 = de 0 a 10, 2 = de 11 a 25, 3 = de 26 a 100, 4 = de 101 a 200, 5 = de 201 a 500, 6 = más de 500'),
    'sexo_femenino': ('control: alumno', '1 = chica, 0 = chico'),
    'inmigrante': ('control: alumno', '1 = inmigrante de primera o segunda generación, 0 = nativo'),
    'composicion_social_centro': ('control: centro', 'Composición social del centro: media en el centro del origen social de sus alumnos (ocupación y estudios de los padres tipificados), en desviaciones típicas de alumno'),
    'dotacion_centro': ('control: centro', 'Dotación del centro según el director (01_EDA_recursos_centro): alumnos por profesor, tamaño de clase y ordenadores por alumno. Media 0 y desviación típica 1; más = mejor dotado'),
    'escs_centro': ('comprobación', 'ESCS medio del centro (índice socioeconómico de la OCDE), alternativa a la composición social'),
    'escasez_centro': ('comprobación', 'Escasez de material y de personal según el director (en 2012, con otras preguntas). Media 0 y desviación típica 1; más = más escasez. Solo compara centros de una misma edición'),
    'deberes_en_u': ('comprobación', '1 = el alumno está en el tercil central de horas de deberes de su edición, 0 = en otro; componente del índice de compromiso'),
    'SKIPPING': ('comprobación', 'Absentismo: 1 = faltó un día entero o se saltó alguna clase en las dos últimas semanas, 0 = no; componente del índice'),
    'TARDYSD': ('comprobación', 'Impuntualidad en las dos últimas semanas: 0 = nunca, 1 = una o dos veces, 2 = tres o más; componente del índice'),
    'tercil_deberes': ('descriptiva', 'Tercil de horas de deberes dentro de la edición: 1 = pocas, 2 = central, 3 = muchas'),
    'tiempo_deberes_totales_diarios': ('descriptiva', 'Horas de deberes al día (máximo 4,5). La pregunta cambia en cada edición, así que el nivel no se compara entre ediciones'),
    'perfil_excluido': ('marca', 'El perfil no llega al mínimo de centros en ninguna edición: sus alumnos no entran en el modelo'),
    'perfil_poco_representativo': ('marca', 'El perfil tiene menos del mínimo de centros en esa edición: sus resultados en esa edición son poco fiables'),
    'con_variables_interes': ('marca', 'El alumno tiene el índice de compromiso y los recursos de estudio'),
    'en_modelo': ('marca', 'El alumno entra en el modelo: tiene las dos variables de interés y su perfil no está excluido'),
}
NOMBRE_DOMINIO = {'MATH': 'matemáticas', 'READ': 'lectura', 'SCIE': 'ciencias'}


def diccionario_columnas(datos: pd.DataFrame) -> pd.DataFrame:
    """Una fila por columna de `datos` (la tabla de `marcar_universo`) con su papel y su descripción."""
    filas = []
    for columna in datos.columns:
        if columna in DESCRIPCION_COLUMNAS:
            papel, descripcion = DESCRIPCION_COLUMNAS[columna]
        elif columna.startswith('PV') and columna[-4:] in NOMBRE_DOMINIO:
            papel = 'objetivo' if columna.endswith('MATH') else 'otras notas'
            descripcion = (f'Valor plausible {columna[2:-4]} de la nota de {NOMBRE_DOMINIO[columna[-4:]]} (escala PISA, media 500 y desviación '
                           'típica 100 en la OCDE de referencia). 2012 tiene 5, las demás 10: PV6 a PV10 vacíos en 2012')
        else:
            raise KeyError(f'Columna sin descripción: {columna}')
        filas.append({'columna': columna, 'papel': papel, 'descripcion': descripcion})
    return pd.DataFrame(filas)
