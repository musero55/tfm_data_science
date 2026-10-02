"""
pisa_txt_a_sav.py
=================

Conversión de las bases PISA antiguas (2000-2012), que la OCDE distribuye como
fichero de texto de ancho fijo (.txt) más un programa SAS (.sas) con las
posiciones de cada columna, a un `.sav` que `src/pisa_pipeline.py` ya sabe leer.

Del script .sas se extraen: posiciones (INPUT), tipo texto/numérico (LENGTH),
etiquetas de variable (LABEL), etiquetas de valor (PROC FORMAT) y los códigos de
no-respuesta (el script recodifica 7/8/9, 97/98/99... a .N/.I/.M). Aquí todos
esos códigos pasan a NaN, igual que pyreadstat hace con los missings especiales
de SAS en las ediciones 2015-2022.

Uso:
    from src.pisa_txt_a_sav import convertir_txt_a_sav
    convertir_txt_a_sav('INT_STU12_DEC03.txt', 'PISA2012_SAS_student.sas',
                        'data/raw/2012/INT_STU12_DEC03.sav')
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import pyreadstat


def _bloque(lineas: list[str], inicio: int) -> tuple[list[str], int]:
    """Líneas desde `inicio` hasta el primer ';' (sin incluir la línea con ';')."""
    salida = []
    i = inicio
    while i < len(lineas) and ';' not in lineas[i].split('"')[0]:
        salida.append(lineas[i])
        i += 1
    return salida, i


def parsear_script_sas(ruta_sas: str) -> dict:
    """Extrae del programa SAS toda la información necesaria para leer el .txt.

    Devuelve un dict con: `columnas` (lista ordenada de (nombre, inicio, fin),
    posiciones base 1 y fin incluido), `es_texto` (set de nombres de columnas
    de texto), `etiquetas` ({NOMBRE_EN_MAYUSCULAS: etiqueta}), `codigos_faltantes`
    ({nombre: [no aplica, inválido, sin respuesta]}), `etiquetas_valor`
    ({nombre: {valor: etiqueta}}) y `ancho_linea`.
    """
    with open(ruta_sas, encoding='utf-8-sig', errors='replace') as f:
        lineas = [l.rstrip('\r\n') for l in f]

    # --- PROC FORMAT: value NOMBRE ... ; -> {formato: {valor: etiqueta}}
    formatos: dict[str, dict] = {}
    i = 0
    while i < len(lineas):
        m = re.match(r'^\s*value\s+(\$?\w+)\s*$', lineas[i], flags=re.I)
        if m:
            nombre = m.group(1).upper()
            cuerpo, i = _bloque(lineas, i + 1)
            entradas = {}
            for l in cuerpo:
                e = re.match(r'^\s*("([^"]*)"|-?\d+(?:\.\d+)?)\s*=\s*"(.*)"\s*$', l)
                if e:
                    clave = e.group(2) if e.group(2) is not None else float(e.group(1))
                    entradas[clave] = e.group(3).strip()
            formatos[nombre] = entradas
        i += 1

    # --- length: tipo (texto si lleva $)
    es_texto: set[str] = set()
    idx_length = next(k for k, l in enumerate(lineas) if re.match(r'^\s*length\s*$', l, flags=re.I))
    cuerpo, _ = _bloque(lineas, idx_length + 1)
    for l in cuerpo:
        m = re.match(r'^\s*(\w+)\s+(\$)?\s*\d+\s*$', l)
        if m and m.group(2):
            es_texto.add(m.group(1))

    # --- infile / input
    ancho_linea = None
    idx_input = None
    for k, l in enumerate(lineas):
        m = re.search(r'linesize\s*=\s*(\d+)', l, flags=re.I)
        if m:
            ancho_linea = int(m.group(1))
        if re.match(r'^\s*input\s*$', l, flags=re.I):
            idx_input = k
    cuerpo, _ = _bloque(lineas, idx_input + 1)
    columnas = []
    for l in cuerpo:
        m = re.match(r'^\s*(\w+)\s+\$?\s*(\d+)(?:\s*-\s*(\d+))?\s*$', l)
        if m:
            ini = int(m.group(2))
            fin = int(m.group(3)) if m.group(3) else ini
            columnas.append((m.group(1), ini, fin))

    # --- label y format de valores (el que viene después de label)
    etiquetas: dict[str, str] = {}
    idx_label = next(k for k, l in enumerate(lineas) if k > idx_input and re.match(r'^\s*label\s*$', l, flags=re.I))
    cuerpo, fin_label = _bloque(lineas, idx_label + 1)
    for l in cuerpo:
        m = re.match(r'^\s*(\w+)\s*=\s*"(.*)"\s*$', l)
        if m:
            etiquetas[m.group(1).upper()] = m.group(2).strip()

    etiquetas_valor: dict[str, dict] = {}
    idx_fmt = next((k for k in range(fin_label, len(lineas)) if re.match(r'^\s*format\s*$', lineas[k], flags=re.I)), None)
    if idx_fmt is not None:
        cuerpo, _ = _bloque(lineas, idx_fmt + 1)
        for l in cuerpo:
            m = re.match(r'^\s*(\w+)\s+(\$?\w+)\.\s*$', l)
            if m and m.group(2).upper() in formatos:
                etiquetas_valor[m.group(1)] = formatos[m.group(2).upper()]

    # --- recodificación de no-respuesta: if X=a or X=. then X=.N; if X=b ... .I; if X=c ... .M;
    codigos: dict[str, list] = {}
    patron = re.compile(r'if\s+(\w+)=(\d+)\s+or\s+\1=\.\s+then\s+\1=\.N;\s*'
                        r'if\s+\1=(\d+)\s+then\s+\1=\.I;\s*if\s+\1=(\d+)\s+then\s+\1=\.M;')
    for l in lineas:
        m = patron.search(l)
        if m:
            codigos[m.group(1)] = [float(m.group(2)), float(m.group(3)), float(m.group(4))]

    return {
        'columnas': columnas,
        'es_texto': es_texto,
        'etiquetas': etiquetas,
        'codigos_faltantes': codigos,
        'etiquetas_valor': etiquetas_valor,
        'ancho_linea': ancho_linea,
    }


def leer_ancho_fijo(ruta_txt: str, esquema: dict, chunk_filas: int = 50_000) -> pd.DataFrame:
    """Lee el .txt de ancho fijo según `esquema` (salida de `parsear_script_sas`)."""
    ancho = esquema['ancho_linea']
    with open(ruta_txt, 'rb') as f:
        f.seek(ancho)
        fin_linea = f.read(2)
    terminador = 2 if fin_linea == b'\r\n' else 1
    largo = ancho + terminador
    tam = os.path.getsize(ruta_txt)
    n_filas, resto = divmod(tam, largo)
    if resto:
        raise ValueError(f'El tamaño de {ruta_txt} ({tam}) no es múltiplo de la longitud de línea ({largo}).')

    mm = np.memmap(ruta_txt, dtype=np.uint8, mode='r', shape=(n_filas, largo))
    columnas = esquema['columnas']
    es_texto = esquema['es_texto']
    cols_num = [c for c in columnas if c[0] not in es_texto]
    cols_txt = [c for c in columnas if c[0] in es_texto]

    matriz = np.empty((n_filas, len(cols_num)), dtype=np.float64)
    textos = {nombre: np.empty(n_filas, dtype=object) for nombre, _, _ in cols_txt}
    ilegibles = {}

    for ini_f in range(0, n_filas, chunk_filas):
        fin_f = min(ini_f + chunk_filas, n_filas)
        bloque = mm[ini_f:fin_f]
        for k, (nombre, a, b) in enumerate(cols_num):
            campo = np.ascontiguousarray(bloque[:, a - 1:b]).view(f'S{b - a + 1}').ravel()
            crudo = np.char.decode(campo, 'latin-1')
            valores = pd.to_numeric(pd.Series(crudo), errors='coerce').to_numpy()
            n_mal = int(np.sum(np.isnan(valores) & (np.char.strip(crudo) != '')))
            if n_mal:
                ilegibles[nombre] = ilegibles.get(nombre, 0) + n_mal
            matriz[ini_f:fin_f, k] = valores
        for nombre, a, b in cols_txt:
            campo = np.ascontiguousarray(bloque[:, a - 1:b]).view(f'S{b - a + 1}').ravel()
            textos[nombre][ini_f:fin_f] = np.char.strip(np.char.decode(campo, 'latin-1'))

    if ilegibles:
        print(f'  Aviso: valores no numéricos convertidos a NaN: {ilegibles}')

    # Códigos de no-respuesta -> NaN
    indice = {nombre: k for k, (nombre, _, _) in enumerate(cols_num)}
    for nombre, cods in esquema['codigos_faltantes'].items():
        if nombre in indice:
            col = matriz[:, indice[nombre]]
            col[np.isin(col, cods)] = np.nan

    datos = {}
    for nombre, _, _ in columnas:
        if nombre in es_texto:
            datos[nombre] = textos[nombre]
        else:
            datos[nombre] = matriz[:, indice[nombre]]
    return pd.DataFrame(datos)


def convertir_txt_a_sav(ruta_txt: str, ruta_sas: str, ruta_sav: str, chunk_filas: int = 50_000) -> pd.DataFrame:
    """Convierte un .txt de ancho fijo PISA (+ su script .sas) a .sav con etiquetas.

    Devuelve el DataFrame resultante para poder comprobarlo sin releer el fichero.
    """
    esquema = parsear_script_sas(ruta_sas)
    print(f'{os.path.basename(ruta_sas)}: {len(esquema["columnas"])} columnas, '
          f'{len(esquema["es_texto"])} de texto, {len(esquema["codigos_faltantes"])} con códigos de no-respuesta, '
          f'{len(esquema["etiquetas_valor"])} con etiquetas de valor')
    df = leer_ancho_fijo(ruta_txt, esquema, chunk_filas)
    print(f'  Leídas {len(df):,} filas × {df.shape[1]} columnas')

    etiquetas_col = [esquema['etiquetas'].get(c.upper(), '')[:255] for c in df.columns]

    etiquetas_valor = {}
    for nombre, mapa in esquema['etiquetas_valor'].items():
        if nombre not in df.columns:
            continue
        if nombre in esquema['es_texto']:
            validas = {k: v[:120] for k, v in mapa.items() if isinstance(k, str) and len(k.encode('latin-1', 'replace')) <= 8}
        else:
            validas = {float(k): v[:120] for k, v in mapa.items() if not isinstance(k, str)}
        if validas:
            etiquetas_valor[nombre] = validas

    os.makedirs(os.path.dirname(os.path.abspath(ruta_sav)), exist_ok=True)
    pyreadstat.write_sav(df, ruta_sav, column_labels=etiquetas_col, variable_value_labels=etiquetas_valor)
    print(f'  Guardado: {ruta_sav} ({os.path.getsize(ruta_sav) / 1e6:.0f} MB)')
    return df
