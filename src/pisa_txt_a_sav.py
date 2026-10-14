"""
pisa_txt_a_sav.py
=================

Convierte las bases PISA antiguas (2000-2012: `.txt` de ancho fijo + script `.sas`
con las posiciones) a un `.sav` que `pisa_pipeline.py` sabe leer. Del `.sas` se
extraen posiciones, tipos, etiquetas y códigos de no-respuesta; estos últimos
pasan a NaN, como hace pyreadstat con los missings de SAS en 2015-2022.

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
    posicion = inicio
    while posicion < len(lineas) and ';' not in lineas[posicion].split('"')[0]:
        salida.append(lineas[posicion])
        posicion += 1
    return salida, posicion


def parsear_script_sas(ruta_sas: str) -> dict:
    """Extrae del `.sas` el esquema del `.txt`: dict con `columnas` ((nombre, inicio,
    fin), base 1), `es_texto`, `etiquetas`, `codigos_faltantes`, `etiquetas_valor`
    y `ancho_linea`."""
    with open(ruta_sas, encoding='utf-8-sig', errors='replace') as fichero:
        lineas = [linea.rstrip('\r\n') for linea in fichero]

    # --- PROC FORMAT: value NOMBRE ... ; -> {formato: {valor: etiqueta}}
    formatos: dict[str, dict] = {}
    posicion_linea = 0
    while posicion_linea < len(lineas):
        coincidencia = re.match(r'^\s*value\s+(\$?\w+)\s*$', lineas[posicion_linea], flags=re.I)
        if coincidencia:
            nombre = coincidencia.group(1).upper()
            cuerpo, posicion_linea = _bloque(lineas, posicion_linea + 1)
            entradas = {}
            for linea in cuerpo:
                coincidencia_entrada = re.match(r'^\s*("([^"]*)"|-?\d+(?:\.\d+)?)\s*=\s*"(.*)"\s*$', linea)
                if coincidencia_entrada:
                    clave = coincidencia_entrada.group(2) if coincidencia_entrada.group(2) is not None else float(coincidencia_entrada.group(1))
                    entradas[clave] = coincidencia_entrada.group(3).strip()
            formatos[nombre] = entradas
        posicion_linea += 1

    # --- length: tipo (texto si lleva $)
    es_texto: set[str] = set()
    idx_length = next(posicion for posicion, linea in enumerate(lineas) if re.match(r'^\s*length\s*$', linea, flags=re.I))
    cuerpo, _ = _bloque(lineas, idx_length + 1)
    for linea in cuerpo:
        coincidencia = re.match(r'^\s*(\w+)\s+(\$)?\s*\d+\s*$', linea)
        if coincidencia and coincidencia.group(2):
            es_texto.add(coincidencia.group(1))

    # --- infile / input
    ancho_linea = None
    idx_input = None
    for posicion, linea in enumerate(lineas):
        coincidencia = re.search(r'linesize\s*=\s*(\d+)', linea, flags=re.I)
        if coincidencia:
            ancho_linea = int(coincidencia.group(1))
        if re.match(r'^\s*input\s*$', linea, flags=re.I):
            idx_input = posicion
    cuerpo, _ = _bloque(lineas, idx_input + 1)
    columnas = []
    for linea in cuerpo:
        coincidencia = re.match(r'^\s*(\w+)\s+\$?\s*(\d+)(?:\s*-\s*(\d+))?\s*$', linea)
        if coincidencia:
            columna_inicio = int(coincidencia.group(2))
            columna_fin = int(coincidencia.group(3)) if coincidencia.group(3) else columna_inicio
            columnas.append((coincidencia.group(1), columna_inicio, columna_fin))

    # --- label y format de valores (el que viene después de label)
    etiquetas: dict[str, str] = {}
    idx_label = next(posicion for posicion, linea in enumerate(lineas) if posicion > idx_input and re.match(r'^\s*label\s*$', linea, flags=re.I))
    cuerpo, fin_label = _bloque(lineas, idx_label + 1)
    for linea in cuerpo:
        coincidencia = re.match(r'^\s*(\w+)\s*=\s*"(.*)"\s*$', linea)
        if coincidencia:
            etiquetas[coincidencia.group(1).upper()] = coincidencia.group(2).strip()

    etiquetas_valor: dict[str, dict] = {}
    idx_fmt = next((posicion for posicion in range(fin_label, len(lineas)) if re.match(r'^\s*format\s*$', lineas[posicion], flags=re.I)), None)
    if idx_fmt is not None:
        cuerpo, _ = _bloque(lineas, idx_fmt + 1)
        for linea in cuerpo:
            coincidencia = re.match(r'^\s*(\w+)\s+(\$?\w+)\.\s*$', linea)
            if coincidencia and coincidencia.group(2).upper() in formatos:
                etiquetas_valor[coincidencia.group(1)] = formatos[coincidencia.group(2).upper()]

    # --- recodificación de no-respuesta: if X=a or X=. then X=.N; if X=b ... .I; if X=c ... .M;
    codigos: dict[str, list] = {}
    patron = re.compile(r'if\s+(\w+)=(\d+)\s+or\s+\1=\.\s+then\s+\1=\.N;\s*'
                        r'if\s+\1=(\d+)\s+then\s+\1=\.I;\s*if\s+\1=(\d+)\s+then\s+\1=\.M;')
    for linea in lineas:
        coincidencia = patron.search(linea)
        if coincidencia:
            codigos[coincidencia.group(1)] = [float(coincidencia.group(2)), float(coincidencia.group(3)), float(coincidencia.group(4))]

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
    with open(ruta_txt, 'rb') as fichero:
        fichero.seek(ancho)
        fin_linea = fichero.read(2)
    terminador = 2 if fin_linea == b'\r\n' else 1
    largo = ancho + terminador
    tamaño_fichero = os.path.getsize(ruta_txt)
    n_filas, resto = divmod(tamaño_fichero, largo)
    if resto:
        raise ValueError(f'El tamaño de {ruta_txt} ({tamaño_fichero}) no es múltiplo de la longitud de línea ({largo}).')

    mapa_memoria = np.memmap(ruta_txt, dtype=np.uint8, mode='r', shape=(n_filas, largo))
    columnas = esquema['columnas']
    es_texto = esquema['es_texto']
    columnas_numericas = [columna_esquema for columna_esquema in columnas if columna_esquema[0] not in es_texto]
    columnas_texto = [columna_esquema for columna_esquema in columnas if columna_esquema[0] in es_texto]

    matriz = np.empty((n_filas, len(columnas_numericas)), dtype=np.float64)
    textos = {nombre: np.empty(n_filas, dtype=object) for nombre, _, _ in columnas_texto}
    ilegibles = {}

    for fila_inicio in range(0, n_filas, chunk_filas):
        fila_fin = min(fila_inicio + chunk_filas, n_filas)
        bloque = mapa_memoria[fila_inicio:fila_fin]
        for posicion_numerica, (nombre, columna_inicio, columna_fin) in enumerate(columnas_numericas):
            campo = np.ascontiguousarray(bloque[:, columna_inicio - 1:columna_fin]).view(f'S{columna_fin - columna_inicio + 1}').ravel()
            crudo = np.char.decode(campo, 'latin-1')
            valores = pd.to_numeric(pd.Series(crudo), errors='coerce').to_numpy()
            n_no_numericos = int(np.sum(np.isnan(valores) & (np.char.strip(crudo) != '')))
            if n_no_numericos:
                ilegibles[nombre] = ilegibles.get(nombre, 0) + n_no_numericos
            matriz[fila_inicio:fila_fin, posicion_numerica] = valores
        for nombre, columna_inicio, columna_fin in columnas_texto:
            campo = np.ascontiguousarray(bloque[:, columna_inicio - 1:columna_fin]).view(f'S{columna_fin - columna_inicio + 1}').ravel()
            textos[nombre][fila_inicio:fila_fin] = np.char.strip(np.char.decode(campo, 'latin-1'))

    if ilegibles:
        print(f'  Aviso: valores no numéricos convertidos a NaN: {ilegibles}')

    # Códigos de no-respuesta -> NaN
    indice = {nombre: posicion_numerica for posicion_numerica, (nombre, _, _) in enumerate(columnas_numericas)}
    for nombre, codigos in esquema['codigos_faltantes'].items():
        if nombre in indice:
            valores_columna = matriz[:, indice[nombre]]
            valores_columna[np.isin(valores_columna, codigos)] = np.nan

    datos = {}
    for nombre, _, _ in columnas:
        if nombre in es_texto:
            datos[nombre] = textos[nombre]
        else:
            datos[nombre] = matriz[:, indice[nombre]]
    return pd.DataFrame(datos)


def convertir_txt_a_sav(ruta_txt: str, ruta_sas: str, ruta_sav: str, chunk_filas: int = 50_000) -> pd.DataFrame:
    """Convierte el `.txt` (+ su `.sas`) a `.sav` con etiquetas y devuelve el DataFrame."""
    esquema = parsear_script_sas(ruta_sas)
    print(f'{os.path.basename(ruta_sas)}: {len(esquema["columnas"])} columnas, '
          f'{len(esquema["es_texto"])} de texto, {len(esquema["codigos_faltantes"])} con códigos de no-respuesta, '
          f'{len(esquema["etiquetas_valor"])} con etiquetas de valor')
    datos = leer_ancho_fijo(ruta_txt, esquema, chunk_filas)
    print(f'  Leídas {len(datos):,} filas × {datos.shape[1]} columnas')

    etiquetas_col = [esquema['etiquetas'].get(columna.upper(), '')[:255] for columna in datos.columns]

    etiquetas_valor = {}
    for nombre, mapa in esquema['etiquetas_valor'].items():
        if nombre not in datos.columns:
            continue
        if nombre in esquema['es_texto']:
            validas = {k: v[:120] for k, v in mapa.items() if isinstance(k, str) and len(k.encode('latin-1', 'replace')) <= 8}
        else:
            validas = {float(k): v[:120] for k, v in mapa.items() if not isinstance(k, str)}
        if validas:
            etiquetas_valor[nombre] = validas

    os.makedirs(os.path.dirname(os.path.abspath(ruta_sav)), exist_ok=True)
    pyreadstat.write_sav(datos, ruta_sav, column_labels=etiquetas_col, variable_value_labels=etiquetas_valor)
    print(f'  Guardado: {ruta_sav} ({os.path.getsize(ruta_sav) / 1e6:.0f} MB)')
    return datos
