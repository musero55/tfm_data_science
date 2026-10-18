"""
pisa_resultados_publicados.py
=============================

Evolución de los resultados de PISA publicados por la OCDE (`data/raw/pisa_<asignatura>_2000_2022.csv`:
una fila por país, edición y asignatura, con la media, la desviación típica y los percentiles).
Ver `notebooks/fase_1_data_ingestion_eda/01_EDA_notas_asignatura.ipynb`.

Los datos se pasan en formato largo, con las columnas `year`, `country`, `subject` y la métrica
(`average`, `p50`, `std_dev`...). Los gráficos son de Plotly.

    * `preparar_evolucion`: filtra países y ordena; añade la media de las tres asignaturas.
    * `plot_evolucion_pais`: un país, sus tres asignaturas y su media.
    * `plot_comparar_paises`: varios países en una asignatura (o en la media de las tres).
    * `plot_comparar_paises_todas_materias`: varios países, un panel por asignatura.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

MATERIAS = ['mathematics', 'reading', 'science']
COLORES_MATERIAS = {'mathematics': '#636EFA', 'reading': '#EF553B', 'science': '#00CC96'}
MEDIA_TRES_MATERIAS = 'average_3subjects'   # valor de `subject` de la media de las tres asignaturas


def obtener_paleta(n_categorias: int) -> list[str]:
    """Al menos `n_categorias` colores distinguibles, para que `px.line(color='country')` no repita
    colores cuando hay muchos países (la paleta por defecto de Plotly solo tiene 10): hasta 10, la
    paleta por defecto; hasta 26, 'Alphabet'; con más, colores muestreados de la escala 'turbo'
    (sin repeticiones, aunque ya no todos se distinguen a simple vista)."""
    if n_categorias <= 10:
        return px.colors.qualitative.Plotly
    elif n_categorias <= 26:
        return px.colors.qualitative.Alphabet
    else:
        return px.colors.sample_colorscale(
            "turbo", [i / (n_categorias - 1) for i in range(n_categorias)]
        )


def preparar_evolucion(
    df: pd.DataFrame,
    paises: list,
    metrica: str = "average",
    incluir_media: bool = True,
) -> pd.DataFrame:
    """Datos de `metrica` de los `paises` (nombres exactos), ordenados por país, asignatura y año.

    Con `incluir_media` añade filas con `subject` = 'average_3subjects': la media de las tres
    asignaturas de cada país y año, solo cuando las tres tienen dato (si falta alguna no se
    calcula, para no sesgar el promedio). Devuelve las columnas year, country, subject y `metrica`.
    """
    faltantes = set(paises) - set(df["country"].unique())
    if faltantes:
        raise ValueError(
            f"País(es) no encontrados: {faltantes}. "
            f"Los nombres deben coincidir exactamente con la columna `country`."
        )

    df_filtrado = df[df["country"].isin(paises) & df["subject"].isin(MATERIAS)].copy()
    df_filtrado = df_filtrado.dropna(subset=[metrica])
    df_filtrado = df_filtrado[["year", "country", "subject", metrica]]

    if incluir_media:
        pivot = df_filtrado.pivot_table(
            index=["year", "country"], columns="subject", values=metrica
        )
        # Solo calcular la media donde están las 3 materias disponibles
        pivot_completo = pivot.dropna(subset=MATERIAS)
        media = pivot_completo[MATERIAS].mean(axis=1).reset_index()
        media.columns = ["year", "country", metrica]
        media["subject"] = MEDIA_TRES_MATERIAS
        media = media[["year", "country", "subject", metrica]]

        df_filtrado = pd.concat([df_filtrado, media], ignore_index=True)

    return df_filtrado.sort_values(["country", "subject", "year"])


def plot_evolucion_pais(
    df: pd.DataFrame,
    pais: str,
    metrica: str = "average",
    mostrar_media: bool = True,
) -> go.Figure:
    """Evolución de `metrica` en un país (nombre exacto): una línea por asignatura y, con
    `mostrar_media`, otra discontinua con la media de las tres."""
    datos = preparar_evolucion(df, [pais], metrica=metrica, incluir_media=mostrar_media)

    fig = go.Figure()

    for subject in MATERIAS:
        sub = datos[datos["subject"] == subject]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["year"], y=sub[metrica],
                mode="lines+markers",
                name=subject.capitalize(),
                line=dict(color=COLORES_MATERIAS.get(subject)),
            )
        )

    if mostrar_media:
        sub = datos[datos["subject"] == MEDIA_TRES_MATERIAS]
        if not sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub["year"], y=sub[metrica],
                    mode="lines+markers",
                    name="Media (3 materias)",
                    line=dict(color="black", dash="dash", width=3),
                )
            )

    fig.update_layout(
        title=f"Evolución PISA — {pais} ({metrica})",
        xaxis_title="Año", yaxis_title=metrica,
        legend_title="Materia",
        template="plotly_white", hovermode="x unified",
    )
    return fig


def plot_comparar_paises(
    df: pd.DataFrame,
    paises: list,
    materia: str = "mathematics",
    metrica: str = "average",
) -> go.Figure:
    """Evolución de `metrica` en varios países para una asignatura ('mathematics', 'reading',
    'science') o para la media de las tres ('average_3subjects')."""
    incluir_media = materia == MEDIA_TRES_MATERIAS
    datos = preparar_evolucion(df, paises, metrica=metrica, incluir_media=incluir_media)
    datos = datos[datos["subject"] == materia]

    titulo_subject = "Media (3 materias)" if incluir_media else materia.capitalize()
    paleta = obtener_paleta(datos["country"].nunique())

    fig = px.line(
        datos, x="year", y=metrica, color="country", markers=True,
        template="plotly_white",
        color_discrete_sequence=paleta,
        title=f"Comparativa entre países — {titulo_subject} ({metrica})",
    )
    fig.update_layout(xaxis_title="Año", yaxis_title=metrica, legend_title="País", hovermode="x unified")
    return fig


def plot_comparar_paises_todas_materias(
    df: pd.DataFrame,
    paises: list,
    metrica: str = "average",
    incluir_media: bool = True,
) -> go.Figure:
    """Evolución de `metrica` en varios países, con un panel por asignatura y, con
    `incluir_media`, otro con la media de las tres."""
    datos = preparar_evolucion(df, paises, metrica=metrica, incluir_media=incluir_media)

    orden = MATERIAS + ([MEDIA_TRES_MATERIAS] if incluir_media else [])
    datos = datos[datos["subject"].isin(orden)]
    paleta = obtener_paleta(datos["country"].nunique())

    fig = px.line(
        datos, x="year", y=metrica, color="country", facet_col="subject",
        markers=True, template="plotly_white",
        category_orders={"subject": orden},
        color_discrete_sequence=paleta,
        title=f"Comparativa entre países por materia ({metrica})",
    )
    fig.update_layout(legend_title="País", hovermode="x unified")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].replace("_", " ").capitalize()))
    return fig
