# -*- coding: utf-8 -*-
"""Genera el pipeline/interfaz de Inkriff (Gradio): consume los CSVs de recomendaciones
ya generados por inkriff_modelado_similitud.ipynb (no vuelve a calcular nada, ni llama
ninguna API) y expone una interfaz con dos flujos: "tengo un libro -> recomiendame
bandas" y "tengo una banda -> recomiendame libros". Produce dos artefactos que
comparten exactamente la misma logica:

1. inkriff_pipeline_recomendador.ipynb -- para correr en Colab con un link publico
   (gr.Interface(share=True)), consistente con el resto del proyecto.
2. app.py + requirements.txt -- para correr la interfaz localmente (python app.py).
"""
import json

from descarga_datos import celda_descarga

CORE_LOGIC = '''import pandas as pd
import gradio as gr

# --- Carga de datos: solo los CSVs ya generados, no se recalcula ni se llama a ninguna API ---
rec_banda_a_libro = pd.read_csv("recomendaciones_banda_a_libro.csv")
rec_libro_a_banda = pd.read_csv("recomendaciones_libro_a_banda.csv")
bandas = pd.read_csv("bandas_completo.csv")
libros = pd.read_csv("libros_con_sinopsis.csv")

BANDAS_DISPONIBLES = sorted(rec_banda_a_libro["banda"].unique().tolist())
LIBROS_DISPONIBLES = sorted(rec_libro_a_banda["libro"].unique().tolist())

SPOTIFY_POR_BANDA = dict(zip(bandas["banda"], bandas["spotify_url"]))
SINOPSIS_POR_LIBRO = dict(zip(libros["titulo_buscado"], libros["sinopsis"]))


def spotify_embed_html(spotify_url):
    """Reproductor embebido de Spotify sin necesitar API key: basta con reescribir
    el link publico de artista al formato /embed/ y ponerlo en un iframe (lo mismo
    que se documento en el canvas del proyecto: MusicBrainz -> link de Spotify ->
    reproductor publico)."""
    if not isinstance(spotify_url, str) or "open.spotify.com" not in spotify_url:
        return "<p><em>Sin enlace de Spotify disponible para esta banda.</em></p>"
    embed_url = spotify_url.replace("open.spotify.com/", "open.spotify.com/embed/")
    return (
        f'<iframe src="{embed_url}" width="100%" height="152" frameborder="0" '
        f'allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" '
        f'loading="lazy"></iframe>'
    )


def recomendar_bandas_para_libro(titulo_libro):
    if not titulo_libro:
        return pd.DataFrame(), "", ""
    fila = rec_libro_a_banda[rec_libro_a_banda["libro"] == titulo_libro].sort_values("rank")
    if fila.empty:
        return pd.DataFrame(), "", "No hay recomendaciones para ese libro."
    tabla = fila[["rank", "banda_recomendada", "subgenero_banda", "similitud"]].rename(columns={
        "rank": "#", "banda_recomendada": "Banda", "subgenero_banda": "Subgenero", "similitud": "Similitud",
    })
    aviso = ""
    if not bool(fila.iloc[0]["libro_perfil_confiable"]):
        aviso = (
            "Nota: este libro no tiene sinopsis disponible, asi que estas recomendaciones se basan "
            "solo en su categoria (no en el contenido de la trama) -- tomalas con mas reserva."
        )
    top_banda = fila.iloc[0]["banda_recomendada"]
    embed = spotify_embed_html(SPOTIFY_POR_BANDA.get(top_banda))
    return tabla, embed, aviso


def recomendar_libros_para_banda(nombre_banda):
    if not nombre_banda:
        return pd.DataFrame(), ""
    fila = rec_banda_a_libro[rec_banda_a_libro["banda"] == nombre_banda].sort_values("rank")
    if fila.empty:
        return pd.DataFrame(), "No hay recomendaciones para esa banda."
    tabla = fila[["rank", "libro_recomendado", "categoria_libro", "similitud"]].rename(columns={
        "rank": "#", "libro_recomendado": "Libro", "categoria_libro": "Categoria", "similitud": "Similitud",
    })
    top_libro = fila.iloc[0]["libro_recomendado"]
    sinopsis = SINOPSIS_POR_LIBRO.get(top_libro)
    if isinstance(sinopsis, str) and sinopsis.strip():
        extracto = sinopsis.strip()[:400] + ("..." if len(sinopsis.strip()) > 400 else "")
        detalle = f"**{top_libro}** -- sinopsis:\\n\\n{extracto}"
    else:
        detalle = f"**{top_libro}** -- no hay sinopsis disponible para este libro."
    return tabla, detalle


with gr.Blocks(title="Inkriff -- Recomendador musica <-> libros") as demo:
    gr.Markdown(
        "# Inkriff\\n"
        "Recomendador bidireccional entre rock/metal y fantasia/romance. "
        "Elige un libro para que te recomiende bandas, o una banda para que te "
        "recomiende libros -- ambos usan el mismo modelo de similitud de contenido "
        "(embeddings + correccion de hubness), ya evaluado contra 105 pares "
        "curados a mano."
    )
    with gr.Tab("Tengo un libro"):
        libro_input = gr.Dropdown(choices=LIBROS_DISPONIBLES, label="Elige un libro", filterable=True)
        boton_libro = gr.Button("Recomendar bandas", variant="primary")
        aviso_libro = gr.Markdown()
        tabla_bandas = gr.Dataframe(label="Top-5 bandas recomendadas", interactive=False)
        reproductor = gr.HTML(label="Escucha a la banda #1")
        boton_libro.click(
            recomendar_bandas_para_libro, inputs=libro_input,
            outputs=[tabla_bandas, reproductor, aviso_libro],
        )
    with gr.Tab("Tengo una banda"):
        banda_input = gr.Dropdown(choices=BANDAS_DISPONIBLES, label="Elige una banda", filterable=True)
        boton_banda = gr.Button("Recomendar libros", variant="primary")
        tabla_libros = gr.Dataframe(label="Top-5 libros recomendados", interactive=False)
        detalle_libro = gr.Markdown()
        boton_banda.click(
            recomendar_libros_para_banda, inputs=banda_input,
            outputs=[tabla_libros, detalle_libro],
        )
    gr.Markdown(
        "---\\n"
        "*Nota metodologica: las recomendaciones vienen de similitud de contenido "
        "(no de un modelo entrenado con estos mismos pares), evaluada con Recall@10 "
        "y MRR contra 105 pares de referencia. Ver el documento de metodologia para "
        "el detalle completo, incluyendo limitaciones.*"
    )
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": text.splitlines(keepends=True),
    }


# ============================== 1. Notebook para Colab ==============================
cells = []

cells.append(md("""# Inkriff — Pipeline / interfaz de uso (Gradio)

Este notebook expone el recomendador ya entrenado y evaluado (`inkriff_modelado_similitud.ipynb`)
en una interfaz simple: eliges un libro y te recomienda bandas, o eliges una banda y te
recomienda libros. **No vuelve a calcular ninguna similitud ni llama a ninguna API** — solo
lee los CSVs de recomendaciones que ya generó ese notebook, así que corre en segundos.

Usa estos 4 archivos (la celda de carga los descarga de GitHub si no están): `recomendaciones_banda_a_libro.csv`,
`recomendaciones_libro_a_banda.csv`, `bandas_completo.csv` (para el enlace de Spotify) y
`libros_con_sinopsis.csv` (para mostrar un extracto de la sinopsis del libro recomendado).

Al correr la última celda, Gradio genera un **link público temporal** (`share=True`) para
que puedas abrir la interfaz en el navegador o compartirla — el mismo mecanismo que usa la
versión final del chat.
"""))

cells.append(code('''!pip -q install gradio
'''))

cells.append(code(celda_descarga(["recomendaciones_banda_a_libro.csv", "recomendaciones_libro_a_banda.csv",
                                   "bandas_completo.csv", "libros_con_sinopsis.csv"])))

cells.append(md("## Interfaz\n"))
cells.append(code(CORE_LOGIC))

cells.append(code('''demo.launch(share=True)
'''))

cells.append(md("""## Notas

- Esta interfaz es deliberadamente simple (dropdown + tabla + reproductor embebido) para
  cubrir el requisito de "Interfaz de uso" del Trabajo Final sin añadir dependencias nuevas
  ni volver a tocar ninguna API externa.
- El reproductor de Spotify se arma reescribiendo el link público de artista al formato
  `/embed/` — no requiere una API key de Spotify, tal como se documentó en el canvas del
  proyecto.
- Si más adelante se agregan más libros/bandas al universo semilla, solo hace falta volver
  a correr `inkriff_extraccion_completa.ipynb` → `inkriff_eda_ingenieria.ipynb` →
  `inkriff_modelado_similitud.ipynb`, y esta interfaz recoge los CSVs nuevos sin cambios.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("inkriff_pipeline_recomendador.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Notebook generado: inkriff_pipeline_recomendador.ipynb")

# ============================== 2. app.py para correr localmente ===============================
APP_PY = '''"""Inkriff -- Recomendador bidireccional musica <-> libros (version 1, local).

Consume los CSVs ya generados por inkriff_modelado_similitud.ipynb -- no recalcula
similitudes ni llama a ninguna API externa en tiempo de ejecucion (salvo el reproductor
embebido de Spotify, que es un iframe publico sin API key). Sube junto a este archivo:
recomendaciones_banda_a_libro.csv, recomendaciones_libro_a_banda.csv,
bandas_completo.csv y libros_con_sinopsis.csv.
"""
''' + CORE_LOGIC + '''

if __name__ == "__main__":
    demo.launch()
'''

with open("app.py", "w", encoding="utf-8") as f:
    f.write(APP_PY)
print("Archivo generado: app.py")

with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write("gradio>=4.0\npandas\n")
print("Archivo generado: requirements.txt")
