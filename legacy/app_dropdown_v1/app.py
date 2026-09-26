"""Inkriff -- Recomendador bidireccional musica <-> libros (Hugging Face Spaces).

Consume los CSVs ya generados por inkriff_modelado_similitud.ipynb -- no recalcula
similitudes ni llama a ninguna API externa en tiempo de ejecucion (salvo el reproductor
embebido de Spotify, que es un iframe publico sin API key). Sube junto a este archivo:
recomendaciones_banda_a_libro.csv, recomendaciones_libro_a_banda.csv,
bandas_completo.csv y libros_con_sinopsis.csv.
"""
import pandas as pd
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
        detalle = f"**{top_libro}** -- sinopsis:\n\n{extracto}"
    else:
        detalle = f"**{top_libro}** -- no hay sinopsis disponible para este libro."
    return tabla, detalle


with gr.Blocks(title="Inkriff -- Recomendador musica <-> libros") as demo:
    gr.Markdown(
        "# Inkriff\n"
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
        "---\n"
        "*Nota metodologica: las recomendaciones vienen de similitud de contenido "
        "(no de un modelo entrenado con estos mismos pares), evaluada con Recall@10 "
        "y MRR contra 105 pares de referencia. Ver el documento de metodologia para "
        "el detalle completo, incluyendo limitaciones.*"
    )


if __name__ == "__main__":
    demo.launch()
