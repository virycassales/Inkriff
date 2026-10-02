"""Inkriff -- Chat conversacional (Hugging Face Spaces).

Sube junto a este archivo: bandas_completo.csv, libros_con_sinopsis.csv,
recomendaciones_banda_a_libro.csv, recomendaciones_libro_a_banda.csv y
embeddings_referencia.npz (este ultimo lo genera inkriff_modelado_similitud.ipynb).

Configura en Settings -> Repository secrets del Space: OPENAI_API_KEY (proveedor principal)
y, opcional pero recomendado, GROQ_API_KEY (gratis, en https://console.groq.com/keys) como
respaldo automatico si algo falla con OpenAI. Si solo quieres usar la gratuita, define
INKRIFF_LLM_PROVEEDOR=groq y basta con GROQ_API_KEY.
"""
import ast
import base64
import csv
import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from functools import lru_cache
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import gradio as gr
from openai import OpenAI


LLM_PROVEEDOR_PRIMARIO = os.environ.get("INKRIFF_LLM_PROVEEDOR", "openai")  # "openai" o "groq"
LLM_PROVEEDOR_RESPALDO = "groq" if LLM_PROVEEDOR_PRIMARIO == "openai" else "openai"

LLM_CLIENTES = {
    # api_key="sin-configurar" evita que falle al importar si todavia no configuras la key;
    # _llamar_llm() revisa por separado si hay al menos una key real antes de usarla.
    "openai": OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "sin-configurar"),
    "groq": OpenAI(api_key=os.environ.get("GROQ_API_KEY") or "sin-configurar", base_url="https://api.groq.com/openai/v1"),
}
LLM_MODELOS = {
    "openai": os.environ.get("INKRIFF_LLM_MODELO_OPENAI", "gpt-4o-mini"),
    "groq": os.environ.get("INKRIFF_LLM_MODELO_GROQ", "openai/gpt-oss-20b"),
}


def _tiene_key(proveedor):
    return bool(os.environ.get("OPENAI_API_KEY" if proveedor == "openai" else "GROQ_API_KEY"))


def _llamar_llm(**kwargs):
    """Llama al proveedor primario y, si falla por cualquier motivo, reintenta automaticamente
    con el proveedor de respaldo (si tiene su propia key configurada) antes de rendirse -- asi
    Inkriff usa de verdad la API de OpenAI sin dejar de tener el
    respaldo gratuito de Groq si se agota la cuota o algo falla."""
    if not _tiene_key(LLM_PROVEEDOR_PRIMARIO):
        if not _tiene_key(LLM_PROVEEDOR_RESPALDO):
            raise RuntimeError("No hay ninguna API key de LLM configurada (OPENAI_API_KEY o GROQ_API_KEY).")
        return LLM_CLIENTES[LLM_PROVEEDOR_RESPALDO].chat.completions.create(model=LLM_MODELOS[LLM_PROVEEDOR_RESPALDO], **kwargs)
    try:
        return LLM_CLIENTES[LLM_PROVEEDOR_PRIMARIO].chat.completions.create(model=LLM_MODELOS[LLM_PROVEEDOR_PRIMARIO], **kwargs)
    except Exception as error_primario:
        if _tiene_key(LLM_PROVEEDOR_RESPALDO):
            try:
                return LLM_CLIENTES[LLM_PROVEEDOR_RESPALDO].chat.completions.create(model=LLM_MODELOS[LLM_PROVEEDOR_RESPALDO], **kwargs)
            except Exception:
                pass  # si el respaldo tambien falla, es mas util reportar el error del primario
        raise error_primario


# 1. Datos y modelo de embeddings -- mismo limpiador de texto y mismo modelo que en inkriff_modelado_similitud.ipynb, para que una busqueda en vivo sea comparable con lo ya evaluado

bandas = pd.read_csv("bandas_completo.csv")
libros = pd.read_csv("libros_con_sinopsis.csv")
rec_banda_a_libro = pd.read_csv("recomendaciones_banda_a_libro.csv")
rec_libro_a_banda = pd.read_csv("recomendaciones_libro_a_banda.csv")
libros["tiene_sinopsis"] = libros["sinopsis"].fillna("").str.strip().str.len() > 0

# Correcciones de METADATOS de despliegue (autor/año) para libros donde Open Library regresó otra obra durante la extraccion (p. ej. "En llamas" quedo con autor Juan Rulfo). Solo
# afectan lo que se MUESTRA (tarjetas, links de compra, busqueda de portada) -- el modelo usa los embeddings ya calculados, que no cambian aqui.
CORRECCIONES_LIBROS = {
    "Rey de cicatrices": {"autor": "Leigh Bardugo", "anio_num": 2019},
    "Cuarto Ala": {"autor": "Rebecca Yarros", "anio_num": 2023},
    "La balada del príncipe roto": {"autor": "Stephanie Garber", "anio_num": 2022},
    "Berserk": {"autor": "Kentaro Miura", "anio_num": 1990},
    "The Locked Tomb": {"autor": "Tamsyn Muir", "anio_num": 2019},
    "El mar sin estrellas": {"autor": "Erin Morgenstern", "anio_num": 2019},
    "En llamas": {"autor": "Suzanne Collins", "anio_num": 2009},
    "Entrevista con el vampiro": {"anio_num": 1976},
    "Caraval": {"anio_num": 2017},
    "Alicia en el País de las Maravillas": {"anio_num": 1865},
}
for _titulo, _campos in CORRECCIONES_LIBROS.items():
    for _col, _valor in _campos.items():
        libros.loc[libros["titulo_buscado"] == _titulo, _col] = _valor

SPOTIFY_POR_BANDA = dict(zip(bandas["banda"], bandas["spotify_url"]))
LIBROS_INFO_POR_TITULO = libros.set_index("titulo_buscado")[
    ["titulo", "autor", "anio_num", "categoria_semilla", "sinopsis"]
].to_dict("index")


def _primer_autor(autor_crudo):
    """El campo 'autor' a veces trae varios nombres pegados con comas (residuo de metadatos de Open Library, por ejemplo editoriales o ilustradores) -- para mostrar en tarjetas y
    armar la busqueda de compra basta con el primer nombre."""
    if not isinstance(autor_crudo, str) or not autor_crudo.strip():
        return None
    return autor_crudo.split(",")[0].strip()


# Links de BUSQUEDA (no de producto exacto -- no tenemos ISBN/ID de catalogo de cada tienda) por libro, para poder comprarlo directo desde la recomendacion.
TIENDAS_LIBROS = {
    "Amazon México": "https://www.amazon.com.mx/s?k={q}",
    "Gandhi": "https://www.gandhi.com.mx/catalogsearch/result/?q={q}",
    "El Sótano": "https://www.elsotano.com/buscar?SotK={q}",
    "Sanborns": "https://www.sanborns.com.mx/resultados?query={q}",
    "Porrúa": "https://porrua.mx/catalogsearch/result/?q={q}",
}


def generar_links_compra(titulo, autor=None):
    """Un link por tienda que busca el libro (titulo + autor) directo en esa tienda. No es un link al producto exacto (no tenemos ISBN ni ID de catalogo de cada tienda), pero deja
    comprarlo en un clic sin salir del chat."""
    if not titulo:
        return {}
    consulta = f"{titulo} {autor}".strip() if autor else titulo
    q = quote_plus(consulta)
    return {tienda: patron.format(q=q) for tienda, patron in TIENDAS_LIBROS.items()}


def spotify_embed_html(spotify_url):
    """Reproductor embebido de Spotify sin necesitar API key: basta con reescribir el link público de artista al formato /embed/ y ponerlo en un iframe.
    Va en un contenedor propio con esquinas redondeada para que combine con el tema visual incluso mientras carga."""
    if not isinstance(spotify_url, str) or "open.spotify.com" not in spotify_url:
        return ""
    embed_url = spotify_url.replace("open.spotify.com/", "open.spotify.com/embed/")
    return (
        f'<div class="inkriff-player">'
        f'<iframe src="{embed_url}" width="100%" height="152" frameborder="0" '
        f'allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" '
        f'loading="lazy"></iframe>'
        f'</div>'
    )


_ref = np.load("embeddings_referencia.npz")
EMB_BANDAS = _ref["emb_bandas"]
EMB_LIBROS = _ref["emb_libros"]
BANDA_COL_MEANS = _ref["banda_col_means"]
LIBRO_ROW_MEANS = _ref["libro_row_means"]
GLOBAL_MEAN = float(_ref["global_mean"])

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS_ES = {
    "el","la","los","las","de","del","y","a","en","un","una","unos","unas","que",
    "es","por","con","para","su","sus","se","lo","como","mas","pero","o","al",
    "este","esta","estos","estas","ya","muy","entre","sin","sobre","tambien",
    "cuando","donde","porque","desde","hasta","hay","fue","son","ser","han",
}
STOPWORDS = STOPWORDS_ES | set(ENGLISH_STOP_WORDS)
RUIDO_EDITORIAL = {
    "https","http","www","com","org","net","source","openlibrary","googlebooks",
    "publisher","published","publication","edition","editions","excerpt","isbn",
    "copyright","cover","flap","bestseller","bestselling","york","times",
    "author","authors","press","description","special","note","content",
}


def normaliza(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def similitud_cadenas(a, b):
    return SequenceMatcher(None, normaliza(a), normaliza(b)).ratio()


def limpia_texto(t):
    if not isinstance(t, str):
        return ""
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\[\d+\]:?", " ", t)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^a-z ]", " ", t.lower())
    palabras = [w for w in t.split() if len(w) > 2 and w not in STOPWORDS and w not in RUIDO_EDITORIAL]
    return " ".join(palabras)


_MODELO_EMBEDDINGS = None


def get_modelo_embeddings():
    """Carga el modelo de sentence-transformers una sola vez."""
    global _MODELO_EMBEDDINGS
    if _MODELO_EMBEDDINGS is None:
        from sentence_transformers import SentenceTransformer
        _MODELO_EMBEDDINGS = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _MODELO_EMBEDDINGS


# 2. Busqueda en vivo de metadatos (misma logica que inkriff_extraccion_completa.ipynb /
# inkriff_eda_ingenieria.ipynb, reusada aqui para bandas/libros que no estan en el catalogo).
MB_HEADERS = {"User-Agent": "InkriffChat/0.1 (https://github.com/virycassales/Inkriff)"}
MB_BASE = "https://musicbrainz.org/ws/2"


def _get_con_reintento(url, params, headers=None, max_retries=3, timeout=15):
    for intento in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** intento)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError):
            time.sleep(2 ** intento)
    return None


def buscar_banda_en_vivo(nombre):
    """Busca una banda en MusicBrainz. Regresa una lista de tags (géneros) o None si no se encontró nada; estos tags son la unica señal de "mood" musical, igual que en la
    extracción original."""
    data = _get_con_reintento(f"{MB_BASE}/artist", {"query": f'artist:"{nombre}"', "fmt": "json", "limit": 1}, headers=MB_HEADERS)
    if not data or not data.get("artists"):
        return None
    artist = data["artists"][0]
    detalle = _get_con_reintento(f"{MB_BASE}/artist/{artist['id']}", {"fmt": "json", "inc": "tags"}, headers=MB_HEADERS)
    if not detalle:
        return None
    tags = sorted(detalle.get("tags", []), key=lambda t: t.get("count", 0), reverse=True)
    return [t["name"] for t in tags[:8]] or None


def buscar_libro_en_vivo(titulo):
    """Busca la sinopsis de un libro en Open Library, con Google Books de respaldo.
    Regresa el texto de la sinopsis o None si no se encontró nada útil."""
    data = _get_con_reintento("https://openlibrary.org/search.json", {"q": titulo, "limit": 1})
    work_key = None
    if data and data.get("docs"):
        work_key = data["docs"][0].get("key")
    if work_key:
        detalle = _get_con_reintento(f"https://openlibrary.org{work_key}.json", {})
        if detalle:
            desc = detalle.get("description")
            if isinstance(desc, dict):
                desc = desc.get("value")
            if isinstance(desc, str) and desc.strip():
                return desc
    gb = _get_con_reintento("https://www.googleapis.com/books/v1/volumes", {"q": titulo, "maxResults": 1})
    if gb and gb.get("items"):
        desc = gb["items"][0].get("volumeInfo", {}).get("description")
        if isinstance(desc, str) and desc.strip():
            return desc
    return None

# 3. Recomendación: catálogo entrenado primero (instantáneo, ya evaluado), búsqueda en vivo como respaldo para cualquier libro/banda fuera de los 81/112.
UMBRAL_MATCH_CATALOGO = 0.82
TAMANO_PAGINA = 5  # cuantas recomendaciones por "página" (pagina 1 = top 1-5, pagina 2 = 6-10...)

# Nombres alternativos con los que la gente suele escribir libros/bandas del catálogo (sobre todo títulos en inglés o abreviaturas). Solo sirven para RECONOCER el nombre: la
# recomendación siempre sale del título/banda canónic del catálogo. Se escribieron a mano porque la columna 'título' de libros_con_sinopsis.csv trae varios títulos equivocados de
# Open Library (p. ej. "En llamas" -> "Pedro Paramo") y no es confiable como alias.
ALIAS_LIBROS = {
    "The Lord of the Rings": "El Senor de los Anillos",
    "Lord of the Rings": "El Senor de los Anillos",
    "A Song of Ice and Fire": "Canción de Hielo y Fuego",
    "Juego de Tronos": "Canción de Hielo y Fuego",
    "Game of Thrones": "Canción de Hielo y Fuego",
    "The Name of the Wind": "El Nombre del Viento",
    "The Hobbit": "El Hobbit",
    "The Silmarillion": "El Silmarillion",
    "The Wise Man's Fear": "El temor de un hombre sabio",
    "Mistborn": "Mistborn: El imperio final",
    "The Final Empire": "Mistborn: El imperio final",
    "The Well of Ascension": "El pozo de la ascensión",
    "The Hero of Ages": "El héroe de las eras",
    "The Way of Kings": "El camino de los reyes",
    "The Stormlight Archive": "El archivo de las tormentas",
    "The Wheel of Time": "La rueda del tiempo",
    "The Priory of the Orange Tree": "El priorato del naranjo",
    "Throne of Glass": "Trono de Cristal",
    "A Court of Thorns and Roses": "Una Corte de Rosas y Espinas",
    "ACOTAR": "Una Corte de Rosas y Espinas",
    "A Court of Mist and Fury": "Una corte de niebla y furia",
    "A Court of Wings and Ruin": "Una corte de alas y ruina",
    "Queen of Shadows": "Reina de sombras",
    "Alas de sangre": "Fourth Wing",
    "Alas de hierro": "Iron Flame",
    "Alas de onix": "Onyx Storm",
    "Once Upon a Broken Heart": "Érase una vez un corazón roto",
    "Shadow and Bone": "Sombra y Hueso",
    "The Ocean at the End of the Lane": "El océano al final del camino",
    "Elric of Melnibone": "Elric de Melniboné",
    "Jonathan Strange and Mr Norrell": "Jonathan Strange y el señor Norrell",
    "City of Bones": "Ciudad de hueso",
    "The Mortal Instruments": "Cazadores de sombras",
    "Six of Crows": "Seis de cuervos",
    "Norse Mythology": "Mitología Nórdica",
    "The Song of Achilles": "La canción de Aquiles",
    "The Lightning Thief": "El ladrón del rayo",
    "The Heroes of Olympus": "Héroes del Olimpo",
    "The Cruel Prince": "El príncipe cruel",
    "The Wicked King": "El rey malvado",
    "The Queen of Nothing": "La reina de nada",
    "The Chronicles of Narnia": "Las crónicas de Narnia",
    "Narnia": "Las crónicas de Narnia",
    "His Dark Materials": "La brújula dorada",
    "The Golden Compass": "La brújula dorada",
    "Northern Lights": "La brújula dorada",
    "Howl's Moving Castle": "El castillo ambulante",
    "Alice in Wonderland": "Alicia en el País de las Maravillas",
    "Through the Looking-Glass": "A través del espejo",
    "The Night Circus": "El circo de la noche",
    "The Starless Sea": "El mar sin estrellas",
    "The First Law": "La primera ley",
    "Malazan": "Malaz: El libro de los caídos",
    "The Black Company": "La compañía negra",
    "Interview with the Vampire": "Entrevista con el vampiro",
    "The Vampire Lestat": "El vampiro Lestat",
    "The Mayfair Witches": "Las brujas de Mayfair",
    "The Picture of Dorian Gray": "El retrato de Dorian Gray",
    "The Castle of Otranto": "El castillo de Otranto",
    "The Hunger Games": "Los juegos del hambre",
    "Catching Fire": "En llamas",
    "Mockingjay": "Sinsajo",
    "Divergent": "Divergente",
    "The Perks of Being a Wallflower": "Las ventajas de ser invisible",
    "To All the Boys I've Loved Before": "A todos los chicos de los que me enamoré",
    "The House in the Cerulean Sea": "La casa en el mar más azul",
}
ALIAS_BANDAS = {
    "Florence and the Machine": "Florence + The Machine",
    "SOAD": "System of a Down",
    "BMTH": "Bring Me The Horizon",
    "A7X": "Avenged Sevenfold",
    "MCR": "My Chemical Romance",
    "Blink 182": "Blink-182",
    "Type O": "Type O Negative",
}

LISTA_BANDAS = bandas["banda"].tolist()
LISTA_LIBROS = libros["titulo_buscado"].tolist()


def _match_en_catalogo(nombre, columna_valores, alias=None):
    """Mejor coincidencia difusa de 'nombre' contra el catálogo (y sus alias, si se dan).
    Regresa el nombre CANONICO del catálogo, o None si nada pasa el umbral."""
    mejor, mejor_score = None, 0.0
    candidatos = [(c, c) for c in columna_valores] + list((alias or {}).items())
    for escrito, canonico in candidatos:
        s = similitud_cadenas(nombre, escrito)
        if s > mejor_score:
            mejor, mejor_score = canonico, s
    return mejor if mejor_score >= UMBRAL_MATCH_CATALOGO else None


def banda_del_catalogo(nombre):
    return _match_en_catalogo(nombre, LISTA_BANDAS, ALIAS_BANDAS)


def libro_del_catalogo(nombre):
    return _match_en_catalogo(nombre, LISTA_LIBROS, ALIAS_LIBROS)


# Matriz completa libros x bandas con la misma corrección de hubness (doble centrado) que generó los CSV de recomendaciones. Reproduce EXACTAMENTE esos top 5, pero además deja
# pedir el top 6-10, 11-15, etc. ("¿qué OTROS libros van con Iron Maiden?"), cosa que los CSV (solo top 5) no permitían. Asume que los .npz y los CSV estan en el mismo orden (asi
# los genera inkriff_modelado_similitud.ipynb); la verificación de abajo lo confirma.
SIM_CORREGIDA = (EMB_LIBROS @ EMB_BANDAS.T) - LIBRO_ROW_MEANS[:, None] - BANDA_COL_MEANS[None, :] + GLOBAL_MEAN
IDX_BANDA = {b: i for i, b in enumerate(LISTA_BANDAS)}
IDX_LIBRO = {t: i for i, t in enumerate(LISTA_LIBROS)}
CANDIDATOS_LIBROS = np.where(libros["tiene_sinopsis"].values)[0]  # igual que en el modelado


def _verificar_matriz_vs_csv():
    """Chequeo de arranque: el top 5 de la matriz debe ser idéntico al de los CSV evaluados."""
    diferencias = 0
    for banda, i in IDX_BANDA.items():
        top_matriz = [LISTA_LIBROS[j] for j in CANDIDATOS_LIBROS[np.argsort(-SIM_CORREGIDA[CANDIDATOS_LIBROS, i])][:5]]
        top_csv = rec_banda_a_libro[rec_banda_a_libro["banda"] == banda].sort_values("rank")["libro_recomendado"].tolist()
        diferencias += top_matriz != top_csv
    for titulo, j in IDX_LIBRO.items():
        top_matriz = [LISTA_BANDAS[i] for i in np.argsort(-SIM_CORREGIDA[j])[:5]]
        top_csv = rec_libro_a_banda[rec_libro_a_banda["libro"] == titulo].sort_values("rank")["banda_recomendada"].tolist()
        diferencias += top_matriz != top_csv
    if diferencias:
        print(f"AVISO: {diferencias} listas no coinciden con los CSV -- revisa que embeddings_referencia.npz sea de la misma corrida.")
    else:
        print("Matriz de similitud verificada: reproduce exactamente los top 5 de los CSV.")


_verificar_matriz_vs_csv()


def _paginar(orden, pagina):
    try:
        pagina = max(1, int(pagina or 1))
    except (TypeError, ValueError):
        pagina = 1
    ini = (pagina - 1) * TAMANO_PAGINA
    return pagina, ini, orden[ini: ini + TAMANO_PAGINA]


def _info_paginacion(pagina, ini, seleccion, total):
    return {
        "pagina": pagina,
        "posiciones": f"{ini + 1}-{ini + len(seleccion)}",
        "hay_mas": ini + len(seleccion) < total,
    }


@lru_cache(maxsize=256)
def _tags_banda_en_vivo(nombre):
    return tuple(buscar_banda_en_vivo(nombre) or ())


@lru_cache(maxsize=256)
def _sinopsis_libro_en_vivo(titulo):
    return buscar_libro_en_vivo(titulo)


def recomendar_libros_para_banda(nombre_banda, pagina=1):
    match = banda_del_catalogo(nombre_banda)
    extra = {}
    if match:
        scores = SIM_CORREGIDA[:, IDX_BANDA[match]]
        fuente, nombre_resuelto = "catalogo", match
    else:
        tags = list(_tags_banda_en_vivo(nombre_banda))
        if not tags:
            return {"encontrado": False, "mensaje": f"No encontre datos de la banda '{nombre_banda}' en MusicBrainz."}
        perfil = limpia_texto(((tags[0] + " ") * 3) + " ".join(tags))
        vec = get_modelo_embeddings().encode([perfil], normalize_embeddings=True)[0]
        raw = EMB_LIBROS @ vec  # similitud contra los 112 libros
        scores = raw - LIBRO_ROW_MEANS - raw.mean() + GLOBAL_MEAN
        fuente, nombre_resuelto = "en_vivo", nombre_banda
        extra["generos_encontrados"] = tags[:5]

    orden = CANDIDATOS_LIBROS[np.argsort(-scores[CANDIDATOS_LIBROS])]
    pagina, ini, seleccion = _paginar(orden, pagina)
    if len(seleccion) == 0:
        return {"encontrado": False, "mensaje": f"Ya se mostraron todos los libros que tengo para {nombre_resuelto} ({len(orden)})."}
    recomendaciones = [
        {
            "rank": k + 1,  # siempre 1-5 en pantalla; la pagina queda en "pagina"
            "nombre": LISTA_LIBROS[i],
            "categoria": libros.iloc[i]["categoria_semilla"],
            "autor": _primer_autor(libros.iloc[i].get("autor")),
        }
        for k, i in enumerate(seleccion)
    ]
    return {
        "encontrado": True, "fuente": fuente, "tipo_resultado": "libros",
        "nombre_resuelto": nombre_resuelto, **extra,
        **_info_paginacion(pagina, ini, seleccion, len(orden)),
        "recomendaciones": recomendaciones,
    }


def recomendar_bandas_para_libro(titulo_libro, pagina=1):
    match = libro_del_catalogo(titulo_libro)
    if match:
        scores = SIM_CORREGIDA[IDX_LIBRO[match]]
        fuente, nombre_resuelto = "catalogo", match
    else:
        sinopsis = _sinopsis_libro_en_vivo(titulo_libro)
        if not sinopsis:
            return {"encontrado": False, "mensaje": f"No encontre sinopsis del libro '{titulo_libro}' en Open Library ni Google Books."}
        perfil = limpia_texto(sinopsis)
        if not perfil:
            return {"encontrado": False, "mensaje": f"Encontre '{titulo_libro}' pero su descripcion no tenia texto util."}
        vec = get_modelo_embeddings().encode([perfil], normalize_embeddings=True)[0]
        raw = EMB_BANDAS @ vec  # similitud contra las 81 bandas
        scores = raw - BANDA_COL_MEANS - raw.mean() + GLOBAL_MEAN
        fuente, nombre_resuelto = "en_vivo", titulo_libro

    orden = np.argsort(-scores)
    pagina, ini, seleccion = _paginar(orden, pagina)
    if len(seleccion) == 0:
        return {"encontrado": False, "mensaje": f"Ya se mostraron todas las bandas que tengo para {nombre_resuelto} ({len(orden)})."}
    recomendaciones = [
        {
            "rank": k + 1,  # siempre 1-5 en pantalla; la pagina queda en "pagina"
            "nombre": LISTA_BANDAS[i],
            "subgenero": bandas.iloc[i]["subgenero_semilla"],
            "spotify_url": SPOTIFY_POR_BANDA.get(LISTA_BANDAS[i]),
        }
        for k, i in enumerate(seleccion)
    ]
    return {
        "encontrado": True, "fuente": fuente, "tipo_resultado": "bandas",
        "nombre_resuelto": nombre_resuelto,
        **_info_paginacion(pagina, ini, seleccion, len(orden)),
        "recomendaciones": recomendaciones,
    }


def canciones_de_banda(nombre_banda):
    """Canciones de una banda específica: no busca en una API nueva ni tiene datos de tracks individuales; usa el mismo enlace público de Spotify que ya guardamos por banda
    (bandas_completo.csv) y aprovecha que el reproductor embebido de artista de Spotify muestra de entrada sus canciones más populares. Por eso solo funciona para bandas del
    catálogo entrenado (las únicas de las que tenemos ese enlace); que son, además, las únicas bandas que esta app recomienda como salida."""
    match = banda_del_catalogo(nombre_banda)
    if not match:
        return {
            "encontrado": False,
            "mensaje": (
                f"No tengo un enlace de Spotify guardado para '{nombre_banda}'."
            ),
        }
    url = SPOTIFY_POR_BANDA.get(match)
    if not isinstance(url, str) or "open.spotify.com" not in url:
        return {"encontrado": False, "mensaje": f"Encontré a '{match}' en mi catálogo pero no tengo su enlace de Spotify guardado."}
    return {"encontrado": True, "tipo_resultado": "canciones", "nombre_resuelto": match, "spotify_url": url}


def _recortar(texto, max_chars=700):
    texto = re.sub(r"\s+", " ", texto or "").strip()
    return texto if len(texto) <= max_chars else texto[:max_chars].rsplit(" ", 1)[0] + "..."


def info_de_libro(titulo_libro):
    """Ficha de un libro (autor, año, categoríaa, sinopsis) para preguntas de seguimiento como
    '¿de que trata el segundo?' o '¿quien escribio Babel?. Antes el chat no tenia de donde
    sacar esto y terminaba inventando o diciendo que no sabia."""
    match = libro_del_catalogo(titulo_libro)
    if not match:
        sinopsis = _sinopsis_libro_en_vivo(titulo_libro)
        if not sinopsis:
            return {"encontrado": False, "mensaje": f"No encontre informacion del libro '{titulo_libro}'."}
        return {"encontrado": True, "fuente": "en_vivo", "tipo_resultado": "info_libro",
                "nombre_resuelto": titulo_libro, "sinopsis": _recortar(sinopsis)}
    info = LIBROS_INFO_POR_TITULO.get(match, {})
    anio = info.get("anio_num")
    j = IDX_LIBRO[match]
    return {
        "encontrado": True, "fuente": "catalogo", "tipo_resultado": "info_libro",
        "nombre_resuelto": match,
        "autor": _primer_autor(info.get("autor")),
        "anio": int(anio) if pd.notna(anio) else None,
        "categoria": info.get("categoria_semilla"),
        "sinopsis": _recortar(info.get("sinopsis")) or "(sin sinopsis en el catalogo)",
        "banda_que_mejor_combina": LISTA_BANDAS[int(np.argmax(SIM_CORREGIDA[j]))],
    }


def _tags_como_lista(valor):
    if isinstance(valor, str):
        try:
            return list(ast.literal_eval(valor))
        except (ValueError, SyntaxError):
            return [valor]
    return []


def info_de_banda(nombre_banda):
    """Ficha de una banda (subgenero, pais, estilo, canción más conocida)."""
    match = banda_del_catalogo(nombre_banda)
    if not match:
        tags = list(_tags_banda_en_vivo(nombre_banda))
        if not tags:
            return {"encontrado": False, "mensaje": f"No encontré información de la banda '{nombre_banda}'."}
        return {"encontrado": True, "fuente": "en_vivo", "tipo_resultado": "info_banda",
                "nombre_resuelto": nombre_banda, "estilos": tags[:5]}
    fila = bandas.iloc[IDX_BANDA[match]]
    # 'titulo_top' viene de Genius y a veces es de OTRO artista (p. ej. Death -> Arctic Monkeys); solo se usa si el artista de Genius coincide con la banda.
    cancion = fila.get("titulo_top") if similitud_cadenas(fila.get("artista_genius"), match) >= 0.8 else None
    i = IDX_BANDA[match]
    return {
        "encontrado": True, "fuente": "catalogo", "tipo_resultado": "info_banda",
        "nombre_resuelto": match,
        "subgenero": fila.get("subgenero_semilla"),
        "pais": fila.get("pais"),
        "estilos": _tags_como_lista(fila.get("tags_top")),
        "cancion_mas_conocida": cancion,
        "libro_que_mejor_combina": LISTA_LIBROS[int(CANDIDATOS_LIBROS[np.argmax(SIM_CORREGIDA[CANDIDATOS_LIBROS, i])])],
    }


def ejecutar_recomendacion(tipo, nombre, pagina=1):
    if not nombre or not nombre.strip():
        return {"encontrado": False, "mensaje": "No se especifico un nombre."}
    if tipo == "banda":
        return recomendar_libros_para_banda(nombre.strip(), pagina)
    return recomendar_bandas_para_libro(nombre.strip(), pagina)


# 3.5 Deteccion de bandas/libros en el mensaje: hecha por CODIGO, no por el LLM. Antes el LLM tenia que adivinar solo si "Iron Maiden" era una banda o un libro; un modelo chico a
# veces se equivocaba (llamaba a bandas_para_libro("Iron Maiden"), no encontraba "ese libro" y contestaba "no encontre libros relacionados con Iron Maiden"). Ahora el código
# detecta los nombres del catálogo y se los pasa al LLM como una pista explícita.

# Nombres de banda/libro que también son palabras comunes -- solo cuentan si aparecen con la misma capitalización del catálogo ("Ghost", "HIM") y no seguidos de "metal"/"rock"
# (para que "death metal" no se detecte como la banda Death).
NOMBRES_AMBIGUOS = {
    "Death", "Sleep", "Ghost", "Tool", "HIM", "Muse", "Emperor", "Architects", "Mayhem",
    "Vicious", "Reckless", "Powerless", "Carry On", "Hell Bent", "En llamas", "Stardust",
}


def _nombres_para_detectar():
    entradas = []
    for b in LISTA_BANDAS:
        entradas.append((b, b, "banda"))
    for alias, b in ALIAS_BANDAS.items():
        entradas.append((alias, b, "banda"))
    for t in LISTA_LIBROS:
        entradas.append((t, t, "libro"))
    for alias, t in ALIAS_LIBROS.items():
        entradas.append((alias, t, "libro"))
    return sorted(entradas, key=lambda e: -len(e[0]))


ENTRADAS_DETECCION = _nombres_para_detectar()


def detectar_entidades(mensaje):
    """Regresa [(nombre_canonico, 'banda'|'libro'), ...] mencionados en el mensaje."""
    if not isinstance(mensaje, str) or not mensaje.strip():
        return []
    texto_norm = f" {normaliza(mensaje)} "
    encontrados = []
    for escrito, canonico, tipo in ENTRADAS_DETECCION:
        if escrito in NOMBRES_AMBIGUOS:
            patron = r"(?<!\w)" + re.escape(escrito) + r"(?!\w)(?!\s+(metal|metalcore|rock|core)\b)"
            if not re.search(patron, mensaje):
                continue
        else:
            n = normaliza(escrito)
            if not n or f" {n} " not in texto_norm:
                continue
        encontrados.append((escrito, canonico, tipo))
    # Si un nombre esta contenido en otro más largo que también se detectó ("Sleep" dentro de "Sleep Token"), se queda solo el largo.
    finales = []
    for escrito, canonico, tipo in encontrados:
        n = normaliza(escrito)
        if any(n != normaliza(e2) and f" {n} " in f" {normaliza(e2)} " for e2, _, _ in encontrados):
            continue
        if (canonico, tipo) not in finales:
            finales.append((canonico, tipo))
    return finales


def _nota_entidades(entidades):
    if not entidades:
        return None
    partes = [f"'{n}' es un{'a BANDA' if t == 'banda' else ' LIBRO'} del catalogo" for n, t in entidades]
    return (
        "[Deteccion automática en el último mensaje del usuario] " + "; ".join(partes) + ". "
        "Usa este dato para elegir la herramienta: una BANDA va en libros_para_banda / "
        "canciones_de_banda / info_banda; un LIBRO va en bandas_para_libro / info_libro."
    )

# 4. Chat con function calling
# Se usan herramientas separadas por tipo de resultado en vez de una sola con un parámetro "tipo"
# así el propio nombre de la función ya dice que tipo de resultado regresa, y el LLM no puede confundir "el usuario menciono una banda" con "el resultado son bandas".
_PARAM_PAGINA = {
    "type": "integer",
    "description": (
        "1 = top 1-5 (por defecto). Usa 2, 3... cuando el usuario pida 'otros', 'mas' o "
        "'diferentes' para la MISMA banda/libro: revisa en el historial que pagina se mostro "
        "la ultima vez y pide la siguiente."
    ),
}
HERRAMIENTAS = [
    {
        "type": "function",
        "function": {
            "name": "libros_para_banda",
            "description": (
                "Recibe el nombre de una BANDA de rock/metal y regresa LIBROS de fantasia/"
                "romance que combinan con ella (el resultado son LIBROS, nunca bandas). "
                "Tambien sirve para 'que otros libros van con <banda>' usando pagina=2, 3..."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nombre_banda": {"type": "string", "description": "Nombre de la banda."},
                    "pagina": _PARAM_PAGINA,
                },
                "required": ["nombre_banda"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bandas_para_libro",
            "description": (
                "Recibe el titulo de un LIBRO y regresa BANDAS de rock/metal que combinan con "
                "el (el resultado son BANDAS, nunca libros). Tambien sirve para 'que otras "
                "bandas van con <libro>' usando pagina=2, 3..."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "titulo_libro": {"type": "string", "description": "Titulo del libro."},
                    "pagina": _PARAM_PAGINA,
                },
                "required": ["titulo_libro"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "canciones_de_banda",
            "description": (
                "El usuario quiere ESCUCHAR o pide CANCIONES de una banda. Regresa un "
                "reproductor de Spotify con sus canciones mas populares. Solo funciona con "
                "bandas del catalogo de Inkriff (todas las que la app recomienda lo estan)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nombre_banda": {"type": "string", "description": "Nombre de la banda."},
                },
                "required": ["nombre_banda"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "info_libro",
            "description": (
                "Ficha de un LIBRO: autor, año, categoria y sinopsis. Usala para preguntas "
                "como '¿de que trata?', '¿quien lo escribio?', '¿es muy oscuro?' sobre un "
                "libro, incluido uno que ya se recomendo ('el segundo libro')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "titulo_libro": {"type": "string", "description": "Titulo del libro."},
                },
                "required": ["titulo_libro"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "info_banda",
            "description": (
                "Ficha de una BANDA: subgenero, pais, estilos y su cancion mas conocida. "
                "Usala para preguntas como '¿que tipo de musica hace?', '¿de donde es?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nombre_banda": {"type": "string", "description": "Nombre de la banda."},
                },
                "required": ["nombre_banda"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "Eres el asistente de Inkriff, una plataforma que conecta música de rock/metal con "
    "libros de fantasía/romance y que deja escuchar ahí mismo a las bandas que recomienda.\n\n"
    "HERRAMIENTAS:\n"
    "- libros_para_banda(nombre_banda, pagina): LIBROS que combinan con una banda.\n"
    "- bandas_para_libro(titulo_libro, pagina): BANDAS que combinan con un libro.\n"
    "- canciones_de_banda(nombre_banda): reproductor de Spotify de una banda.\n"
    "- info_libro(titulo_libro): autor, año, categoría y sinopsis de un libro.\n"
    "- info_banda(nombre_banda): subgenero, país, estilos y canción más conocida.\n"
    "Nunca inventes recomendaciones ni datos de libros/bandas: sácalos de las herramientas. "
    "Nunca digas que no encontraste algo sin haber llamado la herramienta en este turno."
    "El campo 'tipo_resultado' te dice si el resultado son libros, bandas o canciones: no los confundas."
    "SEGUIR EL HILO DE LA CONVERSACION:\n"
    "- A veces recibirás una nota '[Deteccion automatica ...]' que dice si un nombre del "
    "mensaje es BANDA o LIBRO del catálogo. Confía en ella para elegir la herramienta.\n"
    "- Después de tus respuestas anteriores hay notas internas de sistema con las listas "
    "exactas (1 a 5) que vio el usuario y su página. Úsalas para resolver referencias ('el "
    "segundo', 'la del top 3', 'esa banda', 'ese libro') y saber qué página sigue. Son solo "
    "para ti: nunca las copies, cites ni menciones.\n"
    "- Si el usuario pide 'otros', 'más' o 'diferentes' libros/bandas para la misma "
    "banda/libro, llama a la misma herramienta con la página siguiente a la última mostrada. "
    "- Si menciona una banda de forma general ('me encanta Behemoth'), usa libros_para_banda "
    "y canciones_de_banda. Si menciona un libro de forma general, usa bandas_para_libro. Si "
    "es específico ('solo libros', 'ponme sus canciones'), usa solo lo que pide.\n"
    "- Inkriff cruza musica <-> libros: no recomienda bandas parecidas a una banda ni libros "
    "parecidos a un libro. Si lo piden, dilo en una frase y ofrece el cruce (o encadenalo: "
    "libro -> sus bandas -> libros de la banda #1, si el usuario quiere).\n"
    "- Si solo esta platicando o pregunta algo sobre Inkriff, responde sin herramientas.\n"
    "- Si un resultado trae 'nota_correccion', tómalo en cuenta sin hacer preguntas.\n\n"
    "FORMATO DE TU TEXTO:\n"
    "- La app ya muestra las recomendaciones como tarjetas (1 a 5) debajo de tu mensaje, "
    "con portada, sinopsis y compra para los libros, y reproductor de Spotify para cada "
    "banda. Por eso tu texto NUNCA lleva listas, numeraciones, autores, categorias ni links "
    "('Escucha aqui', etc.): solo una o dos frases cálidas presentando lo que encontraste; "
    "puedes nombrar una sola recomendación si aporta algo. No expliques como funciona la app.\n"
    "- Si la fuente fue 'en_vivo', puedes decir que buscaste esos datos al momento porque no "
    "estaban en el catalogo original. Si algo no se encontró, dilo con honestidad y sugiere "
    "revisar el nombre.\n"
    "- Responde en español, tono cálido y natural, como alguien que sabe de música y libros. "
    "Varía como arrancas cada respuesta. No menciones puntajes de similitud ni detalles "
    "técnicos salvo que te los pidan."
)


def _ejecutar_tool_call(nombre_tool, args):
    """Corre una herramienta y regresa su resultado (dict). Si el LLM se equivocó de tipo
    (le paso una BANDA a una herramienta de LIBROS o al revés) y el nombre esta en el
    catálogo, se corrige aquí mismo en vez de regresar 'no encontrado'."""
    pagina = args.get("pagina") or 1
    if nombre_tool in ("bandas_para_libro", "info_libro"):
        titulo = (args.get("titulo_libro") or args.get("nombre_banda") or "").strip()
        if not titulo:
            return {"encontrado": False, "mensaje": "No se especifico un titulo de libro."}
        if not libro_del_catalogo(titulo) and banda_del_catalogo(titulo):
            banda = banda_del_catalogo(titulo)
            r = recomendar_libros_para_banda(banda, pagina) if nombre_tool == "bandas_para_libro" else info_de_banda(banda)
            r["nota_correccion"] = f"'{banda}' es una BANDA, no un libro; se uso la herramienta de bandas."
            return r
        return recomendar_bandas_para_libro(titulo, pagina) if nombre_tool == "bandas_para_libro" else info_de_libro(titulo)
    if nombre_tool in ("libros_para_banda", "canciones_de_banda", "info_banda"):
        nombre_b = (args.get("nombre_banda") or args.get("titulo_libro") or "").strip()
        if not nombre_b:
            return {"encontrado": False, "mensaje": "No se especifico el nombre de una banda."}
        if not banda_del_catalogo(nombre_b) and libro_del_catalogo(nombre_b):
            libro = libro_del_catalogo(nombre_b)
            r = info_de_libro(libro) if nombre_tool == "info_banda" else recomendar_bandas_para_libro(libro, pagina)
            r["nota_correccion"] = f"'{libro}' es un LIBRO, no una banda; se uso la herramienta de libros."
            return r
        if nombre_tool == "libros_para_banda":
            return recomendar_libros_para_banda(nombre_b, pagina)
        if nombre_tool == "canciones_de_banda":
            return canciones_de_banda(nombre_b)
        return info_de_banda(nombre_b)
    return {"encontrado": False, "mensaje": f"Herramienta desconocida: {nombre_tool}"}


def _escapar_html(texto):
    return (
        (texto or "")
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _html_links_compra(titulo, autor):
    links = generar_links_compra(titulo, autor)
    if not links:
        return ""
    botones = "".join(
        f'<a class="inkriff-buy-link" href="{url}" target="_blank" rel="noopener">{_escapar_html(tienda)}</a>'
        for tienda, url in links.items()
    )
    return f'<div class="inkriff-buy-links">{botones}</div>'


def _ficha_libro_html(titulo):
    """Ficha horizontal (portada + datos) para respuestas de info_libro en el chat."""
    info = LIBROS_INFO_POR_TITULO.get(titulo, {})
    autor = _primer_autor(info.get("autor"))
    anio = info.get("anio_num")
    meta = " · ".join(x for x in (autor, str(int(anio)) if pd.notna(anio) else "") if x)
    categoria = _sin_nan(info.get("categoria_semilla"))
    banda = _top_banda_de_libro(titulo)
    sub_banda = _sin_nan(bandas.iloc[IDX_BANDA[banda]].get("subgenero_semilla"))
    sinopsis = _recortar(_sin_nan(info.get("sinopsis")), 420)
    return (
        f'<div class="ink-ficha" style="--acento:{acento_libro(categoria)}">'
        f'{portada_html(titulo)}'
        f'<div class="ink-ficha-info">'
        f'<div class="ink-tile-nombre ink-ficha-titulo">{_escapar_html(titulo)}</div>'
        f'<div class="ink-tile-meta">{_escapar_html(meta)}</div>'
        f'<span class="ink-pill">{_escapar_html(categoria)}</span>'
        f'{f"<p>{_escapar_html(sinopsis)}</p>" if sinopsis else ""}'
        f'<div class="ink-combina">{monograma_html(banda, sub_banda, "ink-mono ink-mono-sm")}'
        f'<div><span class="ink-combina-lbl">🎸 Combina con</span><b>{_escapar_html(banda)}</b>'
        f'<small>{_escapar_html(sub_banda)}</small></div></div>'
        f'{reproductor_desplegable(SPOTIFY_POR_BANDA.get(banda), f"Escucha a {banda}")}'
        f'{_html_links_compra(titulo, autor)}'
        f'</div></div>'
    )


def _construir_tarjetas(resultado):
    """Tarjetas visuales (HTML, construidas por código -- no por el LLM) de cada resultado,
    en mosaico: libros con portada y links de compra; bandas con monograma y su botón para
    escucharlas (la #1 ya abierta). Mantienen la posición real del ranking (si se pidió la página 2, van del 6 al 10)."""
    tipo = resultado.get("tipo_resultado")
    if not resultado.get("encontrado"):
        return ""
    nombre = resultado.get("nombre_resuelto")
    if tipo == "info_libro":
        if nombre in IDX_LIBRO:
            return f'<div class="inkriff-cards">{_ficha_libro_html(nombre)}</div>'
        links = _html_links_compra(nombre, None)
        return f'<div class="inkriff-cards"><div class="inkriff-cards-titulo">Consíguelo · {_escapar_html(nombre)}</div>{links}</div>' if links else ""
    if tipo == "info_banda":
        if nombre in IDX_BANDA:
            return f'<div class="inkriff-cards">{grid_html([tile_banda(nombre)], "ink-grid ink-grid-chat")}</div>'
        return ""
    if tipo not in ("libros", "bandas"):
        return ""
    recomendaciones = resultado.get("recomendaciones") or []
    if not recomendaciones:
        return ""
    if tipo == "libros":
        titulo_seccion = f"📚 Libros para {nombre}"
        tiles = [tile_libro(r["nombre"], r.get("rank"), compacto=True) for r in recomendaciones if r["nombre"] in IDX_LIBRO]
    else:
        titulo_seccion = f"🎸 Bandas para {nombre}"
        tiles = [
            tile_banda(r["nombre"], r.get("rank"), compacto=True, abierto=(k == 0))
            for k, r in enumerate(recomendaciones) if r["nombre"] in IDX_BANDA
        ]
    return (
        f'<div class="inkriff-cards"><div class="inkriff-cards-titulo">{_escapar_html(titulo_seccion)}</div>'
        f'{grid_html(tiles, "ink-grid ink-grid-chat")}</div>'
    )


def _acumular_candidatos_embed(resultado, candidatos_embed):
    """Solo para canciones_de_banda: un reproductor grande y abierto de esa banda. (Las
    bandas recomendadas a partir de un libro ya traen su propio botón de escucha dentro de
    cada tarjeta del mosaico.)"""
    if resultado.get("encontrado") and resultado.get("tipo_resultado") == "canciones":
        candidatos_embed.append((resultado.get("nombre_resuelto"), resultado.get("spotify_url")))


def _construir_embeds(candidatos_embed):
    vistos, partes = set(), []
    for nombre, spotify_url in candidatos_embed:
        if nombre in vistos or not isinstance(spotify_url, str) or "open.spotify.com" not in spotify_url:
            continue
        vistos.add(nombre)
        embed_url = spotify_url.replace("open.spotify.com/", "open.spotify.com/embed/")
        partes.append(
            f'<div class="inkriff-player-label">🎧 Escucha a {_escapar_html(nombre)}</div>'
            f'<div class="inkriff-player"><iframe src="{embed_url}" width="100%" height="352" frameborder="0" '
            f'allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe></div>'
        )
    return "".join(partes)


# Memoria de la conversacion. Antes, lo unico que el LLM veia de sus turnos anteriores era el texto crudo que Gradio guardaba: su frase + TODO el HTML de tarjetas, 25 links de
# tiendas y los <iframe> de Spotify (miles de caracteres de ruido), y NUNCA los resultados reales de las herramientas. Por eso perdia el hilo ("la segunda", "otros libros").
# Ahora cada respuesta lleva, escondido en un comentario HTML invisible, un resumen corto de lo que se mostró; al armar el historial se quita todo el HTML y se deja solo ese resumen legible.
MARCA_MEMORIA = re.compile(r"<!--inkriff:([A-Za-z0-9+/=]+)-->")
MAX_MENSAJES_HISTORIAL = 16  # últimos 8 intercambios -- suficiente contexto sin saturar al modelo


def _resumir_resultado(resultado):
    tipo = resultado.get("tipo_resultado")
    nombre = resultado.get("nombre_resuelto")
    if not resultado.get("encontrado"):
        return f"Busqueda sin resultados: {resultado.get('mensaje', '')}"
    if tipo in ("libros", "bandas"):
        que = "LIBROS para la banda" if tipo == "libros" else "BANDAS para el libro"
        lista = "; ".join(f"{r.get('rank')}. {r.get('nombre')}" for r in resultado.get("recomendaciones") or [])
        return f"{que} {nombre} (pagina {resultado.get('pagina', 1)}): {lista}"
    if tipo == "canciones":
        return f"Reproductor de Spotify de la banda {nombre}"
    if tipo == "info_libro":
        return f"Ficha del libro {nombre}"
    if tipo == "info_banda":
        return f"Ficha de la banda {nombre}"
    return ""


def _texto_de_contenido(contenido):
    """Gradio 6 manda 'content' como lista de bloques ({'type': 'text', 'text': ...}); las
    versiones anteriores lo mandaban como str. Esto acepta ambas y regresa solo el texto."""
    if contenido is None:
        return ""
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, dict):
        return (contenido.get("text") or "") if contenido.get("type", "text") == "text" else ""
    if isinstance(contenido, (list, tuple)):
        return "\n".join(t for t in (_texto_de_contenido(c) for c in contenido) if t)
    return ""


def _limpiar_para_llm(texto):
    """Quita el HTML de un mensaje previo del asistente. Regresa (texto, resumen): el resumen de lo que se mostró va aparte, como nota interna de sistema,
    para que el modelo no lo copie en su respuesta."""
    resumen = None
    m = MARCA_MEMORIA.search(texto)
    if m:
        try:
            resumen = base64.b64decode(m.group(1)).decode("utf-8")
        except Exception:
            resumen = None
        texto = MARCA_MEMORIA.sub("", texto)
    corte = texto.find('<div class="inkriff-')
    if resumen is not None and corte != -1:
        texto = texto[:corte]  # el resumen ya describe las tarjetas/reproductores
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = re.sub(r"[ \t]+", " ", texto).strip()
    return _pulir_texto(texto, False), resumen


def _historial_para_llm(historial):
    limpio = []
    for m in historial or []:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            continue
        texto = _texto_de_contenido(m.get("content"))
        resumen = None
        if m["role"] == "assistant":
            texto, resumen = _limpiar_para_llm(texto)
            if not texto.strip() and resumen:
                texto = "(recomendaciones mostradas)"
        if not texto.strip():
            continue
        # Gradio puede partir una respuesta en varios mensajes seguidos del mismo rol; se
        # juntan para que el proveedor no reciba dos 'assistant' consecutivos.
        if limpio and limpio[-1]["role"] == m["role"]:
            limpio[-1]["content"] += "\n\n" + texto
        else:
            limpio.append({"role": m["role"], "content": texto})
        if resumen:
            limpio.append({"role": "system", "content": f"(Nota interna -- NUNCA la repitas ni la menciones al usuario) En esa respuesta la app mostro tarjetas con: {resumen}"})
    limpio = limpio[-MAX_MENSAJES_HISTORIAL:]
    while limpio and limpio[0]["role"] == "system":
        limpio.pop(0)
    return limpio


_PATRON_FUGA = re.compile(r"\[?\(?\s*(Lo que la app mostr[oó]|Nota interna)[^\]\n]*[\]\)]?", re.IGNORECASE)
_PATRON_ITEM_LISTA = re.compile(r"^\s*(\d+[.)]|[-*•])\s+")


def _pulir_texto(texto, hubo_tarjetas):
    """Limpia el texto del LLM antes de mostrarlo: nunca deja ver las notas internas y, si
    abajo ya van tarjetas, quita cualquier lista/links que las repita (el prompt ya lo pide,
    pero un modelo chico a veces lo hace igual)."""
    texto = _PATRON_FUGA.sub("", texto or "")
    if hubo_tarjetas:
        lineas = [l for l in texto.splitlines() if not _PATRON_ITEM_LISTA.match(l)]
        texto = "\n".join(lineas)
        texto = re.sub(r"\s*[-–—]?\s*\[[^\]]*\]\([^)]*\)", "", texto)  # links markdown
        texto = re.sub(r":[ \t]*$", ".", texto.strip(), flags=re.MULTILINE)
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
    if hubo_tarjetas and not texto:
        texto = "¡Aquí van mis recomendaciones!"
    return texto


def _marca_memoria(resumenes):
    if not resumenes:
        return ""
    datos = " | ".join(resumenes)
    return f"<!--inkriff:{base64.b64encode(datos.encode('utf-8')).decode('ascii')}-->"


def _respuesta_sin_llm(entidades, error):
    """Si el LLM no responde (key mal, cuota, caida del proveedor) pero el mensaje menciona
    una banda/libro del catálogo, igual se muestran sus recomendaciones -- asi el chat nunca
    se queda en blanco solo por un problema del proveedor."""
    tarjetas, candidatos_embed, resumenes = "", [], []
    for nombre, tipo in entidades[:2]:
        resultado = recomendar_libros_para_banda(nombre) if tipo == "banda" else recomendar_bandas_para_libro(nombre)
        tarjetas += _construir_tarjetas(resultado)
        _acumular_candidatos_embed(resultado, candidatos_embed)
        resumenes.append(_resumir_resultado(resultado))
    texto = (
        "No pude conectarme con el modelo de lenguaje en este momento, pero aquí tienes lo "
        f"que tengo en el catálogo.\n\n`{type(error).__name__}: {error}`"
    )
    return texto, tarjetas + _construir_embeds(candidatos_embed) + _marca_memoria(resumenes)


MAX_RONDAS_HERRAMIENTAS = 5  # tope de idas y vueltas con herramientas en un solo turno


def generar_respuesta_completa(mensaje, historial):
    """Toda la lógica de una respuesta del chat. Regresa (texto, extra_html) por separado
    para poder mostrar el texto con efecto de máquina de escribir y pegar tarjetas y
    reproductores completos solo al final.

    Las herramientas (`tools=HERRAMIENTAS, tool_choice="auto"`) se mandan en TODAS las
    rondas, incluida la que redacta el texto final -- si se omiten, Groq puede responder 400
    ('Tool choice is none, but model called a tool')."""
    mensaje = _texto_de_contenido(mensaje)
    if not _tiene_key(LLM_PROVEEDOR_PRIMARIO) and not _tiene_key(LLM_PROVEEDOR_RESPALDO):
        return (
            "Todavía no tengo configurada ninguna API key de LLM. Define la variable de "
            "entorno OPENAI_API_KEY (la que usa esta app por defecto) o, como alternativa "
            "gratis, GROQ_API_KEY antes de correr esta celda/app."
        ), ""

    entidades = detectar_entidades(mensaje)
    mensajes = [{"role": "system", "content": SYSTEM_PROMPT}] + _historial_para_llm(historial)
    nota = _nota_entidades(entidades)
    if nota:
        mensajes.append({"role": "system", "content": nota})
    mensajes.append({"role": "user", "content": mensaje})

    candidatos_embed = []  # (nombre, spotify_url) -- para el/los reproductor(es) al final
    tarjetas_html = ""  # tarjetas visuales de cada lista de libros/bandas, en orden
    resumenes = []  # memoria compacta de lo que se mostro (ver _marca_memoria)
    texto = None
    for ronda in range(MAX_RONDAS_HERRAMIENTAS):
        try:
            resp = _llamar_llm(messages=mensajes, tools=HERRAMIENTAS, tool_choice="auto")
        except Exception as e:
            if ronda == 0:
                if entidades:
                    return _respuesta_sin_llm(entidades, e)
                return f"Hubo un problema hablando con el modelo de lenguaje:\n\n`{type(e).__name__}: {e}`", ""
            extra = tarjetas_html + _construir_embeds(candidatos_embed) + _marca_memoria(resumenes)
            return f"Encontre resultados pero hubo un problema redactando la respuesta:\n\n`{type(e).__name__}: {e}`", extra

        msg = resp.choices[0].message
        if not msg.tool_calls:
            texto = msg.content or ""
            break

        # Se reenvia como dict simple (no el objeto del SDK) para que ambos proveedores lo
        # acepten igual, sin campos extra.
        mensajes.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (TypeError, json.JSONDecodeError):
                args = {}
            resultado = _ejecutar_tool_call(tc.function.name, args if isinstance(args, dict) else {})
            tarjetas_html += _construir_tarjetas(resultado)
            _acumular_candidatos_embed(resultado, candidatos_embed)
            resumen = _resumir_resultado(resultado)
            if resumen:
                resumenes.append(resumen)
            mensajes.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(resultado, ensure_ascii=False, default=str)})

    if texto is None:
        texto = (
            "Encontré varios resultados pero me tardé demasiado armando la respuesta final -- "
            "aquí tienes lo que ya reuní; intenta preguntar de nuevo, quizás mas específico si te faltó algo."
        )

    extra = tarjetas_html + _construir_embeds(candidatos_embed)
    return _pulir_texto(texto, bool(extra)), extra + _marca_memoria(resumenes)


def responder_chat(mensaje, historial):
    """Generador para gr.ChatInterface: revela el texto del LLM palabra por palabra y agrega
    tarjetas, reproductores y la marca de memoria completos hasta el final, para no mandar
    nunca HTML a medio armar mientras se va revelando el texto."""
    texto, extra_html = generar_respuesta_completa(mensaje, historial)
    palabras = texto.split(" ") if texto else []
    acumulado = ""
    for palabra in palabras:
        acumulado = f"{acumulado} {palabra}".strip() if acumulado else palabra
        yield acumulado
        time.sleep(0.02)
    yield (acumulado + extra_html) if extra_html else (acumulado or texto)


# 4.4 Portadas de libros + piezas visuales compartidas (chat, catalogo e inicio)

# Portadas: los CSV no traen imagen, asi que se buscan UNA sola vez en Open Library (misma
# API que ya usaba el proyecto) con Google Books de respaldo, y se guardan en
# portadas_libros.csv. En las siguientes corridas ya no se busca nada. La imagen la carga el navegador de quien usa la app desde
# covers.openlibrary.org / books.google.com. Si un libro no tiene portada, se dibuja una
# portada tipográfica con el color de su categoría; nunca queda un hueco.
# =========================================================================================
RUTA_PORTADAS = "portadas_libros.csv"
OL_HEADERS = {"User-Agent": "InkriffApp/0.1 (proyecto academico, FES Acatlan)"}

# Para varios libros, el título en español del catálogo es una serie o una traducción poco
# comúnn; se busca la portada con el título original (+ autor), que Open Library tiene mucho
# mejor catalogado.
BUSQUEDA_PORTADA = {
    "El Senor de los Anillos": ("The Fellowship of the Ring", "J.R.R. Tolkien"),
    "Canción de Hielo y Fuego": ("A Game of Thrones", "George R. R. Martin"),
    "El Nombre del Viento": ("The Name of the Wind", "Patrick Rothfuss"),
    "El Hobbit": ("The Hobbit", "J.R.R. Tolkien"),
    "El Silmarillion": ("The Silmarillion", "J.R.R. Tolkien"),
    "El temor de un hombre sabio": ("The Wise Man's Fear", "Patrick Rothfuss"),
    "Rey de cicatrices": ("King of Scars", "Leigh Bardugo"),
    "Mistborn: El imperio final": ("The Final Empire", "Brandon Sanderson"),
    "El pozo de la ascensión": ("The Well of Ascension", "Brandon Sanderson"),
    "El héroe de las eras": ("The Hero of Ages", "Brandon Sanderson"),
    "El camino de los reyes": ("The Way of Kings", "Brandon Sanderson"),
    "El archivo de las tormentas": ("Words of Radiance", "Brandon Sanderson"),
    "La rueda del tiempo": ("The Eye of the World", "Robert Jordan"),
    "El priorato del naranjo": ("The Priory of the Orange Tree", "Samantha Shannon"),
    "Trono de Cristal": ("Throne of Glass", "Sarah J. Maas"),
    "Una Corte de Rosas y Espinas": ("A Court of Thorns and Roses", "Sarah J. Maas"),
    "Cuarto Ala": ("Fourth Wing", "Rebecca Yarros"),
    "Una corte de niebla y furia": ("A Court of Mist and Fury", "Sarah J. Maas"),
    "Una corte de alas y ruina": ("A Court of Wings and Ruin", "Sarah J. Maas"),
    "Reina de sombras": ("Queen of Shadows", "Sarah J. Maas"),
    "Érase una vez un corazón roto": ("Once Upon a Broken Heart", "Stephanie Garber"),
    "La balada del príncipe roto": ("The Ballad of Never After", "Stephanie Garber"),
    "La vida invisible de Addie LaRue": ("The Invisible Life of Addie LaRue", "V. E. Schwab"),
    "Sombra y Hueso": ("Shadow and Bone", "Leigh Bardugo"),
    "Drácula": ("Dracula", "Bram Stoker"),
    "Berserk": ("Berserk", "Kentaro Miura"),
    "The Witcher": ("The Last Wish", "Andrzej Sapkowski"),
    "La novena casa": ("Ninth House", "Leigh Bardugo"),
    "El océano al final del camino": ("The Ocean at the End of the Lane", "Neil Gaiman"),
    "Elric de Melniboné": ("Elric of Melniboné", "Michael Moorcock"),
    "La guerra de la amapola": ("The Poppy War", "R. F. Kuang"),
    "The Locked Tomb": ("Harrow the Ninth", "Tamsyn Muir"),
    "Jonathan Strange y el señor Norrell": ("Jonathan Strange & Mr Norrell", "Susanna Clarke"),
    "Un día de diciembre": ("One Day in December", "Josie Silver"),
    "Cazadores de Sombras: Ciudad de Hueso": ("City of Bones", "Cassandra Clare"),
    "The Infernal Devices": ("Clockwork Angel", "Cassandra Clare"),
    "Cazadores de sombras": ("City of Glass", "Cassandra Clare"),
    "Ciudad de hueso": ("City of Bones", "Cassandra Clare"),
    "Seis de cuervos": ("Six of Crows", "Leigh Bardugo"),
    "Mitología Nórdica": ("Norse Mythology", "Neil Gaiman"),
    "La canción de Aquiles": ("The Song of Achilles", "Madeline Miller"),
    "Percy Jackson": ("The Lightning Thief", "Rick Riordan"),
    "El ladrón del rayo": ("The Lightning Thief", "Rick Riordan"),
    "Héroes del Olimpo": ("The Lost Hero", "Rick Riordan"),
    "El príncipe cruel": ("The Cruel Prince", "Holly Black"),
    "El rey malvado": ("The Wicked King", "Holly Black"),
    "La reina de nada": ("The Queen of Nothing", "Holly Black"),
    "Las crónicas de Narnia": ("The Lion, the Witch and the Wardrobe", "C. S. Lewis"),
    "La brújula dorada": ("The Golden Compass", "Philip Pullman"),
    "El castillo ambulante": ("Howl's Moving Castle", "Diana Wynne Jones"),
    "Alicia en el País de las Maravillas": ("Alice's Adventures in Wonderland", "Lewis Carroll"),
    "A través del espejo": ("Through the Looking-Glass", "Lewis Carroll"),
    "El circo de la noche": ("The Night Circus", "Erin Morgenstern"),
    "El mar sin estrellas": ("The Starless Sea", "Erin Morgenstern"),
    "La primera ley": ("The Blade Itself", "Joe Abercrombie"),
    "Malaz: El libro de los caídos": ("Gardens of the Moon", "Steven Erikson"),
    "La compañía negra": ("The Black Company", "Glen Cook"),
    "Entrevista con el vampiro": ("Interview with the Vampire", "Anne Rice"),
    "El vampiro Lestat": ("The Vampire Lestat", "Anne Rice"),
    "Las brujas de Mayfair": ("The Witching Hour", "Anne Rice"),
    "El retrato de Dorian Gray": ("The Picture of Dorian Gray", "Oscar Wilde"),
    "El castillo de Otranto": ("The Castle of Otranto", "Horace Walpole"),
    "Los juegos del hambre": ("The Hunger Games", "Suzanne Collins"),
    "En llamas": ("Catching Fire", "Suzanne Collins"),
    "Sinsajo": ("Mockingjay", "Suzanne Collins"),
    "Divergente": ("Divergent", "Veronica Roth"),
    "Maze Runner": ("The Maze Runner", "James Dashner"),
    "Scott Pilgrim": ("Scott Pilgrim's Precious Little Life", "Bryan Lee O'Malley"),
    "Las ventajas de ser invisible": ("The Perks of Being a Wallflower", "Stephen Chbosky"),
    "A todos los chicos de los que me enamoré": ("To All the Boys I've Loved Before", "Jenny Han"),
    "La casa en el mar más azul": ("The House in the Cerulean Sea", "T. J. Klune"),
}
# Si alguna portada sale equivocada, pega aqui la URL correcta (por ejemplo
# "https://covers.openlibrary.org/b/id/<id>-M.jpg") y tiene prioridad sobre la bpusqueda.
PORTADAS_MANUALES = {}


def _url_portada_openlibrary(cover_id):
    return f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg?default=false"


def _buscar_portada_libro(titulo_catalogo):
    """Regresa (url, fuente) o (None, None). Prueba: Open Library con titulo+autor ->
    Open Library solo con titulo -> Google Books."""
    info = LIBROS_INFO_POR_TITULO.get(titulo_catalogo, {})
    titulo, autor = BUSQUEDA_PORTADA.get(titulo_catalogo, (titulo_catalogo, _primer_autor(info.get("autor"))))
    intentos = [{"title": titulo, "author": autor}] if autor else []
    intentos.append({"title": titulo})
    for params in intentos:
        data = _get_con_reintento(
            "https://openlibrary.org/search.json",
            {**params, "fields": "title,cover_i", "limit": 8},
            headers=OL_HEADERS,
        )
        docs = [d for d in (data or {}).get("docs", []) if d.get("cover_i")]
        # se prefiere el resultado cuyo título más se parece al buscado (evita ediciones raras)
        docs.sort(key=lambda d: -similitud_cadenas(d.get("title", ""), titulo))
        if docs and similitud_cadenas(docs[0].get("title", ""), titulo) >= 0.5:
            return _url_portada_openlibrary(docs[0]["cover_i"]), "openlibrary"
    q = f'intitle:"{titulo}"' + (f' inauthor:"{autor}"' if autor else "")
    gb = _get_con_reintento("https://www.googleapis.com/books/v1/volumes", {"q": q, "maxResults": 5})
    for item in (gb or {}).get("items", []):
        imgs = item.get("volumeInfo", {}).get("imageLinks", {})
        url = imgs.get("thumbnail") or imgs.get("smallThumbnail")
        if url:
            return url.replace("http://", "https://").replace("&edge=curl", ""), "googlebooks"
    return None, None


def cargar_portadas():
    """Lee portadas_libros.csv; si falta (o le faltan libros), busca SOLO los que faltan y
    actualiza el CSV. Cualquier falla de red deja la app funcionando con portadas
    tipográficas. Pon INKRIFF_BUSCAR_PORTADAS=0 para no buscar nunca."""
    filas = {}
    if os.path.exists(RUTA_PORTADAS):
        try:
            previo = pd.read_csv(RUTA_PORTADAS).fillna("")
            filas = {r["titulo_buscado"]: (r["portada_url"], r["fuente"]) for _, r in previo.iterrows()}
        except Exception as e:
            print(f"No pude leer {RUTA_PORTADAS} ({e}); se vuelve a generar.")
    faltan = [t for t in LISTA_LIBROS if t not in filas]
    if faltan and os.environ.get("INKRIFF_BUSCAR_PORTADAS", "1") == "1":
        print(f"Buscando portadas de {len(faltan)} libros (solo la primera vez, ~1 min)...")
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=6) as ex:
            resultados = list(ex.map(_buscar_portada_libro, faltan))
        for t, (url, fuente) in zip(faltan, resultados):
            filas[t] = (url or "", fuente or "sin_portada")
        try:
            pd.DataFrame(
                [{"titulo_buscado": t, "portada_url": u, "fuente": f} for t, (u, f) in filas.items()]
            ).to_csv(RUTA_PORTADAS, index=False)
        except Exception as e:
            print(f"No pude guardar {RUTA_PORTADAS}: {e}")
    portadas = {t: u for t, (u, _) in filas.items() if isinstance(u, str) and u.startswith("http")}
    portadas.update(PORTADAS_MANUALES)
    print(f"Portadas: {len(portadas)} de {len(LISTA_LIBROS)} libros (el resto usa portada tipografica).")
    return portadas


try:
    PORTADAS = cargar_portadas()
except Exception as _e:
    print(f"No se pudieron cargar portadas ({type(_e).__name__}: {_e}); se usan portadas tipograficas.")
    PORTADAS = dict(PORTADAS_MANUALES)


# Fotos de bandas
# Spotify oEmbed (sin API key): dado el link público de artista que ya tenemos en
# bandas_completo.csv, regresa la URL de su foto. Se consulta UNA vez por banda y se guarda
# en fotos_bandas.csv (igual que las portadas). Si no hay foto, se usa el monograma de color.
RUTA_FOTOS_BANDAS = "fotos_bandas.csv"
FOTOS_MANUALES = {}  # banda -> URL de imagen, por si alguna sale mal


def _buscar_foto_banda(nombre):
    """Regresa la URL de la foto, "" si Spotify no tiene foto, o None si hubo un error de red
    (en ese caso NO se guarda en el CSV, para volver a intentarlo en la siguiente corrida)."""
    url = SPOTIFY_POR_BANDA.get(nombre)
    if not isinstance(url, str) or "open.spotify.com" not in url:
        return ""
    try:
        data = _get_con_reintento("https://open.spotify.com/oembed", {"url": url.split("?")[0]})
    except Exception:
        return None
    if data is None:
        return None
    foto = data.get("thumbnail_url")
    return foto if isinstance(foto, str) and foto.startswith("http") else ""


def cargar_fotos_bandas():
    fotos = {}
    if os.path.exists(RUTA_FOTOS_BANDAS):
        try:
            previo = pd.read_csv(RUTA_FOTOS_BANDAS).fillna("")
            fotos = dict(zip(previo["banda"], previo["foto_url"]))
        except Exception as e:
            print(f"No pude leer {RUTA_FOTOS_BANDAS} ({e}); se vuelve a generar.")
    faltan = [b for b in LISTA_BANDAS if b not in fotos]
    if faltan and os.environ.get("INKRIFF_BUSCAR_FOTOS", "1") == "1":
        print(f"Buscando fotos de {len(faltan)} bandas en Spotify (solo la primera vez)...")
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=6) as ex:
            for b, foto in zip(faltan, ex.map(_buscar_foto_banda, faltan)):
                if foto is not None:  # los errores de red se reintentan la proxima vez
                    fotos[b] = foto
        fallaron = len(LISTA_BANDAS) - len(fotos)
        if fallaron:
            print(f"  {fallaron} bandas no respondieron; se reintentan al volver a correr esta celda.")
        try:
            pd.DataFrame({"banda": list(fotos), "foto_url": list(fotos.values())}).to_csv(RUTA_FOTOS_BANDAS, index=False)
        except Exception as e:
            print(f"No pude guardar {RUTA_FOTOS_BANDAS}: {e}")
    fotos = {b: u for b, u in fotos.items() if isinstance(u, str) and u.startswith("http")}
    fotos.update(FOTOS_MANUALES)
    print(f"Fotos de bandas: {len(fotos)} de {len(LISTA_BANDAS)} (el resto usa monograma).")
    return fotos


try:
    FOTOS_BANDAS = cargar_fotos_bandas()
except Exception as _e:
    print(f"No se pudieron cargar fotos de bandas ({type(_e).__name__}: {_e}); se usan monogramas.")
    FOTOS_BANDAS = dict(FOTOS_MANUALES)


# Colores por familia (subgénero de banda / categoría de libro)
FAMILIAS_BANDA = [
    (("death", "black", "deathcore"), "#e11d48"),
    (("power", "symphonic", "folk"), "#d4af37"),
    (("gothic", "occult"), "#a855f7"),
    (("doom", "post-metal"), "#6366f1"),
    (("metalcore", "nu metal", "alternative metal", "progressive", "post-hardcore"), "#2dd4bf"),
    (("pop punk", "emo", "punk", "alternative rock"), "#f472b6"),
    (("hard rock", "heavy"), "#f59e0b"),
]
FAMILIAS_LIBRO = [
    (("romantasy", "romance"), "#f472b6"),
    (("gotic", "oscura"), "#a855f7"),
    (("grimdark",), "#e11d48"),
    (("alta fantasia",), "#d4af37"),
    (("distopia",), "#2dd4bf"),
    (("mitologia",), "#f59e0b"),
]


def _acento(texto, familias, defecto="#818cf8"):
    t = normaliza(texto)
    for claves, color in familias:
        if any(normaliza(c) in t for c in claves):
            return color
    return defecto


def acento_banda(subgenero):
    return _acento(subgenero, FAMILIAS_BANDA)


def acento_libro(categoria):
    return _acento(categoria, FAMILIAS_LIBRO)


def _bandera(pais):
    if isinstance(pais, str) and len(pais) == 2 and pais.isalpha():
        return "".join(chr(0x1F1E6 + ord(c) - 65) for c in pais.upper())
    return ""


def _iniciales(nombre):
    palabras = [p for p in re.findall(r"[A-Za-z0-9]+", nombre or "") if p.lower() not in ("the", "of", "a")]
    if not palabras:
        return "♪"
    return (palabras[0][0] + (palabras[1][0] if len(palabras) > 1 else "")).upper()


def _sin_nan(valor):
    return "" if valor is None or (isinstance(valor, float) and np.isnan(valor)) else str(valor)


def portada_html(titulo, clase="ink-cover"):
    """Portada real (si la hay) encima de una portada tipografica del color de la categoria:
    si la imagen no carga, se ve la tipografica en su lugar."""
    info = LIBROS_INFO_POR_TITULO.get(titulo, {})
    autor = _primer_autor(info.get("autor")) or ""
    url = PORTADAS.get(titulo)
    img = f'<img src="{_escapar_html(url)}" alt="" loading="lazy" referrerpolicy="no-referrer">' if url else ""
    return (
        f'<div class="{clase}" style="--acento:{acento_libro(info.get("categoria_semilla"))}">'
        f'<div class="ink-cover-fallback"><span class="ink-cover-orn">✦</span>'
        f'<span class="ink-cover-titulo">{_escapar_html(titulo)}</span>'
        f'<span class="ink-cover-autor">{_escapar_html(autor)}</span></div>{img}</div>'
    )


def monograma_html(nombre, subgenero, clase="ink-mono"):
    """Foto de la banda (si la hay) encima del monograma de color, que queda de respaldo."""
    foto = FOTOS_BANDAS.get(nombre)
    img = f'<img src="{_escapar_html(foto)}" alt="" loading="lazy" referrerpolicy="no-referrer">' if foto else ""
    return f'<div class="{clase}" style="--acento:{acento_banda(subgenero)}"><span>{_escapar_html(_iniciales(nombre))}</span>{img}</div>'


def reproductor_desplegable(spotify_url, etiqueta="Escucha la banda", alto=352, abierto=False):
    """Boton '🎧 Escucha la banda' que al abrirse despliega el reproductor de Spotify con su
    top de canciones. El iframe vive dentro de un <details> cerrado con loading="lazy", asi
    que el navegador no lo carga hasta que alguien lo abre (se pueden tener decenas en el
    catalogo sin volver lenta la pagina)."""
    if not isinstance(spotify_url, str) or "open.spotify.com" not in spotify_url:
        return ""
    embed_url = spotify_url.replace("open.spotify.com/", "open.spotify.com/embed/")
    return (
        f'<details class="ink-listen"{" open" if abierto else ""}><summary>🎧 {_escapar_html(etiqueta)}</summary>'
        f'<div class="inkriff-player"><iframe src="{embed_url}" width="100%" height="{alto}" frameborder="0" '
        f'allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe></div>'
        f'</details>'
    )


def _mas_desplegable(etiqueta, contenido_html):
    return f'<details class="ink-more"><summary>{etiqueta}</summary>{contenido_html}</details>' if contenido_html else ""


def _top_libro_de_banda(banda):
    i = IDX_BANDA[banda]
    return LISTA_LIBROS[int(CANDIDATOS_LIBROS[np.argmax(SIM_CORREGIDA[CANDIDATOS_LIBROS, i])])]


def _top_banda_de_libro(titulo):
    return LISTA_BANDAS[int(np.argmax(SIM_CORREGIDA[IDX_LIBRO[titulo]]))]


def tile_banda(nombre, rank=None, compacto=False, abierto=False, alto=352):
    """Mosaico de una banda: monograma de color por subgenero, pais, estilos, el libro con el
    que mejor combina (con su portada) y el boton para escucharla."""
    fila = bandas.iloc[IDX_BANDA[nombre]]
    sub = _sin_nan(fila.get("subgenero_semilla"))
    pais = _sin_nan(fila.get("pais"))
    meta = " · ".join(x for x in (pais, sub) if x)
    rank_html = f'<span class="ink-rank">#{rank}</span>' if rank else ""
    partes = [
        f'<div class="ink-tile ink-band-tile" style="--acento:{acento_banda(sub)}">',
        f'<div class="ink-band-head">{monograma_html(nombre, sub)}<div class="ink-band-id">'
        f'<div class="ink-tile-nombre">{_escapar_html(nombre)}</div>'
        f'<div class="ink-tile-meta">{_escapar_html(meta)}</div></div>{rank_html}</div>',
    ]
    if not compacto:
        estilos = [t for t in _tags_como_lista(fila.get("tags_top")) if normaliza(t) != normaliza(sub)][:3]
        if estilos:
            partes.append('<div class="ink-tags">' + "".join(f'<span>{_escapar_html(t)}</span>' for t in estilos) + "</div>")
        libro = _top_libro_de_banda(nombre)
        partes.append(
            f'<div class="ink-combina">{portada_html(libro, "ink-cover ink-cover-xs")}'
            f'<div><span class="ink-combina-lbl">📚 Combina con</span><b>{_escapar_html(libro)}</b></div></div>'
        )
    partes.append(reproductor_desplegable(SPOTIFY_POR_BANDA.get(nombre), "Escucha la banda", alto, abierto))
    partes.append("</div>")
    return "".join(partes)


def tile_libro(titulo, rank=None, compacto=False):
    """Mosaico de un libro: portada, autor/año, categoria, la banda con la que mejor combina
    (con boton para escucharla), sinopsis y donde comprarlo."""
    info = LIBROS_INFO_POR_TITULO.get(titulo, {})
    autor = _primer_autor(info.get("autor"))
    anio = info.get("anio_num")
    meta = " · ".join(x for x in (autor, str(int(anio)) if pd.notna(anio) else "") if x)
    categoria = _sin_nan(info.get("categoria_semilla"))
    rank_html = f'<span class="ink-rank ink-rank-cover">#{rank}</span>' if rank else ""
    if compacto:
        # Chat: toda la tarjeta es clickeable y al abrirse muestra sinopsis + compra.
        sinopsis = _sin_nan(info.get("sinopsis")).strip()
        sinopsis_html = f"<p>{_escapar_html(_recortar(sinopsis, 900))}</p>" if sinopsis else "<p>(Sin sinopsis en el catálogo.)</p>"
        return (
            f'<div class="ink-tile ink-book-tile" style="--acento:{acento_libro(categoria)}">'
            f'<details class="ink-book-click"><summary>'
            f'<div class="ink-cover-wrap">{portada_html(titulo)}{rank_html}</div>'
            f'<div class="ink-book-txt"><div class="ink-tile-nombre">{_escapar_html(titulo)}</div>'
            f'<div class="ink-tile-meta">{_escapar_html(meta)}</div>'
            f'<span class="ink-pill">{_escapar_html(categoria)}</span>'
            f'<span class="ink-hint">📖 Ver sinopsis</span></div>'
            f'</summary><div class="ink-book-extra">{sinopsis_html}{_html_links_compra(titulo, autor)}</div>'
            f'</details></div>'
        )
    partes = [
        f'<div class="ink-tile ink-book-tile" style="--acento:{acento_libro(categoria)}">',
        f'<div class="ink-cover-wrap">{portada_html(titulo)}{rank_html}</div>',
        f'<div class="ink-tile-nombre">{_escapar_html(titulo)}</div>',
        f'<div class="ink-tile-meta">{_escapar_html(meta)}</div>',
        f'<span class="ink-pill">{_escapar_html(categoria)}</span>',
    ]
    if not compacto:
        banda = _top_banda_de_libro(titulo)
        sub_banda = _sin_nan(bandas.iloc[IDX_BANDA[banda]].get("subgenero_semilla"))
        partes.append(
            f'<div class="ink-combina">{monograma_html(banda, sub_banda, "ink-mono ink-mono-sm")}'
            f'<div><span class="ink-combina-lbl">🎸 Combina con</span><b>{_escapar_html(banda)}</b>'
            f'<small>{_escapar_html(sub_banda)}</small></div></div>'
        )
        partes.append(reproductor_desplegable(SPOTIFY_POR_BANDA.get(banda), f"Escucha a {banda}"))
        sinopsis = _sin_nan(info.get("sinopsis")).strip()
        sinopsis_html = f"<p>{_escapar_html(_recortar(sinopsis, 900))}</p>" if sinopsis else ""
        partes.append(_mas_desplegable("📖 Sinopsis y dónde comprarlo", sinopsis_html + _html_links_compra(titulo, autor)))
    partes.append("</div>")
    return "".join(partes)


def grid_html(tiles, clase="ink-grid"):
    return f'<div class="{clase}">{"".join(tiles)}</div>'


# 4.5 Catalogo navegable en mosaico, con filtros. Para BANDAS si tenemos una señal real de popularidad externa (pageviews de Genius), así que su orden es
# "popularidad real". Para LIBROS no capturamos popularidad externa, se ordenan por cuántas bandas del catálogo los recomiendan (frecuencia de recomendación dentro de
# Inkriff). Sin filtros se muestra el top 50; con filtro/búsqueda se busca en todo el catálogo.
TODOS = "Todos"
FRECUENCIA_RECOMENDACION_LIBRO = rec_banda_a_libro["libro_recomendado"].value_counts()
SUBGENEROS = sorted(bandas["subgenero_semilla"].dropna().unique().tolist())
CATEGORIAS = sorted(libros["categoria_semilla"].dropna().unique().tolist())
# 'pageviews' de Genius solo es confiable si el artista que encontro Genius ES la banda: en
# 10 de las 81 no lo es (p. ej. Death -> una cancion de Arctic Monkeys, Ghost -> Kanye West),
# y esas inflaban el ranking (Death salia #1). Esas bandas se quedan sin lugar de
# popularidad (van al final y aparecen al filtrar/buscar) en vez de inventarles uno.
GENIUS_CONFIABLE = pd.Series(
    [similitud_cadenas(a, g) >= 0.8 for a, g in zip(bandas["banda"], bandas["artista_genius"])],
    index=bandas.index,
)
POPULARIDAD_BANDA = bandas["pageviews"].where(GENIUS_CONFIABLE)
ORDEN_BANDAS = bandas.assign(_pop=POPULARIDAD_BANDA).sort_values("_pop", ascending=False, na_position="last")["banda"].tolist()
RANK_BANDA = {b: i + 1 for i, b in enumerate(ORDEN_BANDAS) if GENIUS_CONFIABLE.iloc[IDX_BANDA[b]]}
ORDEN_LIBROS = FRECUENCIA_RECOMENDACION_LIBRO.index.tolist() + sorted(
    t for t in LISTA_LIBROS if t not in FRECUENCIA_RECOMENDACION_LIBRO.index
)
RANK_LIBRO = {t: i + 1 for i, t in enumerate(FRECUENCIA_RECOMENDACION_LIBRO.index)}
TOP_CATALOGO = 50


def _caption(texto):
    return f'<div class="inkriff-catalog-caption">{texto}</div>'


def _vacio(texto):
    return f'<div class="ink-vacio">🔎 {texto}</div>'


def render_catalogo_bandas(subgenero=TODOS, busqueda=""):
    nombres = ORDEN_BANDAS
    if subgenero and subgenero != TODOS:
        nombres = [b for b in nombres if bandas.iloc[IDX_BANDA[b]]["subgenero_semilla"] == subgenero]
    q = normaliza(busqueda)
    if q:
        nombres = [
            b for b in nombres
            if q in normaliza(b) or q in normaliza(" ".join(_tags_como_lista(bandas.iloc[IDX_BANDA[b]]["tags_top"])))
        ]
    filtrado = (subgenero and subgenero != TODOS) or q
    if not nombres:
        return _vacio("No hay bandas con ese filtro. Prueba otro subgénero o búsqueda.")
    if filtrado:
        cap = f"{len(nombres)} banda{'s' if len(nombres) != 1 else ''} · # = lugar en popularidad (actividad en Genius; sin # = sin dato confiable)"
    else:
        nombres = nombres[:TOP_CATALOGO]
        cap = f"Top {TOP_CATALOGO} por popularidad real (actividad en Genius) · filtra o busca para ver las {len(ORDEN_BANDAS)}"
    return _caption(cap) + grid_html([tile_banda(b, RANK_BANDA.get(b)) for b in nombres])


def render_catalogo_libros(categoria=TODOS, busqueda=""):
    titulos = ORDEN_LIBROS
    if categoria and categoria != TODOS:
        titulos = [t for t in titulos if LIBROS_INFO_POR_TITULO.get(t, {}).get("categoria_semilla") == categoria]
    q = normaliza(busqueda)
    if q:
        titulos = [
            t for t in titulos
            if q in normaliza(t) or q in normaliza(_primer_autor(LIBROS_INFO_POR_TITULO.get(t, {}).get("autor")) or "")
        ]
    filtrado = (categoria and categoria != TODOS) or q
    if not titulos:
        return _vacio("No hay libros con ese filtro. Prueba otra categoría o búsqueda.")
    if filtrado:
        cap = f"{len(titulos)} libro{'s' if len(titulos) != 1 else ''} · # = cuántas bandas del catálogo lo recomiendan (orden)"
    else:
        titulos = titulos[:TOP_CATALOGO]
        cap = (f"Top {TOP_CATALOGO} por cuántas bandas del catálogo los recomiendan (no tenemos popularidad externa "
               f"de libros) · filtra o busca para ver los {len(ORDEN_LIBROS)}")
    return _caption(cap) + grid_html([tile_libro(t, RANK_LIBRO.get(t)) for t in titulos], "ink-grid ink-grid-libros")


def _parejas_destacadas(n=6):
    """Las n parejas libro-banda con mayor similitud (corregida), sin repetir libro ni banda."""
    pares, usados_l, usados_b = [], set(), set()
    sub = SIM_CORREGIDA[CANDIDATOS_LIBROS]
    for plano in np.argsort(-sub, axis=None):
        fila, col = np.unravel_index(plano, sub.shape)
        j = int(CANDIDATOS_LIBROS[fila])
        if j in usados_l or col in usados_b:
            continue
        pares.append((LISTA_LIBROS[j], LISTA_BANDAS[int(col)]))
        usados_l.add(j)
        usados_b.add(col)
        if len(pares) == n:
            break
    return pares


def _html_pareja(titulo, banda):
    sub = _sin_nan(bandas.iloc[IDX_BANDA[banda]].get("subgenero_semilla"))
    return (
        f'<div class="ink-pair">'
        f'<div class="ink-pair-visual">{portada_html(titulo, "ink-cover ink-cover-sm")}'
        f'<span class="ink-pair-x">✕</span>{monograma_html(banda, sub, "ink-mono ink-mono-foto")}</div>'
        f'<div class="ink-pair-txt"><b>{_escapar_html(titulo)}</b><span>con</span><b>{_escapar_html(banda)}</b></div>'
        f'<div class="ink-pair-botones">'
        f'<a class="ink-amazon" href="{_link_amazon(titulo)}" target="_blank" rel="noopener">🛒 Comprar en Amazon</a>'
        f'{reproductor_desplegable(SPOTIFY_POR_BANDA.get(banda), f"Escucha a {banda}")}'
        f'</div></div>'
    )


def _link_amazon(titulo):
    autor = _primer_autor(LIBROS_INFO_POR_TITULO.get(titulo, {}).get("autor"))
    return _escapar_html(generar_links_compra(titulo, autor)["Amazon México"])


# 4.6 Sugerencias de la comunidad -- sin cuentas ni login. Cualquiera puede mandar una
# sugerencia y queda guardada en un CSV. Si montaste Google Drive (celda opcional), el CSV
# se guarda ahí y sobrevive al reiniciar Colab; sin Drive, la app funciona igual.
_CARPETA_DRIVE = "/content/drive/MyDrive/Inkriff"
if os.path.isdir("/content/drive/MyDrive"):
    os.makedirs(_CARPETA_DRIVE, exist_ok=True)
    RUTA_SUGERENCIAS_LOCAL = os.path.join(_CARPETA_DRIVE, "sugerencias_usuarios.csv")
else:
    RUTA_SUGERENCIAS_LOCAL = "sugerencias_usuarios.csv"
MAX_LARGO_CAMPO_CORTO = 300
MAX_LARGO_NOTA = 1000


def guardar_sugerencia(tipo, nombre, combina_con, nota):
    nombre = (nombre or "").strip()
    combina_con = (combina_con or "").strip()
    nota = (nota or "").strip()
    if not nombre:
        return "Escribe el nombre de la banda o el libro que quieres sugerir."
    if len(nombre) > MAX_LARGO_CAMPO_CORTO or len(combina_con) > MAX_LARGO_CAMPO_CORTO:
        return "Ese campo es demasiado largo -- intenta resumirlo."
    if len(nota) > MAX_LARGO_NOTA:
        return "Tu nota es demasiado larga -- intenta resumirla un poco."
    fila = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tipo": tipo or "Banda",
        "nombre": nombre,
        "combina_con": combina_con,
        "nota": nota,
    }
    try:
        existe = os.path.exists(RUTA_SUGERENCIAS_LOCAL)
        with open(RUTA_SUGERENCIAS_LOCAL, "a", newline="", encoding="utf-8") as f:
            escritor = csv.DictWriter(f, fieldnames=list(fila.keys()))
            if not existe:
                escritor.writeheader()
            escritor.writerow(fila)
    except Exception as e:
        return f"No pude guardar tu sugerencia: {type(e).__name__}: {e}"
    return f'¡Gracias! Guardamos tu sugerencia de {fila["tipo"].lower()} "{nombre}". La revisaremos para agregarla al catálogo.'

# 5. Tema visual e interfaz -- paleta (violeta/oro sobre fondo casi negro), igual en modo claro y oscuro del navegador para que la marca se vea
# siempre igual.
_OSCURO = "#0d0b14"
_PANEL = "#1a1625"
_PANEL_2 = "#221c33"
_BORDE = "#3a2f52"
TEMA_INKRIFF = gr.themes.Base(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill=_OSCURO, body_background_fill_dark=_OSCURO,
    body_text_color="#f3f1f7", body_text_color_dark="#f3f1f7",
    body_text_color_subdued="#a89cc8", body_text_color_subdued_dark="#a89cc8",
    background_fill_primary=_PANEL, background_fill_primary_dark=_PANEL,
    background_fill_secondary="#151121", background_fill_secondary_dark="#151121",
    block_background_fill=_PANEL, block_background_fill_dark=_PANEL,
    block_border_color=_BORDE, block_border_color_dark=_BORDE,
    block_title_text_color="#d4af37", block_title_text_color_dark="#d4af37",
    block_label_text_color="#a89cc8", block_label_text_color_dark="#a89cc8",
    block_label_background_fill=_PANEL, block_label_background_fill_dark=_PANEL,
    input_background_fill=_PANEL_2, input_background_fill_dark=_PANEL_2,
    border_color_primary=_BORDE, border_color_primary_dark=_BORDE,
    color_accent_soft=_PANEL_2, color_accent_soft_dark=_PANEL_2,
    button_primary_background_fill="linear-gradient(90deg, #7c3aed, #a855f7)",
    button_primary_background_fill_dark="linear-gradient(90deg, #7c3aed, #a855f7)",
    button_primary_background_fill_hover="linear-gradient(90deg, #6d28d9, #9333ea)",
    button_primary_background_fill_hover_dark="linear-gradient(90deg, #6d28d9, #9333ea)",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill=_PANEL_2, button_secondary_background_fill_dark=_PANEL_2,
    button_secondary_text_color="#d4af37", button_secondary_text_color_dark="#d4af37",
    button_secondary_border_color=_BORDE, button_secondary_border_color_dark=_BORDE,
)

CSS_INKRIFF = """
.gradio-container { width: 100% !important; max-width: 1240px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* ---------- Encabezado ---------- */
.inkriff-header { text-align: center; padding: 18px 12px 6px 12px; }
.inkriff-header-emoji { font-size: 2em; line-height: 1; }
.inkriff-header-title {
    font-size: 2.3em; font-weight: 800; letter-spacing: .05em; margin-top: 2px;
    color: #d4af37;
    background: linear-gradient(90deg, #f4d160, #c084fc 55%, #9333ea);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.inkriff-header-tagline { color: #a89cc8; font-size: 1em; margin-top: 2px; }

/* ---------- Mosaico (catalogo y chat) ---------- */
.ink-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 14px; align-items: start; }
.ink-grid-libros { grid-template-columns: repeat(auto-fill, minmax(165px, 1fr)); }
.ink-grid-chat { grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px; }
.ink-tile {
    position: relative; display: flex; flex-direction: column; gap: 8px; padding: 14px;
    border-radius: 16px; background: linear-gradient(160deg, #1f1830, #15111f);
    border: 1px solid #3a2f52; border-top: 3px solid var(--acento, #a855f7);
    transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease; min-width: 0;
}
.ink-tile:hover {
    transform: translateY(-3px); border-color: var(--acento, #a855f7);
    box-shadow: 0 10px 28px -12px var(--acento, #a855f7);
}
.ink-tile-nombre { font-weight: 700; color: #f3f1f7; font-size: 1.02em; line-height: 1.25; }
.ink-tile-meta { color: #a89cc8; font-size: 0.8em; }
.ink-rank {
    margin-left: auto; align-self: flex-start; font-weight: 800; font-size: 0.78em; color: #0d0b14;
    background: var(--acento, #d4af37); padding: 2px 8px; border-radius: 999px;
}
.ink-rank-cover { position: absolute; top: 8px; left: 8px; margin: 0; box-shadow: 0 2px 8px rgba(0,0,0,.5); z-index: 2; }
.ink-pill {
    align-self: flex-start; font-size: 0.7em; font-weight: 600; padding: 3px 10px; border-radius: 999px;
    color: var(--acento); border: 1px solid var(--acento);
    background: color-mix(in srgb, var(--acento) 14%, transparent);
}
.ink-tags { display: flex; flex-wrap: wrap; gap: 5px; }
.ink-tags span { font-size: 0.7em; color: #d8d2e6; background: #2a2240; padding: 2px 8px; border-radius: 999px; }

/* Banda: monograma de color */
.ink-band-head { display: flex; align-items: center; gap: 12px; }
.ink-band-id { min-width: 0; padding-right: 34px; }
.ink-band-tile .ink-rank { position: absolute; top: 10px; right: 10px; margin: 0; }
.ink-mono {
    flex-shrink: 0; width: 48px; height: 48px; border-radius: 14px; display: grid; place-items: center;
    font-weight: 800; font-size: 1.05em; color: #fff; letter-spacing: .02em;
    background: radial-gradient(circle at 30% 25%, color-mix(in srgb, var(--acento) 85%, #fff 15%), color-mix(in srgb, var(--acento) 45%, #0d0b14));
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.12), 0 6px 16px -8px var(--acento);
}
.ink-mono { position: relative; overflow: hidden; }
.ink-mono img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
.ink-mono-foto { width: 104px; height: 104px; border-radius: 16px; font-size: 1.6em; }
.ink-mono-sm { width: 34px; height: 34px; border-radius: 10px; font-size: .8em; }
.ink-mono-lg { width: 64px; height: 64px; border-radius: 18px; font-size: 1.3em; }

/* Libro: portada real sobre portada tipografica */
.ink-cover-wrap { position: relative; }
.ink-cover {
    position: relative; aspect-ratio: 2 / 3; width: 100%; border-radius: 10px; overflow: hidden;
    background: linear-gradient(160deg, var(--acento), #1a1625 88%);
    box-shadow: 0 8px 22px -10px rgba(0,0,0,.8), inset 0 0 0 1px rgba(255,255,255,.08);
}
.ink-cover img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
.ink-cover-fallback {
    position: absolute; inset: 0; display: flex; flex-direction: column; justify-content: center; align-items: center;
    gap: 8px; padding: 14px; text-align: center; border: 1px solid rgba(255,255,255,.18); margin: 6px; border-radius: 6px;
}
.ink-cover-orn { color: rgba(255,255,255,.7); font-size: 1.1em; }
.ink-cover-titulo { font-family: Georgia, 'Times New Roman', serif; font-weight: 700; color: #fff; font-size: 1.05em; line-height: 1.2; text-shadow: 0 2px 6px rgba(0,0,0,.5); }
.ink-cover-autor { color: rgba(255,255,255,.75); font-size: .72em; text-transform: uppercase; letter-spacing: .08em; }
.ink-cover-xs { width: 34px; flex-shrink: 0; border-radius: 4px; }
.ink-cover-xs .ink-cover-fallback { margin: 0; padding: 2px; border: none; }
.ink-cover-xs .ink-cover-titulo, .ink-cover-xs .ink-cover-autor { display: none; }
.ink-cover-xs .ink-cover-orn { font-size: .8em; }
.ink-cover-sm { width: 76px; flex-shrink: 0; }
.ink-cover-sm .ink-cover-fallback { margin: 3px; padding: 4px; }
.ink-cover-sm .ink-cover-titulo { font-size: .62em; }
.ink-cover-sm .ink-cover-autor { display: none; }

/* "Combina con" */
.ink-combina {
    display: flex; align-items: center; gap: 10px; padding: 8px 10px; border-radius: 12px;
    background: rgba(255,255,255,.04); border: 1px dashed #3a2f52;
}
.ink-combina > div:last-child { display: flex; flex-direction: column; min-width: 0; }
.ink-combina-lbl { font-size: .68em; text-transform: uppercase; letter-spacing: .06em; color: #a89cc8; }
.ink-combina b { color: #2dd4bf; font-size: .9em; }
.ink-combina small { color: #a89cc8; font-size: .72em; }

/* Botones desplegables: escuchar / sinopsis / comprar */
.ink-listen > summary, .ink-more > summary {
    list-style: none; cursor: pointer; user-select: none; font-size: .8em; font-weight: 600;
    padding: 7px 12px; border-radius: 999px; text-align: center;
}
.ink-listen > summary::-webkit-details-marker, .ink-more > summary::-webkit-details-marker { display: none; }
.ink-listen > summary { color: #0d0b14; background: linear-gradient(90deg, #1ed760, #2dd4bf); }
.ink-listen > summary:hover { filter: brightness(1.1); }
.ink-listen[open] > summary { margin-bottom: 8px; }
.ink-more > summary { color: #d4af37; background: #221c33; border: 1px solid #3a2f52; }
.ink-more > summary:hover { border-color: #d4af37; }
.ink-more[open] > summary { margin-bottom: 6px; }
.ink-more .inkriff-buy-links { margin-top: 8px; }
.ink-more p { margin: 0; font-size: .8em; line-height: 1.5; color: #d8d2e6; max-height: 190px; overflow-y: auto; padding-right: 4px; }

/* Links de compra */
.inkriff-buy-links { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0; }
.inkriff-buy-link {
    font-size: 0.72em; padding: 4px 10px; border-radius: 999px; font-weight: 600;
    background: #221c33; color: #d4af37 !important; border: 1px solid #3a2f52;
    text-decoration: none !important; white-space: nowrap;
}
.inkriff-buy-link:hover { background: #2c2440; border-color: #d4af37; }

/* Reproductor */
.inkriff-player { border-radius: 12px; overflow: hidden; border: 1px solid #3a2f52; background: #120f1c; }
.inkriff-player iframe { display: block; }
.inkriff-player-label { font-weight: 700; color: #d4af37; font-size: 0.82em; margin: 12px 0 6px 0; }

/* Al abrir un reproductor (o la sinopsis en el chat) la tarjeta ocupa dos columnas, para
   que el reproductor de Spotify se vea completo con los titulos de las canciones. */
.ink-grid, .ink-pairs { grid-auto-flow: row dense; }
.ink-tile:has(.ink-listen[open]), .ink-pair:has(.ink-listen[open]), .ink-tile:has(.ink-book-click[open]) { grid-column: span 2; }
@media (max-width: 700px) {
    .ink-tile:has(.ink-listen[open]), .ink-pair:has(.ink-listen[open]), .ink-tile:has(.ink-book-click[open]) { grid-column: auto; }
}
/* Libro clickeable (chat) */
.ink-book-click > summary { list-style: none; cursor: pointer; display: flex; flex-direction: column; gap: 6px; }
.ink-book-click > summary::-webkit-details-marker { display: none; }
.ink-book-txt { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
.ink-hint { font-size: .72em; font-weight: 600; color: #d4af37; }
.ink-book-click[open] .ink-hint { display: none; }
.ink-book-click[open] > summary { display: grid; grid-template-columns: 120px 1fr; gap: 14px; align-items: start; }
.ink-book-extra { margin-top: 10px; display: flex; flex-direction: column; gap: 8px; }
.ink-book-extra p { margin: 0; font-size: .85em; line-height: 1.55; color: #d8d2e6; }

/* ---------- Chat ---------- */
.inkriff-cards {
    min-width: min(820px, 82vw); box-sizing: border-box;
    margin-top: 10px; padding: 14px; border-radius: 16px;
    background: linear-gradient(135deg, #1a1625, #241c38); border: 1px solid #3a2f52;
}
.inkriff-cards-titulo {
    font-weight: 700; color: #d4af37; letter-spacing: .05em; margin-bottom: 10px;
    font-size: 0.75em; text-transform: uppercase;
}
.ink-grid-chat .ink-tile { padding: 10px; gap: 6px; }
.ink-grid-chat .ink-tile-nombre { font-size: .9em; }
.ink-grid-chat .ink-mono { width: 38px; height: 38px; border-radius: 11px; font-size: .85em; }
.ink-grid-chat .ink-band-head { gap: 9px; }
.ink-grid-chat:has(.ink-band-tile) { grid-template-columns: repeat(auto-fill, minmax(215px, 1fr)); }
.ink-ficha { display: grid; grid-template-columns: 150px 1fr; gap: 16px; align-items: start; }
.ink-ficha-info { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.ink-ficha-titulo { font-size: 1.2em; }
.ink-ficha p { margin: 0; font-size: .85em; line-height: 1.5; color: #d8d2e6; }
@media (max-width: 700px) {
    .inkriff-cards { min-width: 0; }
    .ink-ficha { grid-template-columns: 100px 1fr; }
    .ink-grid-chat { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); }
}

/* ---------- Catálogo ---------- */
.inkriff-catalog-caption { color: #a89cc8; font-size: 0.85em; margin: 2px 0 14px 2px; }
.ink-vacio { text-align: center; color: #a89cc8; padding: 40px 10px; border: 1px dashed #3a2f52; border-radius: 16px; }

/* ---------- Inicio ---------- */
.ink-hero {
    display: grid; grid-template-columns: 1.3fr 1fr; gap: 24px; align-items: center; padding: 26px;
    border-radius: 20px; border: 1px solid #3a2f52; margin-bottom: 22px;
    background: radial-gradient(circle at 15% 20%, rgba(147,51,234,.25), transparent 55%),
                radial-gradient(circle at 90% 90%, rgba(212,175,55,.14), transparent 50%), #15111f;
}
.ink-hero h2 { margin: 0 0 10px 0; font-size: 1.7em; color: #f3f1f7; line-height: 1.2; }
.ink-hero h2 span { color: #d4af37; }
.ink-hero p { margin: 0; color: #d8d2e6; line-height: 1.6; }
.ink-hero-cta { margin-top: 14px; color: #2dd4bf; font-weight: 600; font-size: .9em; }
.inkriff-stats-row { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
.inkriff-stat { text-align: center; padding: 16px 8px; border-radius: 14px; background: rgba(255,255,255,.04); border: 1px solid #3a2f52; }
.inkriff-stat-numero { font-size: 1.8em; font-weight: 800; color: #f4d160; }
.inkriff-stat-etiqueta { font-size: 0.74em; color: #a89cc8; text-transform: uppercase; letter-spacing: .04em; margin-top: 2px; }
.ink-seccion-titulo { color: #d4af37; font-size: 1.1em; font-weight: 700; margin: 6px 0 4px 2px; }
.ink-seccion-sub { color: #a89cc8; font-size: .85em; margin: 0 0 12px 2px; }
.ink-pairs { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin-bottom: 26px; align-items: start; }
.ink-pair { display: flex; flex-direction: column; gap: 10px; padding: 14px; border-radius: 16px; background: linear-gradient(160deg, #1f1830, #15111f); border: 1px solid #3a2f52; }
.ink-pair-visual { display: flex; align-items: center; justify-content: center; gap: 14px; }
.ink-pair-x { color: #d4af37; font-size: 1.2em; }
.ink-pair-txt { text-align: center; display: flex; flex-direction: column; font-size: .88em; }
.ink-pair-txt b { color: #f3f1f7; }
.ink-pair-txt span { color: #a89cc8; font-size: .8em; }
.ink-pair-botones { display: flex; flex-direction: column; gap: 8px; }
.ink-amazon {
    text-align: center; font-size: .8em; font-weight: 700; padding: 7px 12px; border-radius: 999px;
    color: #0d0b14 !important; background: linear-gradient(90deg, #f4d160, #f59e0b); text-decoration: none !important;
}
.ink-amazon:hover { filter: brightness(1.08); }
.ink-pair-visual .ink-cover-sm { width: 70px; }
.ink-pasos { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 18px; }
.ink-paso { padding: 16px; border-radius: 16px; background: #15111f; border: 1px solid #3a2f52; }
.ink-paso-icono { font-size: 1.6em; }
.ink-paso b { display: block; color: #2dd4bf; margin: 6px 0 4px 0; }
.ink-paso p { margin: 0; color: #d8d2e6; font-size: .85em; line-height: 1.5; }
.inkriff-landing-creditos { text-align: center; color: #a89cc8; font-size: 0.82em; padding-top: 14px; border-top: 1px solid #3a2f52; }
@media (max-width: 760px) { .ink-hero { grid-template-columns: 1fr; } }
"""

HEADER_HTML = """<div class="inkriff-header">
<div class="inkriff-header-emoji">🎸📖</div>
<div class="inkriff-header-title">Inkriff</div>
<div class="inkriff-header-tagline">Donde el metal se encuentra con la fantasía</div>
</div>"""

LANDING_HTML = f"""<div class="inkriff-landing">
<div class="ink-hero">
  <div>
    <h2>Tu próxima lectura tiene <span>soundtrack</span>.</h2>
    <p>Inkriff es un recomendador bidireccional entre música de rock/metal y libros de
    fantasía/romance: dime qué banda escuchas y te digo qué leer, o cuéntame qué libro te
    atrapó y te pongo la banda que suena igual.</p>
    <div class="ink-hero-cta">→ Pruébalo en la pestaña Chat, o explora el Catálogo.</div>
  </div>
  <div class="inkriff-stats-row">
    <div class="inkriff-stat"><div class="inkriff-stat-numero">{len(LISTA_BANDAS)}</div><div class="inkriff-stat-etiqueta">Bandas</div></div>
    <div class="inkriff-stat"><div class="inkriff-stat-numero">{len(LISTA_LIBROS)}</div><div class="inkriff-stat-etiqueta">Libros</div></div>
    <div class="inkriff-stat"><div class="inkriff-stat-numero">{len(SUBGENEROS)}</div><div class="inkriff-stat-etiqueta">Subgéneros</div></div>
    <div class="inkriff-stat"><div class="inkriff-stat-numero">{len(CATEGORIAS)}</div><div class="inkriff-stat-etiqueta">Categorías de libro</div></div>
  </div>
</div>

<div class="ink-seccion-titulo">✦ Parejas destacadas</div>
<div class="ink-seccion-sub">Las combinaciones libro–banda con mayor afinidad según el modelo.</div>
<div class="ink-pairs">{"".join(_html_pareja(t, b) for t, b in _parejas_destacadas(8))}</div>

<div class="ink-seccion-titulo">¿Cómo funciona?</div>
<div class="ink-pasos">
  <div class="ink-paso"><div class="ink-paso-icono">🗂️</div><b>Datos</b><p>MusicBrainz (géneros y subgéneros de bandas), Open Library y Google Books (sinopsis y portadas de libros).</p></div>
  <div class="ink-paso"><div class="ink-paso-icono">🧠</div><b>Modelo</b><p>Embeddings multilingües de sentence-transformers, con corrección de “hubness” para que no dominen siempre las mismas bandas o libros.</p></div>
  <div class="ink-paso"><div class="ink-paso-icono">💬</div><b>Chat</b><p>Un LLM (OpenAI, con Groq como respaldo automático) entiende tu pregunta y llama a la herramienta de recomendación correcta.</p></div>
  <div class="ink-paso"><div class="ink-paso-icono">🎧</div><b>Escucha</b><p>Reproductor de Spotify embebido con el top de canciones de cada banda.</p></div>
</div>
<div class="inkriff-landing-creditos">Proyecto Académico</div>
</div>"""

with gr.Blocks(title="Inkriff") as demo:
    gr.HTML(HEADER_HTML)
    with gr.Tabs():
        with gr.Tab("Inicio"):
            gr.HTML(LANDING_HTML)
        with gr.Tab("Chat"):
            gr.ChatInterface(
                fn=responder_chat,
                examples=[
                    "¿Qué libros me recomiendas si me gusta Behemoth?",
                    "Acabo de terminar Babel, ¿qué banda le queda?",
                    "Quiero escuchar canciones de Bad Omens",
                    "¿De qué trata Una Corte de Rosas y Espinas?",
                ],
                # sanitize_html=False: sin esto Gradio borra los <iframe> de Spotify y las
                # tarjetas del mosaico. Todo ese HTML lo construye nuestro propio codigo (el
                # texto del LLM y del usuario nunca se inserta como HTML), asi que es seguro.
                chatbot=gr.Chatbot(sanitize_html=False, show_label=False, height=680),
            )
        with gr.Tab("Catálogo"):
            with gr.Tabs():
                with gr.Tab("🎸 Bandas"):
                    with gr.Row():
                        filtro_subgenero = gr.Dropdown([TODOS] + SUBGENEROS, value=TODOS, label="Subgénero", scale=1)
                        busqueda_banda = gr.Textbox(label="Buscar", placeholder="Nombre o estilo (p. ej. nu metal, pop punk, gojira)", scale=2)
                    catalogo_bandas = gr.HTML(render_catalogo_bandas())
                    for _control in (filtro_subgenero, busqueda_banda):
                        _control.change(render_catalogo_bandas, [filtro_subgenero, busqueda_banda], catalogo_bandas)
                with gr.Tab("📚 Libros"):
                    with gr.Row():
                        filtro_categoria = gr.Dropdown([TODOS] + CATEGORIAS, value=TODOS, label="Categoría", scale=1)
                        busqueda_libro = gr.Textbox(label="Buscar", placeholder="Título o autor", scale=2)
                    catalogo_libros = gr.HTML(render_catalogo_libros())
                    for _control in (filtro_categoria, busqueda_libro):
                        _control.change(render_catalogo_libros, [filtro_categoria, busqueda_libro], catalogo_libros)
        with gr.Tab("Sugerir"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### ¿Conoces una banda o un libro que debería estar en Inkriff?")
                    gr.Markdown("Mándanoslo y lo revisamos para agregarlo al catálogo -- no necesitas crear ninguna cuenta.")
                    tipo_sugerencia = gr.Radio(["Banda", "Libro"], value="Banda", label="¿Qué quieres sugerir?")
                    nombre_sugerencia = gr.Textbox(label="Nombre de la banda o del libro")
                    combina_sugerencia = gr.Textbox(label="¿Con qué banda o libro combina? (opcional)")
                    nota_sugerencia = gr.Textbox(label="Cuéntanos por qué combinan (opcional)", lines=3)
                    boton_sugerencia = gr.Button("Enviar sugerencia", variant="primary")
                    resultado_sugerencia = gr.Markdown()
                with gr.Column(scale=2):
                    gr.HTML(
                        '<div class="ink-paso" style="margin-top:8px"><div class="ink-paso-icono">✍️</div>'
                        '<b>¿Qué pasa con tu sugerencia?</b><p>La revisamos a mano: si la banda o el libro '
                        'tienen datos suficientes (géneros en MusicBrainz, sinopsis en Open Library), se '
                        'agrega al catálogo en la siguiente actualización del modelo.</p></div>'
                    )
            boton_sugerencia.click(
                fn=guardar_sugerencia,
                inputs=[tipo_sugerencia, nombre_sugerencia, combina_sugerencia, nota_sugerencia],
                outputs=resultado_sugerencia,
            )


if __name__ == "__main__":
    demo.launch(theme=TEMA_INKRIFF, css=CSS_INKRIFF)
