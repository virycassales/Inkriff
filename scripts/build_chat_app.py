# -*- coding: utf-8 -*-
"""Genera el pipeline conversacional de Inkriff: un chat (LLM con function calling) que
recibe lenguaje natural, entiende si mencionas un libro o una banda, y recomienda del
lado contrario -- funciona tanto para las 81 bandas / 112 libros ya entrenados (usa los
CSVs precalculados, respuesta instantanea) como para CUALQUIER libro o banda nuevo (busca
sus datos en vivo en MusicBrainz / Open Library / Google Books, calcula su embedding al
vuelo con el mismo modelo de sentence-transformers de inkriff_modelado_similitud.ipynb, y
aplica la misma correccion de hubness).

*** CAMBIO DE ARQUITECTURA IMPORTANTE ***
La version anterior del pipeline solo leia CSVs ya calculados (cero llamadas a APIs en
tiempo de ejecucion). Esta version, para cumplir "que funcione con cualquier libro o
banda", llama en vivo a tres servicios nuevos cada vez que hace falta:
  1. Un LLM con function calling (Groq por defecto -- gratis -- u OpenAI) para entender
     el lenguaje natural del chat y redactar la respuesta.
  2. MusicBrainz, cuando preguntas por una banda que no esta en el catalogo de 81.
  3. Open Library / Google Books, cuando preguntas por un libro que no esta en el
     catalogo de 112.
Necesitas generar y configurar una API key para el LLM (ver seccion 0 del notebook / los
comentarios en app.py) -- sin eso el chat no puede responder.

Produce dos artefactos que comparten la misma logica:
1. inkriff_chat_recomendador.ipynb -- para correr en Colab (gr.ChatInterface(share=True)).
2. app.py + requirements.txt -- listos para Hugging Face Spaces (variables de entorno
   como Secrets del Space).
"""
import json

CORE_LOGIC = r'''import json
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import requests
import gradio as gr
from openai import OpenAI

# =========================================================================================
# 0. Configuracion del LLM (function calling) -- OpenAI es el proveedor PRIMARIO (asi lo pedia
# el plan original del proyecto) y Groq (gratis) es el RESPALDO automatico si OpenAI falla por
# cualquier motivo (cuota agotada, limite de velocidad, etc.) -- tal como se penso desde el
# canvas: "Groq/Gemini/DeepSeek -- respaldo si se satura la cuota". Configura las API keys como
# variables de entorno (en Colab: la celda "0. Tu API key de LLM" te las pide; en Hugging Face
# Spaces: agregalas en Settings -> Repository secrets). Con solo GROQ_API_KEY configurada (sin
# OPENAI_API_KEY) el chat funciona igual, usando directamente el respaldo.
# =========================================================================================
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
    Inkriff usa de verdad la API de OpenAI (como pedia el plan original) sin dejar de tener el
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


# =========================================================================================
# 1. Datos y modelo de embeddings -- mismo limpiador de texto y mismo modelo que en
# inkriff_modelado_similitud.ipynb, para que una busqueda en vivo sea comparable con lo ya
# evaluado.
# =========================================================================================
bandas = pd.read_csv("bandas_completo.csv")
libros = pd.read_csv("libros_con_sinopsis.csv")
rec_banda_a_libro = pd.read_csv("recomendaciones_banda_a_libro.csv")
rec_libro_a_banda = pd.read_csv("recomendaciones_libro_a_banda.csv")
libros["tiene_sinopsis"] = libros["sinopsis"].fillna("").str.strip().str.len() > 0

SPOTIFY_POR_BANDA = dict(zip(bandas["banda"], bandas["spotify_url"]))


def spotify_embed_html(spotify_url):
    """Reproductor embebido de Spotify sin necesitar API key: basta con reescribir el link
    publico de artista al formato /embed/ y ponerlo en un iframe (misma idea que ya se uso
    en la primera version del pipeline). Va en un contenedor propio con esquinas redondeadas
    para que combine con el tema visual incluso mientras carga."""
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
    """Carga el modelo de sentence-transformers una sola vez (peso ~470MB la primera vez
    -- normal que tarde uno o dos minutos la primera pregunta que use busqueda en vivo)."""
    global _MODELO_EMBEDDINGS
    if _MODELO_EMBEDDINGS is None:
        from sentence_transformers import SentenceTransformer
        _MODELO_EMBEDDINGS = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _MODELO_EMBEDDINGS


# =========================================================================================
# 2. Busqueda en vivo de metadatos (misma logica que inkriff_extraccion_completa.ipynb /
# inkriff_eda_ingenieria.ipynb, reusada aqui para bandas/libros que no estan en el catalogo).
# =========================================================================================
MB_HEADERS = {"User-Agent": "InkriffChat/0.1 (contacto: tu_correo@ejemplo.com)"}
MB_BASE = "https://musicbrainz.org/ws/2"


def _get_con_reintento(url, params, headers=None, max_retries=3, timeout=15):
    for intento in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (503, 429):
                time.sleep(2 ** intento)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError):
            time.sleep(2 ** intento)
    return None


def buscar_banda_en_vivo(nombre):
    """Busca una banda en MusicBrainz. Regresa una lista de tags (generos) o None si no
    se encontro nada -- estos tags son la unica señal de "mood" musical, igual que en la
    extraccion original."""
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
    Regresa el texto de la sinopsis o None si no se encontro nada util."""
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


# =========================================================================================
# 3. Recomendacion: catalogo entrenado primero (instantaneo, ya evaluado), busqueda en vivo
# como respaldo para cualquier libro/banda fuera de los 81/112.
# =========================================================================================
UMBRAL_MATCH_CATALOGO = 0.82


def _match_en_catalogo(nombre, columna_valores):
    mejor, mejor_score = None, 0.0
    for candidato in columna_valores:
        s = similitud_cadenas(nombre, candidato)
        if s > mejor_score:
            mejor, mejor_score = candidato, s
    return mejor if mejor_score >= UMBRAL_MATCH_CATALOGO else None


def recomendar_libros_para_banda(nombre_banda):
    match = _match_en_catalogo(nombre_banda, bandas["banda"].tolist())
    if match:
        fila = rec_banda_a_libro[rec_banda_a_libro["banda"] == match].sort_values("rank")
        recomendaciones = [
            {"nombre": r["libro_recomendado"], "categoria": r["categoria_libro"]}
            for _, r in fila.iterrows()
        ]
        return {"encontrado": True, "fuente": "catalogo", "tipo_resultado": "libros", "nombre_resuelto": match, "recomendaciones": recomendaciones}

    tags = buscar_banda_en_vivo(nombre_banda)
    if not tags:
        return {"encontrado": False, "mensaje": f"No encontre datos de la banda '{nombre_banda}' en MusicBrainz."}

    perfil = limpia_texto(((tags[0] + " ") * 3) + " ".join(tags))
    vec = get_modelo_embeddings().encode([perfil], normalize_embeddings=True)[0]
    raw = EMB_LIBROS @ vec  # similitud contra los 112 libros
    corregida = raw - LIBRO_ROW_MEANS - raw.mean() + GLOBAL_MEAN
    candidatos = np.where(libros["tiene_sinopsis"].values)[0]
    orden = candidatos[np.argsort(-corregida[candidatos])][:5]
    recomendaciones = [
        {"nombre": libros.iloc[i]["titulo_buscado"], "categoria": libros.iloc[i]["categoria_semilla"]}
        for i in orden
    ]
    return {"encontrado": True, "fuente": "en_vivo", "tipo_resultado": "libros", "nombre_resuelto": nombre_banda, "generos_encontrados": tags[:5], "recomendaciones": recomendaciones}


def recomendar_bandas_para_libro(titulo_libro):
    match = _match_en_catalogo(titulo_libro, libros["titulo_buscado"].tolist())
    if match:
        fila = rec_libro_a_banda[rec_libro_a_banda["libro"] == match].sort_values("rank")
        recomendaciones = [
            {
                "nombre": r["banda_recomendada"],
                "subgenero": r["subgenero_banda"],
                "spotify_url": SPOTIFY_POR_BANDA.get(r["banda_recomendada"]),
            }
            for _, r in fila.iterrows()
        ]
        return {"encontrado": True, "fuente": "catalogo", "tipo_resultado": "bandas", "nombre_resuelto": match, "recomendaciones": recomendaciones}

    sinopsis = buscar_libro_en_vivo(titulo_libro)
    if not sinopsis:
        return {"encontrado": False, "mensaje": f"No encontre sinopsis del libro '{titulo_libro}' en Open Library ni Google Books."}

    perfil = limpia_texto(sinopsis)
    if not perfil:
        return {"encontrado": False, "mensaje": f"Encontre '{titulo_libro}' pero su descripcion no tenia texto util."}
    vec = get_modelo_embeddings().encode([perfil], normalize_embeddings=True)[0]
    raw = EMB_BANDAS @ vec  # similitud contra las 81 bandas
    corregida = raw - BANDA_COL_MEANS - raw.mean() + GLOBAL_MEAN
    orden = np.argsort(-corregida)[:5]
    recomendaciones = [
        {
            "nombre": bandas.iloc[i]["banda"],
            "subgenero": bandas.iloc[i]["subgenero_semilla"],
            "spotify_url": SPOTIFY_POR_BANDA.get(bandas.iloc[i]["banda"]),
        }
        for i in orden
    ]
    return {"encontrado": True, "fuente": "en_vivo", "tipo_resultado": "bandas", "nombre_resuelto": titulo_libro, "recomendaciones": recomendaciones}


def canciones_de_banda(nombre_banda):
    """Canciones de una banda especifica -- no busca en una API nueva ni tiene datos de
    tracks individuales; usa el mismo enlace publico de Spotify que ya guardamos por banda
    (bandas_completo.csv) y aprovecha que el reproductor embebido de artista de Spotify
    muestra de entrada sus canciones mas populares. Por eso solo funciona para bandas del
    catalogo entrenado (las unicas de las que tenemos ese enlace) -- que son, ademas, las
    unicas bandas que esta app recomienda como salida."""
    match = _match_en_catalogo(nombre_banda, bandas["banda"].tolist())
    if not match:
        return {
            "encontrado": False,
            "mensaje": (
                f"No tengo un enlace de Spotify guardado para '{nombre_banda}' -- solo tengo "
                "para las bandas de mi catalogo entrenado (que son, ademas, las unicas que "
                "esta app recomienda)."
            ),
        }
    url = SPOTIFY_POR_BANDA.get(match)
    if not isinstance(url, str) or "open.spotify.com" not in url:
        return {"encontrado": False, "mensaje": f"Encontre a '{match}' en mi catalogo pero no tengo su enlace de Spotify guardado."}
    return {"encontrado": True, "tipo_resultado": "canciones", "nombre_resuelto": match, "spotify_url": url}


def ejecutar_recomendacion(tipo, nombre):
    """Version generica usada internamente/en pruebas -- el chat real llama directo a
    recomendar_libros_para_banda / recomendar_bandas_para_libro (ver HERRAMIENTAS)."""
    if not nombre or not nombre.strip():
        return {"encontrado": False, "mensaje": "No se especifico un nombre."}
    if tipo == "banda":
        return recomendar_libros_para_banda(nombre.strip())
    return recomendar_bandas_para_libro(nombre.strip())


# =========================================================================================
# 4. Chat con function calling
# =========================================================================================
# Se usan TRES herramientas separadas (una por tipo de resultado) en vez de una sola con un
# parametro "tipo" -- asi el propio nombre de la funcion ya dice que tipo de resultado
# regresa, y el LLM no puede confundir "el usuario menciono una banda" con "el resultado son
# bandas" (ese mezclado fue justo el bug reportado: pedia bandas parecidas a Bad Omens y la
# respuesta le llamaba "bandas" a lo que en realidad eran libros).
HERRAMIENTAS = [
    {
        "type": "function",
        "function": {
            "name": "libros_para_banda",
            "description": (
                "El usuario menciono una BANDA de rock/metal y quiere recomendaciones de "
                "LECTURA. Esta herramienta regresa una lista de LIBROS de fantasia/romance "
                "relacionados con esa banda -- el resultado son LIBROS, no otras bandas ni "
                "canciones. Funciona para cualquier banda, no solo un catalogo fijo. No uses "
                "esta herramienta si lo que pide el usuario es 'otras bandas parecidas' o "
                "'canciones de esa banda' -- para eso usa `canciones_de_banda`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nombre_banda": {"type": "string", "description": "El nombre de la banda que menciono el usuario."},
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
                "El usuario menciono un LIBRO de fantasia/romance y quiere recomendaciones "
                "de MUSICA. Esta herramienta regresa una lista de BANDAS de rock/metal "
                "relacionadas con ese libro -- el resultado son BANDAS, no otros libros. "
                "Funciona para cualquier libro, no solo un catalogo fijo. No uses esta "
                "herramienta si lo que pide el usuario es 'otros libros parecidos' -- eso no "
                "es lo que hace Inkriff."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "titulo_libro": {"type": "string", "description": "El titulo del libro que menciono el usuario."},
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
                "El usuario quiere ESCUCHAR o pide CANCIONES de una banda especifica -- por "
                "ejemplo una banda que ya se recomendo antes en la conversacion, o cualquier "
                "banda de rock/metal que menciona por su nombre. Regresa un reproductor de "
                "Spotify con las canciones mas populares de esa banda, listo para escuchar de "
                "inmediato. Solo funciona si la banda esta en el catalogo entrenado de "
                "Inkriff (que son, ademas, las unicas bandas que la app recomienda como "
                "salida, asi que cualquier banda que TU hayas recomendado antes siempre "
                "funciona aqui)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nombre_banda": {"type": "string", "description": "El nombre de la banda de la que el usuario quiere canciones."},
                },
                "required": ["nombre_banda"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "Eres el asistente de Inkriff, una plataforma que conecta musica de rock/metal con "
    "libros de fantasia/romance, y que ademas deja escuchar directamente a las bandas que "
    "recomienda. Tienes tres herramientas: `libros_para_banda` (dale el nombre de una "
    "banda, regresa LIBROS relacionados), `bandas_para_libro` (dale el titulo de un libro, "
    "regresa BANDAS relacionadas), y `canciones_de_banda` (dale el nombre de una banda, "
    "regresa un reproductor de Spotify con sus CANCIONES). Usa SIEMPRE la herramienta que "
    "corresponda cuando el usuario quiera una recomendacion -- nunca inventes resultados tu "
    "mismo, y nunca confundas el tipo de la entrada con el tipo del resultado: los "
    "resultados de `libros_para_banda` SIEMPRE son libros/novelas (nunca les digas "
    "'bandas'); los de `bandas_para_libro` SIEMPRE son bandas/grupos (nunca les digas "
    "'libros'); los de `canciones_de_banda` son canciones para escuchar, no una lista de "
    "texto. El campo 'tipo_resultado' de cada resultado ('libros', 'bandas' o 'canciones') "
    "te lo confirma -- usalo para no equivocarte.\n\n"
    "El punto de Inkriff es recomendar un poco de todo (libros, bandas Y canciones), asi "
    "que cuando el usuario mencione una banda o un libro sin aclarar que tipo de "
    "recomendacion quiere, es buena idea usar mas de una herramienta a la vez -- por "
    "ejemplo, si menciona una banda de forma general ('me encanta Behemoth', 'cuentame de "
    "Bad Omens'), usa tanto `libros_para_banda` (para sugerirle lectura) como "
    "`canciones_de_banda` (para que la pueda escuchar ahi mismo). Si el usuario es "
    "especifico ('solo quiero libros', 'ponme sus canciones'), usa unicamente la "
    "herramienta que corresponda a eso.\n\n"
    "Inkriff no tiene datos de canciones sueltas de bandas fuera de su catalogo, no "
    "recomienda otras bandas parecidas a una banda, ni otros libros parecidos a un libro. Si "
    "el usuario pide explicitamente algo de eso ('que otras bandas se parecen a Bad Omens', "
    "'que otros libros son como X'), acláralo con honestidad y brevedad (una frase, sin "
    "sonar repetitiva ni como disculpa larga) y de inmediato ofrece la alternativa cruzada "
    "que si tienes.\n\n"
    "Cuando das varias recomendaciones (varios libros o varias bandas), menciona el orden "
    "de forma natural (la primera, la segunda, la del top 3...) usando el mismo orden en el "
    "que te las dio la herramienta -- asi, si despues el usuario pregunta por 'la del top "
    "3' o 'la segunda', puedes saber exactamente a cual se refiere revisando el historial de "
    "la conversacion, y llamar a `canciones_de_banda` o `libros_para_banda` con el nombre "
    "correcto.\n\n"
    "Si una herramienta no encuentra nada, dilo con honestidad y sugiere revisar el nombre "
    "(por ejemplo probar el titulo en ingles). Si la fuente fue 'en_vivo', puedes mencionar "
    "brevemente que buscaste esos datos al momento porque no estaban en el catalogo "
    "original. Los resultados pueden traer un campo 'spotify_url' -- ignoralo al redactar tu "
    "respuesta (no lo menciones ni pegues el link). La aplicacion ya se encarga de mostrar un "
    "reproductor de Spotify por separado, y lo hace para TODAS las bandas de una "
    "recomendacion de `bandas_para_libro` (no solo la primera) -- asi que si el usuario pide "
    "escuchar las bandas que le acabas de recomendar de un libro, NO hace falta que llames "
    "`canciones_de_banda` de nuevo para cada una, ya se van a mostrar solas; usa "
    "`canciones_de_banda` solo para una banda puntual que no venga de esa lista reciente "
    "(por ejemplo una que el usuario menciono el mismo, o una de hace varios turnos).\n\n"
    "Ademas del reproductor, la aplicacion ya muestra los libros o bandas de cada "
    "recomendacion como tarjetas visuales (con su nombre y categoria/subgenero) debajo de tu "
    "mensaje -- por eso NO tienes que enumerar cada titulo o nombre uno por uno en tu texto, "
    "eso seria repetir la misma informacion dos veces. Tu escribes solo una reaccion breve y "
    "calida (una o dos frases) a lo que encontraste -- puedes mencionar por nombre una o dos "
    "si de verdad aportan algo al comentario (tu favorita, una que destaque), pero no listes "
    "las cinco.\n\n"
    "Responde en español, con un tono calido y natural, como platicando con alguien que "
    "sabe de musica y libros -- varia la forma en que presentas las recomendaciones y como "
    "arrancas cada respuesta (no repitas siempre la misma plantilla ni la misma frase de "
    "apertura), mantente breve, y no menciones puntajes de similitud ni detalles tecnicos "
    "del modelo salvo que te los pregunten explicitamente. Si el usuario solo esta "
    "platicando, o hace una pregunta de seguimiento sobre algo que ya se menciono antes en "
    "la conversacion, sigue el hilo con naturalidad en vez de tratar cada mensaje como si "
    "fuera aislado."
)


def _ejecutar_tool_call(tc):
    """Corre una sola tool call del LLM y regresa su resultado (dict)."""
    try:
        args = json.loads(tc.function.arguments)
    except (TypeError, json.JSONDecodeError):
        args = {}
    if tc.function.name == "bandas_para_libro":
        titulo = (args.get("titulo_libro") or "").strip()
        return recomendar_bandas_para_libro(titulo) if titulo else {"encontrado": False, "mensaje": "No se especifico un titulo de libro."}
    if tc.function.name == "libros_para_banda":
        nombre_b = (args.get("nombre_banda") or "").strip()
        return recomendar_libros_para_banda(nombre_b) if nombre_b else {"encontrado": False, "mensaje": "No se especifico el nombre de una banda."}
    if tc.function.name == "canciones_de_banda":
        nombre_b = (args.get("nombre_banda") or "").strip()
        return canciones_de_banda(nombre_b) if nombre_b else {"encontrado": False, "mensaje": "No se especifico el nombre de una banda."}
    return {"encontrado": False, "mensaje": f"Herramienta desconocida: {tc.function.name}"}


def _acumular_candidatos_embed(resultado, candidatos_embed):
    """Junta (nombre, spotify_url) de un resultado de herramienta -- si fueron BANDAS
    (recomendadas a partir de un libro) se agregan TODAS las que traigan spotify_url, no
    solo la #1, para poder listar el reproductor de cada una (a peticion explicita); si fue
    CANCIONES (el usuario pidio escuchar una banda puntual) se agrega esa sola."""
    if not resultado.get("encontrado"):
        return
    if resultado.get("tipo_resultado") == "canciones":
        candidatos_embed.append((resultado.get("nombre_resuelto"), resultado.get("spotify_url")))
    elif resultado.get("tipo_resultado") == "bandas":
        for rec in resultado.get("recomendaciones") or []:
            if rec.get("spotify_url"):
                candidatos_embed.append((rec["nombre"], rec["spotify_url"]))


def _escapar_html(texto):
    return (
        (texto or "")
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _construir_tarjetas(resultado):
    """Tarjetas visuales (HTML, construidas por codigo -- no por el LLM) para una lista de
    libros o bandas recomendadas: nombre + categoria/subgenero, en el orden de la
    herramienta. Reemplazan la lista en texto plano que antes redactaba el LLM."""
    tipo = resultado.get("tipo_resultado")
    if tipo not in ("libros", "bandas") or not resultado.get("encontrado"):
        return ""
    recomendaciones = resultado.get("recomendaciones") or []
    if not recomendaciones:
        return ""
    icono = "📚" if tipo == "libros" else "🎸"
    etiqueta_campo = "categoria" if tipo == "libros" else "subgenero"
    titulo_seccion = "Libros recomendados" if tipo == "libros" else "Bandas recomendadas"
    filas = "".join(
        f'<div class="inkriff-card">'
        f'<span class="inkriff-card-rank">{i + 1}</span>'
        f'<span class="inkriff-card-icon">{icono}</span>'
        f'<span class="inkriff-card-nombre">{_escapar_html(rec.get("nombre"))}</span>'
        f'<span class="inkriff-badge inkriff-badge-{tipo}">{_escapar_html(rec.get(etiqueta_campo) or "")}</span>'
        f'</div>'
        for i, rec in enumerate(recomendaciones)
    )
    return f'<div class="inkriff-cards"><div class="inkriff-cards-titulo">{titulo_seccion}</div>{filas}</div>'


def _construir_embeds(candidatos_embed):
    vistos, partes = set(), []
    for nombre, spotify_url in candidatos_embed:
        if nombre in vistos:
            continue
        vistos.add(nombre)
        embed = spotify_embed_html(spotify_url)
        if embed:
            partes.append(f'<div class="inkriff-player-label">🎧 Escucha a {_escapar_html(nombre)}</div>{embed}')
    return "".join(partes)


MAX_RONDAS_HERRAMIENTAS = 5  # tope de idas y vueltas con herramientas en un solo turno


def generar_respuesta_completa(mensaje, historial):
    """Toda la logica de una respuesta del chat (decidir si hace falta una herramienta,
    correrla -- posiblemente en varias rondas, si el LLM necesita encadenar mas de una -- y
    redactar el texto final). Regresa (texto, embeds_html) por separado para que quien la use
    pueda, por ejemplo, mostrar el texto con efecto de maquina de escribir y pegar los
    reproductores de Spotify completos solo al final (nunca a la mitad de un <iframe>).

    Importante: las herramientas (`tools=HERRAMIENTAS, tool_choice="auto"`) se mandan en
    TODAS las rondas, incluyendo la que redacta el texto final -- si se omiten en esa ultima
    llamada (como se hacia antes), un modelo que todavia quiere usar una herramienta puede
    intentarlo de todos modos y Groq lo rechaza con un 400 ('Tool choice is none, but model
    called a tool'). Dejando 'auto' siempre disponible, el modelo simplemente sigue pidiendo
    herramientas (hasta el tope de rondas) o entrega su respuesta de texto cuando ya esta
    listo, sin ese choque."""
    if not _tiene_key(LLM_PROVEEDOR_PRIMARIO) and not _tiene_key(LLM_PROVEEDOR_RESPALDO):
        return (
            "Todavia no tengo configurada ninguna API key de LLM. Define la variable de "
            "entorno OPENAI_API_KEY (la que usa esta app por defecto) o, como alternativa "
            "gratis, GROQ_API_KEY (en https://console.groq.com/keys) antes de correr esta "
            "celda/app."
        ), ""

    # Gradio manda cada turno del historial con claves propias de su UI (metadata, options,
    # etc.) ademas de role/content -- Groq (y OpenAI en modo estricto) rechazan esas claves
    # extra con un 400, asi que nos quedamos solo con role/content antes de reenviarlo.
    historial_limpio = [
        {"role": m.get("role"), "content": m.get("content") or ""}
        for m in historial
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
    ]
    mensajes = [{"role": "system", "content": SYSTEM_PROMPT}] + historial_limpio + [{"role": "user", "content": mensaje}]

    candidatos_embed = []  # (nombre, spotify_url) -- para el/los reproductor(es) al final
    tarjetas_html = ""  # tarjetas visuales de cada lista de libros/bandas, en orden
    texto = None
    for ronda in range(MAX_RONDAS_HERRAMIENTAS):
        try:
            resp = _llamar_llm(messages=mensajes, tools=HERRAMIENTAS, tool_choice="auto")
        except Exception as e:
            # Se muestra el detalle real del error (no solo el tipo de excepcion) para poder
            # diagnosticar (revisa tambien la celda "Prueba de conexion" de mas arriba).
            extra = tarjetas_html + _construir_embeds(candidatos_embed)
            if ronda == 0:
                return f"Hubo un problema hablando con el modelo de lenguaje:\n\n`{type(e).__name__}: {e}`", ""
            return f"Encontre resultados pero hubo un problema redactando la respuesta:\n\n`{type(e).__name__}: {e}`", extra

        msg = resp.choices[0].message
        if not msg.tool_calls:
            texto = msg.content or ""
            break

        mensajes.append(msg)
        for tc in msg.tool_calls:
            resultado = _ejecutar_tool_call(tc)
            tarjetas_html += _construir_tarjetas(resultado)
            _acumular_candidatos_embed(resultado, candidatos_embed)
            mensajes.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(resultado, ensure_ascii=False)})

    if texto is None:
        # Se acabaron las rondas sin que el LLM diera una respuesta final -- igual regresamos
        # lo que ya encontramos (texto, tarjetas y reproductores) en vez de dejar al usuario
        # sin nada.
        texto = (
            "Encontre varios resultados pero me tarde demasiado armando la respuesta final -- "
            "aqui tienes lo que ya reuni; intenta preguntar de nuevo, quizas mas especifico, "
            "si te falto algo."
        )

    return texto, tarjetas_html + _construir_embeds(candidatos_embed)


def responder_chat(mensaje, historial):
    """Generador para gr.ChatInterface: revela el texto del LLM palabra por palabra (efecto
    de estar escribiendo en vivo, como en una conversacion normal) y agrega las tarjetas de
    recomendaciones y el/los reproductor(es) de Spotify ya completos hasta el final, para no
    mandar nunca HTML a medio armar mientras se va revelando el texto."""
    texto, extra_html = generar_respuesta_completa(mensaje, historial)
    palabras = texto.split(" ") if texto else []
    acumulado = ""
    for palabra in palabras:
        acumulado = f"{acumulado} {palabra}".strip() if acumulado else palabra
        yield acumulado
        time.sleep(0.02)
    yield (acumulado + extra_html) if extra_html else (acumulado or texto)


# =========================================================================================
# 5. Tema visual e interfaz -- paleta "metal se encuentra con fantasia" (violeta/oro sobre
# fondo casi negro), la misma en modo claro y oscuro del navegador para que la marca se vea
# igual siempre, mas tarjetas de recomendacion y reproductores con estilo propio via CSS.
# =========================================================================================
TEMA_INKRIFF = gr.themes.Base(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#0d0b14",
    body_background_fill_dark="#0d0b14",
    body_text_color="#f3f1f7",
    body_text_color_dark="#f3f1f7",
    body_text_color_subdued="#a89cc8",
    body_text_color_subdued_dark="#a89cc8",
    block_background_fill="#1a1625",
    block_background_fill_dark="#1a1625",
    block_border_color="#3a2f52",
    block_border_color_dark="#3a2f52",
    block_title_text_color="#d4af37",
    block_title_text_color_dark="#d4af37",
    block_label_text_color="#a89cc8",
    block_label_text_color_dark="#a89cc8",
    input_background_fill="#221c33",
    input_background_fill_dark="#221c33",
    border_color_primary="#3a2f52",
    border_color_primary_dark="#3a2f52",
    button_primary_background_fill="linear-gradient(90deg, #7c3aed, #a855f7)",
    button_primary_background_fill_dark="linear-gradient(90deg, #7c3aed, #a855f7)",
    button_primary_background_fill_hover="linear-gradient(90deg, #6d28d9, #9333ea)",
    button_primary_background_fill_hover_dark="linear-gradient(90deg, #6d28d9, #9333ea)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#221c33",
    button_secondary_background_fill_dark="#221c33",
    button_secondary_text_color="#d4af37",
    button_secondary_text_color_dark="#d4af37",
    button_secondary_border_color="#3a2f52",
    button_secondary_border_color_dark="#3a2f52",
)

CSS_INKRIFF = """
.gradio-container { max-width: 780px !important; margin: 0 auto !important; }
.inkriff-header { text-align: center; padding: 14px 12px 4px 12px; }
.inkriff-header-emoji { font-size: 2em; line-height: 1; }
.inkriff-header-title {
    font-size: 1.9em; font-weight: 800; letter-spacing: .04em; margin-top: 2px;
    color: #d4af37; /* respaldo por si el navegador no soporta el degradado de abajo */
    background: linear-gradient(90deg, #f4d160, #9333ea);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.inkriff-header-tagline { color: #a89cc8; font-size: 0.95em; margin-top: 2px; }
.inkriff-cards {
    margin-top: 10px; padding: 14px 16px; border-radius: 14px;
    background: linear-gradient(135deg, #1a1625, #241c38); border: 1px solid #3a2f52;
}
.inkriff-cards-titulo {
    font-weight: 700; color: #d4af37; letter-spacing: .04em;
    margin-bottom: 8px; font-size: 0.75em; text-transform: uppercase;
}
.inkriff-card {
    display: flex; align-items: center; gap: 10px; padding: 7px 4px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.inkriff-card:last-child { border-bottom: none; }
.inkriff-card-rank { font-weight: 700; color: #a89cc8; width: 1.5em; text-align: center; flex-shrink: 0; }
.inkriff-card-icon { font-size: 1.05em; flex-shrink: 0; }
.inkriff-card-nombre { flex: 1; color: #f3f1f7; font-weight: 600; }
.inkriff-badge {
    font-size: 0.72em; padding: 3px 10px; border-radius: 999px; font-weight: 600;
    white-space: nowrap; flex-shrink: 0;
}
.inkriff-badge-libros { background: rgba(45, 212, 191, 0.15); color: #2dd4bf; border: 1px solid rgba(45,212,191,0.4); }
.inkriff-badge-bandas { background: rgba(225, 29, 72, 0.15); color: #fb7185; border: 1px solid rgba(225,29,72,0.4); }
.inkriff-player-label { font-weight: 700; color: #d4af37; font-size: 0.82em; margin: 12px 0 6px 0; }
.inkriff-player {
    border-radius: 12px; overflow: hidden; border: 1px solid #3a2f52; background: #120f1c;
}
.inkriff-player iframe { display: block; }
.inkriff-landing { padding: 4px 6px 16px 6px; }
.inkriff-landing-section { margin-bottom: 18px; }
.inkriff-landing-section h3 { color: #d4af37; font-size: 1.05em; margin-bottom: 6px; }
.inkriff-landing-section p { color: #f3f1f7; line-height: 1.5; margin: 0; }
.inkriff-stats-row { display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap; }
.inkriff-stat {
    flex: 1; min-width: 110px; text-align: center; padding: 14px 8px; border-radius: 14px;
    background: linear-gradient(135deg, #1a1625, #241c38); border: 1px solid #3a2f52;
}
.inkriff-stat-numero { font-size: 1.6em; font-weight: 800; color: #f4d160; }
.inkriff-stat-etiqueta { font-size: 0.78em; color: #a89cc8; text-transform: uppercase; letter-spacing: .03em; margin-top: 2px; }
.inkriff-landing-lista { list-style: none; padding: 0; margin: 0; }
.inkriff-landing-lista li {
    padding: 9px 4px; border-bottom: 1px solid rgba(255,255,255,0.06); color: #f3f1f7; line-height: 1.45;
}
.inkriff-landing-lista li:last-child { border-bottom: none; }
.inkriff-landing-lista strong { color: #2dd4bf; }
.inkriff-landing-creditos {
    text-align: center; color: #a89cc8; font-size: 0.82em; margin-top: 6px;
    padding-top: 14px; border-top: 1px solid #3a2f52;
}
footer { display: none !important; }
"""

HEADER_HTML = """<div class="inkriff-header">
<div class="inkriff-header-emoji">🎸📖</div>
<div class="inkriff-header-title">Inkriff</div>
<div class="inkriff-header-tagline">Donde el metal se encuentra con la fantasía</div>
</div>"""

LANDING_HTML = """<div class="inkriff-landing">
<div class="inkriff-landing-section">
<h3>¿Qué es Inkriff?</h3>
<p>Un recomendador bidireccional entre música de rock/metal y libros de fantasía/romance:
te sugiere bandas a partir de un libro que te gustó, o libros a partir de una banda que
escuchas -- y te deja escuchar sus canciones ahí mismo, sin salir del chat.</p>
</div>
<div class="inkriff-stats-row">
<div class="inkriff-stat"><div class="inkriff-stat-numero">81</div><div class="inkriff-stat-etiqueta">Bandas</div></div>
<div class="inkriff-stat"><div class="inkriff-stat-numero">112</div><div class="inkriff-stat-etiqueta">Libros</div></div>
<div class="inkriff-stat"><div class="inkriff-stat-numero">3</div><div class="inkriff-stat-etiqueta">Fuentes de datos</div></div>
</div>
<div class="inkriff-landing-section">
<h3>¿Cómo funciona?</h3>
<ul class="inkriff-landing-lista">
<li><strong>Datos:</strong> MusicBrainz (géneros/subgéneros de bandas), Open Library y Google Books (sinopsis de libros) -- todo gratis, sin llaves.</li>
<li><strong>Modelo:</strong> embeddings de sentence-transformers multilingües, con una corrección de "hubness" para que no dominen siempre las mismas bandas o libros en las recomendaciones.</li>
<li><strong>Chat:</strong> un LLM (OpenAI, con Groq como respaldo automático) entiende lo que preguntas y llama a la herramienta de recomendación correcta.</li>
<li><strong>Escucha:</strong> reproductor de Spotify embebido, sin necesitar una API key de Spotify.</li>
</ul>
</div>
<div class="inkriff-landing-creditos">Proyecto de Viry — Diplomado en Ciencia de Datos, FES Acatlán, UNAM.</div>
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
                    "Quiero escuchar canciones de Bad Omens",
                    "Cuéntame un libro de fantasía que te guste y te digo qué banda combina",
                ],
                # sanitize_html=False: sin esto Gradio borra el <iframe> del reproductor de
                # Spotify y las tarjetas de recomendacion que agregamos en responder_chat. Todo
                # ese HTML lo construye nuestro propio codigo (nunca texto libre del usuario ni
                # del LLM), asi que es seguro desactivar el sanitizado.
                chatbot=gr.Chatbot(sanitize_html=False, show_label=False),
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

cells.append(md("""# Inkriff — Chat conversacional (recomendador con LLM + búsqueda en vivo)

Esta es la versión conversacional del pipeline: escribes en lenguaje natural (por ejemplo
*"¿qué banda me recomiendas si me gustó Trono de Cristal?"* o *"acabo de descubrir a
Wintersun, ¿qué debería leer?"*) y un LLM con **function calling** entiende la intención,
llama a la misma lógica de recomendación de siempre, y te responde de forma conversacional.

**Ya no está limitado al catálogo de 81 bandas / 112 libros que entrenamos**: si mencionas
algo que no está ahí, el notebook busca sus datos en vivo (MusicBrainz para bandas, Open
Library/Google Books para libros), calcula su embedding al momento con el mismo modelo de
`inkriff_modelado_similitud.ipynb`, y aplica la misma corrección de hubness ya evaluada.

**Cambio de arquitectura importante:** la versión anterior del pipeline (dropdown + tabla)
solo leía CSVs ya calculados, cero llamadas a APIs en tiempo de ejecución. Esta versión
llama en vivo a tres servicios nuevos cada vez que hace falta: un LLM (Groq por defecto,
gratis, u OpenAI), MusicBrainz, y Open Library/Google Books.

**De vuelta el reproductor de Spotify -- y ahora también canciones:** cuando el chat te
recomienda bandas (porque mencionaste un libro), o cuando le pides directamente escuchar a
una banda (por ejemplo una que ya te haya recomendado antes: *"ponme las canciones de la
que quedó en el top 3"*), la respuesta incluye un reproductor embebido de Spotify con sus
canciones más populares -- el mismo truco sin API key de la primera versión del pipeline
(reescribe el link público de artista al formato `/embed/`), ahora integrado dentro del
chat. Solo funciona con bandas del catálogo entrenado (que son, además, las únicas que la
app recomienda como salida).

**Más pulido en esta versión:** el texto de la respuesta aparece palabra por palabra (como
en una conversación real, no todo de golpe), hay ejemplos sugeridos debajo del cuadro de
texto para arrancar, el chat sigue el hilo de la conversación (puedes preguntar por "la
segunda banda" o "la del top 3" y sabe a cuál te refieres), y se corrigió un bug donde a
veces le llamaba "bandas" a resultados que en realidad eran libros (o viceversa).

**Antes de correr, sube:**
- `bandas_completo.csv`, `libros_con_sinopsis.csv` (del universo entrenado)
- `recomendaciones_banda_a_libro.csv`, `recomendaciones_libro_a_banda.csv` (del recomendador ya evaluado)
- `embeddings_referencia.npz` (nuevo — lo genera la sección 3.2, recién agregada, de
  `inkriff_modelado_similitud.ipynb`; **tienes que volver a correr ese notebook una vez**
  para generarlo antes de poder usar este chat)

**Y necesitas una API key de LLM** (ver la celda de configuración abajo): la app usa OpenAI
como proveedor principal (`OPENAI_API_KEY`, de pago) y cae automáticamente a Groq (gratis,
`GROQ_API_KEY` en https://console.groq.com/keys) como respaldo si algo falla con OpenAI —
puedes poner solo una de las dos si prefieres, pero configurar ambas es lo más robusto.
"""))

cells.append(code('''!pip -q install gradio openai sentence-transformers
'''))

cells.append(code('''try:
    from google.colab import files
    print("Sube bandas_completo.csv, libros_con_sinopsis.csv, recomendaciones_banda_a_libro.csv, recomendaciones_libro_a_banda.csv y embeddings_referencia.npz:")
    files.upload()
except ImportError:
    pass
'''))

cells.append(md("""## 0. Tu API key de LLM

Esta app usa **OpenAI como proveedor principal** (asi se penso desde el plan original del
proyecto) y **Groq como respaldo automatico** -- si algo falla con OpenAI (se acaba tu cuota,
un limite de velocidad, etc.), el chat reintenta solo con Groq (gratis) sin que tengas que
hacer nada. Pega tu OPENAI_API_KEY abajo; la de Groq es opcional pero muy recomendable
(tarda 30 segundos crearla gratis). Ninguna de las dos queda guardada en el notebook.
"""))

cells.append(code('''import os
from getpass import getpass

os.environ["INKRIFF_LLM_PROVEEDOR"] = "openai"  # cambia a "groq" si solo quieres usar la gratuita
os.environ["OPENAI_API_KEY"] = getpass("Pega tu OPENAI_API_KEY (https://platform.openai.com/api-keys): ")

_groq_key = getpass("(Opcional, recomendado) Pega tu GROQ_API_KEY como respaldo gratis, o deja vacio y da Enter: ")
if _groq_key:
    os.environ["GROQ_API_KEY"] = _groq_key
'''))

cells.append(md("""## 0.1 Prueba de conexión (corre esto antes que el chat)

Hace una sola llamada simple a cada proveedor con key configurada, sin herramientas ni datos
de por medio, para separar "el LLM no me contesta" de cualquier otro bug del recomendador. Si
esta celda falla, el error que imprime ya te dice exactamente qué está mal (key incorrecta,
modelo no disponible, cuota agotada, etc.) — cópiamelo si necesitas que lo revisemos juntas.
"""))

cells.append(code('''from openai import OpenAI

_config_prueba = {
    "openai": (os.environ.get("OPENAI_API_KEY"), os.environ.get("INKRIFF_LLM_MODELO_OPENAI", "gpt-4o-mini"), None),
    "groq": (os.environ.get("GROQ_API_KEY"), os.environ.get("INKRIFF_LLM_MODELO_GROQ", "openai/gpt-oss-20b"), "https://api.groq.com/openai/v1"),
}
for _prov, (_key, _modelo_prueba, _base_url) in _config_prueba.items():
    if not _key:
        print(f"Proveedor: {_prov} | sin API key configurada -- se omite")
        continue
    print(f"Proveedor: {_prov} | Modelo: {_modelo_prueba}")
    try:
        _cliente_prueba = OpenAI(api_key=_key, base_url=_base_url) if _base_url else OpenAI(api_key=_key)
        _resp_prueba = _cliente_prueba.chat.completions.create(
            model=_modelo_prueba,
            messages=[{"role": "user", "content": "Responde solo con la palabra: listo"}],
        )
        print("  Conexion OK. Respuesta del modelo:", _resp_prueba.choices[0].message.content)
    except Exception as e:
        print(f"  FALLO la conexion -- {type(e).__name__}: {e}")
'''))

cells.append(md("## Chat\n"))
cells.append(code(CORE_LOGIC))

cells.append(code('''demo.launch(share=True, theme=TEMA_INKRIFF, css=CSS_INKRIFF)
'''))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("inkriff_chat_recomendador.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Notebook generado: inkriff_chat_recomendador.ipynb")

# ============================== 2. app.py para Hugging Face Spaces ==============================
APP_PY = '''"""Inkriff -- Chat conversacional (Hugging Face Spaces).

Sube junto a este archivo: bandas_completo.csv, libros_con_sinopsis.csv,
recomendaciones_banda_a_libro.csv, recomendaciones_libro_a_banda.csv y
embeddings_referencia.npz (este ultimo lo genera inkriff_modelado_similitud.ipynb).

Configura en Settings -> Repository secrets del Space: OPENAI_API_KEY (proveedor principal)
y, opcional pero recomendado, GROQ_API_KEY (gratis, en https://console.groq.com/keys) como
respaldo automatico si algo falla con OpenAI. Si solo quieres usar la gratuita, define
INKRIFF_LLM_PROVEEDOR=groq y basta con GROQ_API_KEY.
"""
''' + CORE_LOGIC + '''

if __name__ == "__main__":
    demo.launch(theme=TEMA_INKRIFF, css=CSS_INKRIFF)
'''

with open("app.py", "w", encoding="utf-8") as f:
    f.write(APP_PY)
print("Archivo generado: app.py")

with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write("gradio>=4.0\npandas\nnumpy\nrequests\nopenai\nsentence-transformers\nscikit-learn\n")
print("Archivo generado: requirements.txt")
