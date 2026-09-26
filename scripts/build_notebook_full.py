# -*- coding: utf-8 -*-
"""Genera inkriff_extraccion_completa.ipynb — version escalada del piloto a TODO
el universo semilla (todas las bandas y libros, no solo el primero de cada categoria).

v2: el universo semilla (SEED_BANDS/SEED_BOOKS) se importa de seed_data.py, que ahora
incluye tambien las ~41 bandas y ~98 libros nuevos de pares_semilla_final.csv (la lista
de 105 pares libro<->banda de Viry), para que la extraccion cubra tambien los anclajes
de entrenamiento supervisado del dual encoder."""
import json
from pprint import pformat

from seed_data import SEED_BANDS, SEED_BOOKS

N_BANDS = sum(len(v) for v in SEED_BANDS.values())
N_BOOKS = sum(len(v) for v in SEED_BOOKS.values())
SEED_BANDS_SRC = "SEED_BANDS = " + pformat(SEED_BANDS, width=100, sort_dicts=False)
SEED_BOOKS_SRC = "SEED_BOOKS = " + pformat(SEED_BOOKS, width=100, sort_dicts=False)

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }

cells = []

cells.append(md(f"""# Inkriff — Extraccion completa de datos

Version escalada del notebook piloto (`inkriff_pilot_extraccion.ipynb`), que ya
validamos: usa el MISMO codigo probado (MusicBrainz con reintentos, Genius solo
metadatos oficiales, Open Library con Google Books de respaldo), pero corre sobre
**todo el universo semilla** (~{N_BANDS} bandas, ~{N_BOOKS} libros) en vez de solo una
muestra.

Esta version (v2) ya incluye las bandas y libros de `pares_semilla_final.csv` — los 105
pares libro<->banda que curaste a partir de tu propio conocimiento de fandom — ademas
del universo original. Asi la extraccion cubre tambien los anclajes que vamos a usar
como entrenamiento supervisado del dual encoder, no solo la muestra inicial.

Cada banda y cada libro queda etiquetado con su subgenero/categoria semilla, para que
el CSV resultante ya se pueda usar directo en la siguiente fase (EDA / ingenieria de
variables).

**Tiempo estimado:** unos 10-15 minutos en total (MusicBrainz es la fuente mas lenta por
su limite de tasa — son ~{N_BANDS} bandas x ~3s cada una, mas reintentos; es normal ver
algun `[MusicBrainz 503] reintentando...` en el camino. Si Colab se desconecta a mitad,
solo vuelve a correr las celdas en orden).

Corre las celdas en orden.
"""))

cells.append(code("""!pip -q install requests
"""))

cells.append(md("""## 0. Universo de datos semilla completo
"""))

cells.append(code(SEED_BANDS_SRC + "\n\n" + SEED_BOOKS_SRC + '''

# Diccionarios inversos: banda/libro -> su subgenero/categoria semilla
BAND_TO_SUBGENRE = {banda: sg for sg, bandas in SEED_BANDS.items() for banda in bandas}
BOOK_TO_CATEGORY = {libro: cat for cat, libros in SEED_BOOKS.items() for libro in libros}

ALL_BANDS = list(BAND_TO_SUBGENRE.keys())
ALL_BOOKS = list(BOOK_TO_CATEGORY.keys())
print(f"{len(ALL_BANDS)} bandas | {len(ALL_BOOKS)} libros")
'''))

cells.append(md("""## 1. MusicBrainz — todas las bandas

Requiere un `User-Agent` personalizado y respetar el limite de ~1 request/segundo
(uso anonimo). No requiere API key. Reintenta automaticamente ante 503/429.
"""))

cells.append(code('''import time
import requests

MB_HEADERS = {"User-Agent": "InkriffDataCollector/0.1 (contacto: tu_correo@ejemplo.com)"}
MB_BASE = "https://musicbrainz.org/ws/2"


def mb_get(url, params, max_retries=5):
    """GET a un endpoint de MusicBrainz con reintento + backoff ante 503/429."""
    for intento in range(max_retries):
        r = requests.get(url, params=params, headers=MB_HEADERS, timeout=20)
        if r.status_code in (503, 429):
            espera = 2 ** intento  # 1, 2, 4, 8, 16 segundos
            print(f"  [MusicBrainz {r.status_code}] reintentando en {espera}s...")
            time.sleep(espera)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"MusicBrainz no respondio tras {max_retries} intentos: {url}")


def mb_search_artist(name):
    data = mb_get(f"{MB_BASE}/artist", {"query": f\'artist:"{name}"\', "fmt": "json", "limit": 1})
    return data["artists"][0] if data.get("artists") else None


def mb_artist_detail(mbid):
    return mb_get(f"{MB_BASE}/artist/{mbid}", {"fmt": "json", "inc": "tags+url-rels"})


mb_results = []
for name in ALL_BANDS:
    try:
        artist = mb_search_artist(name)
        time.sleep(1.5)  # un poco mas holgado que el minimo de 1 req/seg
        if not artist:
            print(f"[!] No encontrado en MusicBrainz: {name}")
            continue
        detail = mb_artist_detail(artist["id"])
        time.sleep(1.5)
        tags = sorted(detail.get("tags", []), key=lambda t: t.get("count", 0), reverse=True)
        spotify_url = next(
            (rel["url"]["resource"] for rel in detail.get("relations", [])
             if "spotify.com" in rel.get("url", {}).get("resource", "")),
            None,
        )
        row = {
            "banda": name,
            "subgenero_semilla": BAND_TO_SUBGENRE[name],
            "mbid": artist["id"],
            "pais": detail.get("country"),
            "tags_top": [t["name"] for t in tags[:5]],
            "spotify_url": spotify_url,
        }
        mb_results.append(row)
        print(row)
    except Exception as e:
        print(f"[!] Error con {name}: {e}")
        continue

print(f"\\nTotal de bandas con datos de MusicBrainz: {len(mb_results)} de {len(ALL_BANDS)}")
'''))

cells.append(md("""## 2. Genius — todas las bandas (API oficial, solo metadatos)

Igual que en el piloto: solo metadatos (titulo, popularidad), nunca letras — Genius
bloquea el scraping con Cloudflare/reCAPTCHA (403), ademas de que iba contra sus
Terminos de Servicio.
"""))

cells.append(code('''GENIUS_TOKEN = "PEGA_AQUI_TU_TOKEN"  # https://genius.com/api-clients
GENIUS_HEADERS = {"Authorization": f"Bearer {GENIUS_TOKEN}"}


def genius_search_song(name):
    r = requests.get(
        "https://api.genius.com/search",
        params={"q": name},
        headers=GENIUS_HEADERS, timeout=15,
    )
    r.raise_for_status()
    hits = r.json().get("response", {}).get("hits", [])
    return hits[0]["result"] if hits else None


genius_results = []
for name in ALL_BANDS:
    try:
        result = genius_search_song(name)
        time.sleep(0.5)
        if not result:
            print(f"[!] No encontrado en Genius: {name}")
            continue
        row = {
            "banda_buscada": name,
            "titulo_top": result.get("title"),
            "artista_genius": result.get("primary_artist", {}).get("name"),
            "genius_id": result.get("id"),
            "pageviews": result.get("stats", {}).get("pageviews"),
        }
        genius_results.append(row)
        print(row)
    except Exception as e:
        print(f"[!] Error con {name}: {type(e).__name__}: {e}")

print(f"\\nTotal de bandas con datos de Genius: {len(genius_results)} de {len(ALL_BANDS)}")
'''))

cells.append(md("""## 3. Open Library — todos los libros (con Google Books como respaldo)

**Nota:** igual que con MusicBrainz, es normal que Open Library o Google Books tarden
o no respondan de vez en cuando (`ConnectTimeout`, `ReadTimeout`) — son APIs publicas
compartidas por muchisima gente, sobre todo desde Colab. El codigo de abajo reintenta
automaticamente con backoff, y si una fuente sigue sin responder despues de varios
intentos, lo dice mas no truena el resto del ciclo.
"""))

cells.append(code('''def get_con_reintento(url, params, max_retries=4, timeout=20):
    """GET con reintento + backoff ante 503/429 y ante fallas de conexion/timeout."""
    for intento in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in (503, 429):
                espera = 2 ** intento
                print(f"  [{r.status_code}] reintentando en {espera}s...")
                time.sleep(espera)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError) as e:
            espera = 2 ** intento
            print(f"  [{type(e).__name__}] reintentando en {espera}s...")
            time.sleep(espera)
    raise RuntimeError(f"No se pudo conectar tras {max_retries} intentos: {url}")


def openlibrary_search(title):
    try:
        data = get_con_reintento("https://openlibrary.org/search.json", {"q": title, "limit": 1})
        docs = data.get("docs", [])
        return docs[0] if docs else None
    except Exception as e:
        print(f"  [Open Library] no disponible para \\"{title}\\": {type(e).__name__}: {e}")
        return None


def googlebooks_search(title):
    try:
        data = get_con_reintento("https://www.googleapis.com/books/v1/volumes", {"q": title, "maxResults": 1})
        items = data.get("items", [])
        return items[0] if items else None
    except Exception as e:
        print(f"  [Google Books] no disponible para \\"{title}\\": {type(e).__name__}: {e}")
        return None


book_results = []
for title in ALL_BOOKS:
    ol = openlibrary_search(title)
    row = {"titulo_buscado": title, "categoria_semilla": BOOK_TO_CATEGORY[title]}
    if ol:
        row.update({
            "titulo": ol.get("title"),
            "autor": ", ".join(ol.get("author_name", [])),
            "anio": ol.get("first_publish_year"),
            "fuente": "Open Library",
        })
    else:
        gb = googlebooks_search(title)
        if gb:
            info = gb.get("volumeInfo", {})
            row.update({
                "titulo": info.get("title"),
                "autor": ", ".join(info.get("authors", [])),
                "anio": info.get("publishedDate"),
                "fuente": "Google Books",
            })
        else:
            row.update({"titulo": None, "autor": None, "anio": None, "fuente": None})
    book_results.append(row)
    print(row)
    time.sleep(0.3)

print(f"\\nTotal de libros con datos: {len(book_results)} de {len(ALL_BOOKS)}")
'''))

cells.append(md("""## 4. Consolidar dataset completo

Guarda 3 CSVs por fuente, mas un cuarto (`bandas_completo.csv`) que ya junta
MusicBrainz + Genius por banda, listo para la siguiente fase.
"""))

cells.append(code('''import pandas as pd

df_bandas = pd.DataFrame(mb_results)
df_genius = pd.DataFrame(genius_results)
df_libros = pd.DataFrame(book_results)

df_bandas.to_csv("bandas_musicbrainz.csv", index=False)
df_genius.to_csv("metadatos_genius.csv", index=False)
df_libros.to_csv("libros.csv", index=False)

df_bandas_completo = df_bandas.merge(
    df_genius, left_on="banda", right_on="banda_buscada", how="left"
).drop(columns=["banda_buscada"])
df_bandas_completo.to_csv("bandas_completo.csv", index=False)

print("Bandas (MusicBrainz + Genius):")
display(df_bandas_completo)
print("Libros:")
display(df_libros)
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

with open("inkriff_extraccion_completa.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook generado: inkriff_extraccion_completa.ipynb")
