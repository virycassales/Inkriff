# -*- coding: utf-8 -*-
"""Genera inkriff_pilot_extraccion.ipynb a partir de celdas definidas aqui."""
import json

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

cells.append(md("""# Inkriff — Piloto de extraccion de datos

Este notebook prueba, con un puñado de ejemplos, que las fuentes de datos del proyecto
funcionan como esperamos:

1. **MusicBrainz** — metadatos y generos/subgeneros de bandas de metal/rock.
2. **Genius** (API oficial, solo metadatos) — titulo, artista y popularidad de la
   cancion mas conocida de cada banda. **No se hace scraping de letras**: se probo y
   Genius lo bloquea a nivel tecnico (Cloudflare + reCAPTCHA, error 403), ademas de que
   iba contra sus Terminos de Servicio. Los tags/generos de MusicBrainz quedan como la
   señal principal de "mood" musical para el modelo.
3. **Open Library** (con Google Books como respaldo) — metadatos y sinopsis de libros.

**Por que corre en Colab y no en el entorno de Claude:** el entorno de Claude en el que
se redacto este notebook tiene el acceso a internet restringido a un whitelist (pypi,
npm, la API de Anthropic); no puede llamar a APIs externas como MusicBrainz o Genius.
Colab (y tu propia maquina) no tienen esa restriccion, asi que este es el lugar correcto
para validar el pipeline real.

Corre las celdas en orden. Al final se genera un CSV piloto pequeño para revisar juntas
la calidad antes de escalar a la extraccion completa.
"""))

cells.append(code("""!pip -q install requests
"""))

cells.append(md("""## 0. Universo de datos semilla

Lista curada a mano (no proviene de Kaggle ni de ningun dataset preexistente).
"""))

cells.append(code('''SEED_BANDS = {
    "Hard Rock / Rock clasico": ["Led Zeppelin", "AC/DC", "Deep Purple"],
    "Heavy Metal": ["Iron Maiden", "Judas Priest", "Black Sabbath"],
    "Power Metal": ["Blind Guardian", "DragonForce", "Sabaton"],
    "Symphonic Metal": ["Nightwish", "Epica", "Within Temptation"],
    "Folk Metal": ["Eluveitie", "Korpiklaani", "Wintersun"],
    "Gothic Metal": ["Type O Negative", "Paradise Lost", "Lacuna Coil"],
    "Progressive Metal": ["Dream Theater", "Opeth", "Tool"],
    "Death Metal": ["Death", "Cannibal Corpse", "At the Gates"],
    "Black Metal": ["Mayhem", "Emperor", "Dimmu Borgir"],
    "Metalcore": ["Killswitch Engage", "Bring Me the Horizon", "Architects"],
    "Deathcore": ["Suicide Silence", "Whitechapel"],
    "Nu Metal": ["Slipknot", "Korn", "System of a Down"],
    "Doom Metal / Stoner": ["Electric Wizard", "Sleep", "Candlemass"],
    "Post-Metal / Atmospheric": ["Alcest", "Agalloch"],
}

SEED_BOOKS = {
    "Alta fantasia / epica": ["El Senor de los Anillos", "Cancion de Hielo y Fuego", "El Nombre del Viento"],
    "Romantasy (fantasia romantica)": ["Trono de Cristal", "Una Corte de Rosas y Espinas", "Cuarto Ala"],
    "Fantasia oscura / gotica": ["Sombra y Hueso", "Dracula", "Vicious"],
    "Romance historico / gotico": ["Outlander", "Jane Eyre"],
    "Fantasia urbana": ["Cazadores de Sombras: Ciudad de Hueso"],
    "Mitologia (nordica / celta)": ["Norse Mythology"],
}

# Para el piloto usamos solo la primera banda/libro de cada categoria
PILOT_BANDS = [v[0] for v in SEED_BANDS.values()]
PILOT_BOOKS = [v[0] for v in SEED_BOOKS.values()]
print(f"{len(PILOT_BANDS)} bandas piloto | {len(PILOT_BOOKS)} libros piloto")
'''))

cells.append(md("""## 1. MusicBrainz

Requiere un `User-Agent` personalizado y respetar el limite de ~1 request/segundo
(uso anonimo). No requiere API key.

**Nota:** es normal recibir errores `503 Service Temporarily Unavailable` de vez en
cuando — sobre todo desde Colab, donde muchas personas comparten la misma IP de salida
y eso hace que, en conjunto, se rebase el limite de tasa de MusicBrainz aunque tu propio
codigo respete su propio `sleep`. Por eso el codigo de abajo reintenta automaticamente
con backoff exponencial (espera 1s, luego 2s, 4s, 8s...) en vez de fallar de inmediato,
y sigue con la siguiente banda si una en particular no responde tras varios intentos.
"""))

cells.append(code('''import time
import requests

MB_HEADERS = {"User-Agent": "InkriffDataCollector/0.1 (contacto: tu_correo@ejemplo.com)"}
MB_BASE = "https://musicbrainz.org/ws/2"


def mb_get(url, params, max_retries=5):
    """GET a un endpoint de MusicBrainz con reintento + backoff ante 503/429.

    Estos codigos son comunes en su API publica (limite de tasa compartido entre
    todos los usuarios anonimos, mas notorio desde IPs compartidas como Colab).
    """
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
for name in PILOT_BANDS:
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
'''))

cells.append(md("""## 2. Genius (API oficial — solo metadatos, sin letras)

Necesitas un **Client Access Token** gratuito (toma ~2 minutos, sin aprobacion manual):

1. Entra a https://genius.com/api-clients y crea una cuenta/app (cualquier nombre sirve).
2. Copia el "Client Access Token" que te genera.
3. Pegalo abajo en `GENIUS_TOKEN`.

**Por que ya no pedimos letras:** se probo con `lyricsgenius` (que consigue la letra
haciendo scraping de la pagina web de Genius, ya que su API oficial no la incluye) y
Genius lo bloquea con un challenge de Cloudflare/reCAPTCHA (error 403) — no es un
limite de tasa que se resuelva con reintentos, es una barrera anti-bot real. Ademas
ese scraping iba contra los Terminos de Servicio de Genius. Por eso nos quedamos solo
con su API oficial (permitida, sin scraping) para metadatos como titulo y popularidad,
y usamos los tags/generos de MusicBrainz como señal principal de "mood" musical — esto
no afecta el objetivo del proyecto, solo simplifica de donde sale esa señal.
"""))

cells.append(code('''import requests

GENIUS_TOKEN = "PEGA_AQUI_TU_TOKEN"  # https://genius.com/api-clients
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
for name in PILOT_BANDS:  # solo metadatos: ya no hay que ir despacio como con letras
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

print(f"\\nTotal de bandas con datos de Genius: {len(genius_results)} de {len(PILOT_BANDS)}")
'''))

cells.append(md("""## 3. Open Library (con Google Books como respaldo)

Open Library no requiere API key. Google Books funciona sin key para busquedas basicas
(con cuota diaria limitada), se usa como respaldo si Open Library no trae sinopsis.

**Nota:** igual que con MusicBrainz, es normal que estas APIs tarden o no respondan de
vez en cuando (`ConnectTimeout`, `ReadTimeout`) — el codigo de abajo reintenta con
backoff automaticamente.
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
for title in PILOT_BOOKS:
    ol = openlibrary_search(title)
    row = {"titulo_buscado": title}
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
'''))

cells.append(md("""## 4. Consolidar dataset piloto

Une los 3 resultados en un CSV pequeño para revisar calidad antes de escalar.
"""))

cells.append(code('''import pandas as pd

df_bandas = pd.DataFrame(mb_results)
df_letras = pd.DataFrame(genius_results)
df_libros = pd.DataFrame(book_results)

df_bandas.to_csv("piloto_bandas_musicbrainz.csv", index=False)
df_letras.to_csv("piloto_letras_genius.csv", index=False)
df_libros.to_csv("piloto_libros.csv", index=False)

print("Bandas:")
display(df_bandas)
print("Letras (features, no texto completo):")
display(df_letras)
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

with open("inkriff_pilot_extraccion.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook generado: inkriff_pilot_extraccion.ipynb")
