# -*- coding: utf-8 -*-
"""Genera inkriff_eda_ingenieria.ipynb — EDA + ingenieria de variables sobre
los 3 CSVs de la fase de extraccion (bandas_completo.csv, libros.csv,
pares_semilla_final.csv). Incluye tambien la extraccion complementaria de
sinopsis de libros (necesaria para Zipf/nube de palabras/TF-IDF), porque la
extraccion original solo trajo titulo/autor/anio."""
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

cells.append(md("""# Inkriff — EDA e ingenieria de variables

Este notebook toma los 3 CSVs que ya generamos en la fase de extraccion:

- `bandas_completo.csv` (81 bandas: MusicBrainz + Genius)
- `libros.csv` (112 libros: Open Library / Google Books)
- `pares_semilla_final.csv` (105 pares libro<->banda curados a mano — el ancla de
  entrenamiento supervisado del dual encoder)

**Antes de correr**, sube estos 3 archivos a la sesion de Colab (icono de carpeta
en el panel izquierdo -> arrastra los archivos, o usa la celda de carga de abajo).

El notebook tiene 3 partes:

1. **EDA** sobre bandas, libros y los pares de entrenamiento (distribuciones,
   control de calidad, tags mas frecuentes).
2. **Extraccion complementaria de sinopsis** — la extraccion original solo trajo
   titulo/autor/año de cada libro, pero para Zipf, nube de palabras y TF-IDF
   necesitamos el TEXTO de la sinopsis. Esta seccion lo agrega (Google Books trae
   la sinopsis en la misma respuesta; Open Library necesita una llamada extra).
3. **Ingenieria de variables**: tags de MusicBrainz -> vector multi-hot,
   subgenero/categoria -> one-hot, sinopsis -> TF-IDF, y el merge final que arma
   la tabla de pares lista para el dual encoder (`pares_features.csv`).

Corre las celdas en orden.
"""))

cells.append(code("""!pip -q install scikit-learn wordcloud
"""))

cells.append(code('''import ast
import re
import unicodedata
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_colwidth", 80)

# Paleta simple y consistente para todo el notebook (evita rojo/verde puro,
# que es dificil de distinguir para personas con daltonismo).
PALETA = ["#4C6EF5", "#F76707", "#12B886", "#BE4BDB", "#FAB005", "#1098AD", "#E64980"]
plt.rcParams["figure.dpi"] = 100
'''))

cells.append(md("""## 0. Cargar los datos
"""))

cells.append(code('''try:
    from google.colab import files
    print("Sube bandas_completo.csv, libros.csv y pares_semilla_final.csv:")
    files.upload()
except ImportError:
    pass  # no estamos en Colab; se asume que los CSVs ya estan en el directorio actual

bandas = pd.read_csv("bandas_completo.csv")
libros = pd.read_csv("libros.csv")
pares = pd.read_csv("pares_semilla_final.csv")

print(f"bandas: {bandas.shape} | libros: {libros.shape} | pares: {pares.shape}")
bandas.head()
'''))

cells.append(md("""## 1. EDA — Bandas
"""))

cells.append(code('''conteo_subgenero = bandas["subgenero_semilla"].value_counts()

fig, ax = plt.subplots(figsize=(8, 8))
conteo_subgenero.sort_values().plot(kind="barh", ax=ax, color=PALETA[0])
ax.set_title("Bandas por subgenero semilla")
ax.set_xlabel("Numero de bandas")
plt.tight_layout()
plt.show()

print(f"{len(conteo_subgenero)} subgeneros | {conteo_subgenero.sum()} bandas en total")
'''))

cells.append(md("""**Nota:** el universo se amplio a 24 subgeneros (los 14 originales + 10 que
aportaron los pares de Viry: Pop Punk, Alternative Metal, Alternative Rock,
Post-Hardcore, Gothic Rock, Occult Rock, Punk Rock, Melodic Death Metal,
Blackened Death Metal, Emo/Alternative Rock). Es normal que varios subgeneros
nuevos tengan pocas bandas (2-9): son categorias reales, solo que el pilar del
proyecto sigue siendo la escena metal/rock clasica, mas grande desde el inicio.
"""))

cells.append(code('''fig, ax = plt.subplots(figsize=(8, 4))
bandas["pais"].value_counts().head(15).plot(kind="bar", ax=ax, color=PALETA[1])
ax.set_title("Pais de origen de las bandas (top 15)")
ax.set_ylabel("Numero de bandas")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
'''))

cells.append(md("""### Tags de MusicBrainz — la señal de "mood" musical

Como ya no usamos letras (Genius bloquea el scraping con Cloudflare — ver la
decision documentada en el canvas del proyecto), los tags de MusicBrainz son la
UNICA señal de texto/mood del lado de la musica. Vale la pena verlos de cerca.
"""))

cells.append(code('''bandas["tags_lista"] = bandas["tags_top"].dropna().apply(ast.literal_eval)
bandas["tags_lista"] = bandas["tags_lista"].apply(lambda x: x if isinstance(x, list) else [])

conteo_tags = Counter(t for lst in bandas["tags_lista"] for t in lst)
top_tags = conteo_tags.most_common(25)

fig, ax = plt.subplots(figsize=(8, 8))
etiquetas, valores = zip(*top_tags[::-1])
ax.barh(etiquetas, valores, color=PALETA[2])
ax.set_title("Tags de MusicBrainz mas frecuentes (top 25 de 81 bandas)")
ax.set_xlabel("Numero de bandas con ese tag")
plt.tight_layout()
plt.show()

print(f"{len(conteo_tags)} tags distintos en total sobre {len(bandas)} bandas")
'''))

cells.append(code('''# Vista rapida de Ley de Zipf sobre los tags (rango vs frecuencia, log-log).
# Es una version preliminar: la version real (sobre texto libre) se hace mas
# abajo con las sinopsis de los libros, que tienen muchas mas palabras unicas.
frecuencias = sorted(conteo_tags.values(), reverse=True)
rangos = range(1, len(frecuencias) + 1)

fig, ax = plt.subplots(figsize=(6, 5))
ax.loglog(rangos, frecuencias, marker="o", linestyle="none", color=PALETA[3], alpha=0.7)
ax.set_xlabel("Rango (log)")
ax.set_ylabel("Frecuencia (log)")
ax.set_title("Tags de MusicBrainz: rango vs. frecuencia")
plt.tight_layout()
plt.show()
'''))

cells.append(md("""### Control de calidad — Genius

La busqueda de texto libre en Genius a veces regresa la cancion mas popular con
ese texto en vez de la banda correcta, sobre todo cuando el nombre de la banda es
una palabra generica ("Death", "Sleep", "Ghost", "Emperor", "The Cure"...). Antes
de usar `pageviews`/`titulo_top` como feature, marcamos con una bandera las filas
donde el artista devuelto no se parece al nombre buscado (similaridad de cadenas).
"""))

cells.append(code('''def normaliza(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\\s+", " ", s).strip()


from difflib import SequenceMatcher

def similitud(a, b):
    return SequenceMatcher(None, normaliza(a), normaliza(b)).ratio()


bandas["similitud_artista_genius"] = bandas.apply(
    lambda r: similitud(r["banda"], r["artista_genius"]), axis=1
)
bandas["genius_confiable"] = bandas["similitud_artista_genius"] >= 0.5

print(f"{(~bandas['genius_confiable']).sum()} de {len(bandas)} bandas con match de Genius poco confiable:")
bandas.loc[~bandas["genius_confiable"], ["banda", "titulo_top", "artista_genius", "similitud_artista_genius"]]
'''))

cells.append(code('''# Distribucion de popularidad (pageviews de Genius), solo donde el match es confiable
pageviews_ok = bandas.loc[bandas["genius_confiable"], "pageviews"].dropna()

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(np.log10(pageviews_ok + 1), bins=20, color=PALETA[4], edgecolor="white")
ax.set_title("Popularidad en Genius (log10 pageviews), solo matches confiables")
ax.set_xlabel("log10(pageviews + 1)")
ax.set_ylabel("Numero de bandas")
plt.tight_layout()
plt.show()
'''))

cells.append(md("""## 2. EDA — Libros
"""))

cells.append(code('''conteo_categoria = libros["categoria_semilla"].value_counts()

fig, ax = plt.subplots(figsize=(8, 8))
conteo_categoria.sort_values().plot(kind="barh", ax=ax, color=PALETA[0])
ax.set_title("Libros por categoria semilla")
ax.set_xlabel("Numero de libros")
plt.tight_layout()
plt.show()

print(f"{len(conteo_categoria)} categorias | {conteo_categoria.sum()} libros en total")
print(f"Sin match en Open Library/Google Books: {libros['titulo'].isna().sum()} libro(s)")
'''))

cells.append(code('''libros["anio_num"] = pd.to_numeric(libros["anio"], errors="coerce")
libros["decada"] = (libros["anio_num"] // 10 * 10)

fig, ax = plt.subplots(figsize=(8, 4))
libros["decada"].value_counts().sort_index().plot(kind="bar", ax=ax, color=PALETA[5])
ax.set_title("Libros por decada de publicacion")
ax.set_ylabel("Numero de libros")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
'''))

cells.append(md("""## 3. EDA — Pares de entrenamiento (ancla del dual encoder)
"""))

cells.append(code('''pares["banda"] = pares["banda_subgenero"].str.split(" — ").str[0]
pares["banda_subg_texto"] = pares["banda_subgenero"].str.split(" — ").str[1]
pares["libro_titulo"] = pares["libro"].str.split(" — ").str[0]

combo = (pares["categoria_libro"] + " <-> " + pares["banda_subg_texto"]).value_counts().head(15)

fig, ax = plt.subplots(figsize=(8, 6))
combo.sort_values().plot(kind="barh", ax=ax, color=PALETA[6])
ax.set_title("Combinaciones categoria de libro <-> subgenero de banda mas repetidas")
ax.set_xlabel("Numero de pares")
plt.tight_layout()
plt.show()

print(f"{pares[\'categoria_libro\'].nunique()} categorias de libro distintas en los pares")
print(f"{pares[\'banda_subg_texto\'].nunique()} subgeneros de banda distintos en los pares (etiqueta libre de Viry)")
'''))

cells.append(md("""## 4. Extraccion complementaria — sinopsis de libros

La extraccion original (`inkriff_extraccion_completa.ipynb`) solo trajo
titulo/autor/año de cada libro. Para Zipf, nube de palabras y TF-IDF necesitamos
el TEXTO de la sinopsis. Google Books la trae directo en la misma busqueda; Open
Library requiere una llamada adicional al detalle de la obra (`/works/OLxxxxW.json`).

Mismo patron de reintentos que ya conocemos (503/429 y timeouts).
"""))

cells.append(code('''import time
import requests


def get_con_reintento(url, params, max_retries=4, timeout=20):
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
    return None


def sinopsis_googlebooks(title):
    data = get_con_reintento("https://www.googleapis.com/books/v1/volumes", {"q": title, "maxResults": 1})
    if not data:
        return None
    items = data.get("items", [])
    if not items:
        return None
    return items[0].get("volumeInfo", {}).get("description")


def sinopsis_openlibrary(title):
    """Busca la obra y, si la encuentra, pide el detalle para sacar la description."""
    data = get_con_reintento("https://openlibrary.org/search.json", {"q": title, "limit": 1})
    if not data or not data.get("docs"):
        return None
    key = data["docs"][0].get("key")  # p.ej. "/works/OL27448W"
    if not key:
        return None
    detail = get_con_reintento(f"https://openlibrary.org{key}.json", {})
    if not detail:
        return None
    desc = detail.get("description")
    if isinstance(desc, dict):
        desc = desc.get("value")
    return desc


sinopsis_resultados = []
for _, row in libros.iterrows():
    titulo = row["titulo_buscado"]
    sinopsis = sinopsis_googlebooks(titulo)
    fuente_sinopsis = "Google Books"
    if not sinopsis:
        sinopsis = sinopsis_openlibrary(titulo)
        fuente_sinopsis = "Open Library" if sinopsis else None
    sinopsis_resultados.append({"titulo_buscado": titulo, "sinopsis": sinopsis, "fuente_sinopsis": fuente_sinopsis})
    print(f"{titulo}: {\'OK (\' + fuente_sinopsis + \')\' if sinopsis else \'[!] sin sinopsis\'}")
    time.sleep(0.3)

df_sinopsis = pd.DataFrame(sinopsis_resultados)
libros = libros.merge(df_sinopsis, on="titulo_buscado", how="left")
libros.to_csv("libros_con_sinopsis.csv", index=False)

print(f"\\n{libros[\'sinopsis\'].notna().sum()} de {len(libros)} libros con sinopsis")
'''))

cells.append(md("""## 5. Limpieza de texto y Ley de Zipf sobre las sinopsis

Las sinopsis vienen mezcladas en español e ingles (segun el libro se haya
encontrado en Open Library o Google Books en un idioma u otro), asi que la lista
de "stopwords" cubre ambos idiomas.
"""))

cells.append(code('''from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Las sinopsis de Open Library a veces traen notas del editor pegadas al texto
# (URLs, "([Source][3])", creditos de edicion/bestseller) -- eso se limpia por
# separado antes de tokenizar. Para las stopwords: la lista en espanol es
# manual, pero para ingles se usa la lista completa de sklearn (no una lista
# corta a mano) para que "but/has/one/will/into..." no salgan como las
# palabras "mas frecuentes" del corpus.
STOPWORDS_ES = {
    "el","la","los","las","de","del","y","a","en","un","una","unos","unas","que",
    "es","por","con","para","su","sus","se","lo","como","mas","pero","o","al",
    "este","esta","estos","estas","ese","esa","esos","esas","ya","muy","entre",
    "sin","sobre","tambien","cuando","donde","porque","desde","hasta","hay",
    "fue","son","ser","han","habia","era","eran","le","les","nos","me","mi",
    "tu","si","no","asi","solo","cada","otro","otra","otros","otras","todo",
    "toda","todos","todas","tras","durante","mientras","aunque","ademas",
}
STOPWORDS = STOPWORDS_ES | set(ENGLISH_STOP_WORDS)

RUIDO_EDITORIAL = {
    "https","http","www","com","org","net","source","openlibrary","googlebooks",
    "google","books","publisher","published","publication","edition","editions",
    "excerpt","isbn","copyright","cover","flap","bestseller","bestselling",
    "york","times","newyork","nytimes","author","authors","press","description",
    "special","note","content",
}


def limpia_texto(t):
    if not isinstance(t, str):
        return []
    t = re.sub(r"https?://\\S+", " ", t)          # URLs completas
    t = re.sub(r"\\[\\d+\\]:?", " ", t)              # referencias tipo markdown "[1]:"
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^a-z ]", " ", t.lower())
    palabras = [
        w for w in t.split()
        if len(w) > 2 and w not in STOPWORDS and w not in RUIDO_EDITORIAL
    ]
    return palabras


libros["sinopsis_tokens"] = libros["sinopsis"].apply(limpia_texto)
libros["sinopsis_limpia"] = libros["sinopsis_tokens"].apply(lambda ws: " ".join(ws))

todas_palabras = Counter(w for tokens in libros["sinopsis_tokens"] for w in tokens)
print(f"{len(todas_palabras)} palabras unicas sobre {libros[\'sinopsis\'].notna().sum()} sinopsis")
todas_palabras.most_common(20)
'''))

cells.append(code('''frecuencias = sorted(todas_palabras.values(), reverse=True)
rangos = range(1, len(frecuencias) + 1)

fig, ax = plt.subplots(figsize=(6, 5))
ax.loglog(rangos, frecuencias, marker="o", linestyle="none", color=PALETA[0], alpha=0.5, markersize=3)
ax.set_xlabel("Rango (log)")
ax.set_ylabel("Frecuencia (log)")
ax.set_title("Ley de Zipf — palabras en las sinopsis de los libros")
plt.tight_layout()
plt.show()
'''))

cells.append(code('''from wordcloud import WordCloud

texto_completo = " ".join(libros["sinopsis_limpia"].dropna())
if texto_completo.strip():
    nube = WordCloud(width=900, height=500, background_color="white",
                      colormap="viridis", max_words=100).generate(texto_completo)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(nube, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Nube de palabras — sinopsis de los libros")
    plt.tight_layout()
    plt.show()
else:
    print("[!] No hay texto de sinopsis para generar la nube de palabras "
          "(revisa la celda de extraccion de sinopsis mas arriba).")
'''))

cells.append(md("""## 6. Ingenieria de variables

Tres piezas:

1. **Bandas**: tags de MusicBrainz -> vector multi-hot (top 30 tags), subgenero ->
   one-hot, popularidad -> `log1p(pageviews)` (NaN si el match de Genius no es
   confiable).
2. **Libros**: sinopsis limpia -> TF-IDF (300 features), categoria -> one-hot,
   año -> decada.
3. **Pares**: merge de las dos tablas anteriores sobre `pares_semilla_final.csv`
   para armar la tabla final de entrenamiento del dual encoder.
"""))

cells.append(code('''from sklearn.preprocessing import MultiLabelBinarizer

top_tags = [t for t, _ in conteo_tags.most_common(30)]
bandas["tags_filtrados"] = bandas["tags_lista"].apply(lambda lst: [t for t in lst if t in top_tags])

mlb = MultiLabelBinarizer(classes=top_tags)
tag_matrix = mlb.fit_transform(bandas["tags_filtrados"])
tags_df = pd.DataFrame(tag_matrix, columns=[f"tag_{t.replace(\' \', \'_\')}" for t in top_tags])

subgenero_dummies = pd.get_dummies(bandas["subgenero_semilla"], prefix="subg")

bandas["popularidad_log"] = np.where(
    bandas["genius_confiable"], np.log1p(bandas["pageviews"]), np.nan
)

bandas_features = pd.concat(
    [bandas[["banda", "subgenero_semilla", "pais", "popularidad_log"]].reset_index(drop=True),
     subgenero_dummies.reset_index(drop=True),
     tags_df.reset_index(drop=True)],
    axis=1,
)
bandas_features.to_csv("bandas_features.csv", index=False)
print(f"bandas_features: {bandas_features.shape}")
bandas_features.head()
'''))

cells.append(code('''from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=300, min_df=2)
tiene_sinopsis = libros["sinopsis_limpia"].fillna("").str.len() > 0
if tiene_sinopsis.sum() >= 2:
    tfidf_matrix = tfidf.fit_transform(libros.loc[tiene_sinopsis, "sinopsis_limpia"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()],
        index=libros.loc[tiene_sinopsis].index,
    )
else:
    print("[!] Muy pocos libros con sinopsis para calcular TF-IDF; "
          "libros_features solo tendra categoria/decada.")
    tfidf_df = pd.DataFrame(index=libros.index)

categoria_dummies = pd.get_dummies(libros["categoria_semilla"], prefix="cat")

libros_features = pd.concat(
    [libros[["titulo_buscado", "categoria_semilla", "decada"]], categoria_dummies], axis=1
).join(tfidf_df)
libros_features.to_csv("libros_features.csv", index=False)
print(f"libros_features: {libros_features.shape} ({tiene_sinopsis.sum()} libros con TF-IDF, resto solo categoria/decada)")
libros_features.head()
'''))

cells.append(code('''def normaliza_llave(s):
    return normaliza(s)  # mismo normalizador de mas arriba, reusado como llave de merge

pares["banda_key"] = pares["banda"].apply(normaliza_llave)
bandas_features["banda_key"] = bandas_features["banda"].apply(normaliza_llave)

pares["libro_key"] = pares["libro_titulo"].apply(normaliza_llave)
libros_features["libro_key"] = libros_features["titulo_buscado"].apply(normaliza_llave)

pares_features = (
    pares
    .merge(bandas_features, on="banda_key", how="left", suffixes=("", "_banda"))
    .merge(libros_features, on="libro_key", how="left", suffixes=("", "_libro"))
)

faltantes_banda = pares_features["subgenero_semilla"].isna().sum()
faltantes_libro = pares_features["categoria_semilla"].isna().sum()
print(f"Pares sin features de banda: {faltantes_banda} | sin features de libro: {faltantes_libro}")

pares_features.to_csv("pares_features.csv", index=False)
print(f"pares_features: {pares_features.shape}")
pares_features.head()
'''))

cells.append(md("""## 7. Resumen

Archivos generados en esta sesion:

- `libros_con_sinopsis.csv` — los 112 libros con su sinopsis (cuando se encontro).
- `bandas_features.csv` — 81 bandas con tags multi-hot + subgenero one-hot + popularidad.
- `libros_features.csv` — 112 libros con TF-IDF de la sinopsis + categoria one-hot + decada.
- `pares_features.csv` — los 105 pares de entrenamiento, ya con las features de banda
  y de libro pegadas — listo para la fase de modelado (dual encoder).

Descarga estos 4 archivos (panel de archivos de Colab, boton derecho -> Download)
para la siguiente fase.
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

with open("inkriff_eda_ingenieria.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook generado: inkriff_eda_ingenieria.ipynb")
