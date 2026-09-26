# -*- coding: utf-8 -*-
"""
Reprocesa libros_con_sinopsis.csv con una limpieza de texto mejorada (quita URLs,
referencias tipo markdown "[1]:", y una lista de "ruido editorial" -- ISBN,
"source", "edition", "publisher", "bestselling", nombres de dominio, etc. -- que
se colaba en el TF-IDF porque las sinopsis de Open Library a veces traen notas
del editor / links de fuente pegados al texto). No vuelve a llamar ninguna API:
parte de libros_con_sinopsis.csv que Viry ya genero y subio.

Regenera libros_features.csv y pares_features.csv (bandas_features.csv no
cambia, se reusa el que Viry ya genero).
"""
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# La primera version solo tenia ~15 stopwords en ingles a mano, lo que dejaba
# pasar puro "glue" (but, has, one, will, into, are, when, who...) como si
# fueran las palabras mas frecuentes del corpus -- inutil para Zipf/TF-IDF.
# Ahora se usa la lista completa de sklearn (~320 palabras) + una lista en
# español un poco mas larga que la original.
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

# "Ruido editorial": vocabulario que no describe la trama, viene de notas de la
# fuente (Open Library / Google Books) pegadas al final de la sinopsis: URLs,
# creditos de editorial, menciones de premios/bestseller, metadatos de edicion.
RUIDO_EDITORIAL = {
    "https","http","www","com","org","net","source","openlibrary","googlebooks",
    "google","books","publisher","published","publication","edition","editions",
    "excerpt","isbn","copyright","cover","flap","bestseller","bestselling",
    "york","times","newyork","nytimes","author","authors","press","description",
    "special","note","content","special note",
}


def normaliza(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def limpia_texto_v2(t):
    if not isinstance(t, str):
        return []
    # 1. quitar URLs completas antes de tokenizar
    t = re.sub(r"https?://\S+", " ", t)
    # 2. quitar referencias estilo markdown "[1]:" o "[3]"
    t = re.sub(r"\[\d+\]:?", " ", t)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^a-z ]", " ", t.lower())
    palabras = [
        w for w in t.split()
        if len(w) > 2 and w not in STOPWORDS and w not in RUIDO_EDITORIAL
    ]
    return palabras


libros = pd.read_csv("libros_con_sinopsis.csv")
libros["sinopsis_tokens"] = libros["sinopsis"].apply(limpia_texto_v2)
libros["sinopsis_limpia"] = libros["sinopsis_tokens"].apply(lambda ws: " ".join(ws))

todas_palabras = Counter(w for tokens in libros["sinopsis_tokens"] for w in tokens)
print(f"{len(todas_palabras)} palabras unicas sobre {libros['sinopsis'].notna().sum()} sinopsis (v2, con limpieza de ruido editorial)")
print("Top 20:", todas_palabras.most_common(20))

# --- TF-IDF (igual que antes, ahora sobre texto limpio v2) ---
tfidf = TfidfVectorizer(max_features=300, min_df=2)
tiene_sinopsis = libros["sinopsis_limpia"].fillna("").str.len() > 0
tfidf_matrix = tfidf.fit_transform(libros.loc[tiene_sinopsis, "sinopsis_limpia"])
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()],
    index=libros.loc[tiene_sinopsis].index,
)

# chequeo: ya no deberia haber ruido editorial en el vocabulario del TF-IDF
vocab = set(tfidf.get_feature_names_out())
interseccion = vocab & RUIDO_EDITORIAL
print(f"Palabras de ruido editorial que SIGUEN en el vocabulario TF-IDF: {interseccion or 'ninguna -- limpio'}")

categoria_dummies = pd.get_dummies(libros["categoria_semilla"], prefix="cat")
libros_features = pd.concat(
    [libros[["titulo_buscado", "categoria_semilla", "decada"]], categoria_dummies], axis=1
).join(tfidf_df)
libros_features.to_csv("libros_features.csv", index=False)
print(f"libros_features (v2): {libros_features.shape}")

# --- merge final con pares_semilla_final.csv y bandas_features.csv ---
bandas_features = pd.read_csv("bandas_features.csv")
pares = pd.read_csv("pares_semilla_final.csv")

pares["banda"] = pares["banda_subgenero"].str.split(" — ").str[0]
pares["banda_subg_texto"] = pares["banda_subgenero"].str.split(" — ").str[1]
pares["libro_titulo"] = pares["libro"].str.split(" — ").str[0]

pares["banda_key"] = pares["banda"].apply(normaliza)
bandas_features["banda_key"] = bandas_features["banda"].apply(normaliza)
pares["libro_key"] = pares["libro_titulo"].apply(normaliza)
libros_features["libro_key"] = libros_features["titulo_buscado"].apply(normaliza)

pares_features = (
    pares
    .merge(bandas_features, on="banda_key", how="left", suffixes=("", "_banda"))
    .merge(libros_features, on="libro_key", how="left", suffixes=("", "_libro"))
)

faltantes_banda = pares_features["subgenero_semilla"].isna().sum()
faltantes_libro = pares_features["categoria_semilla"].isna().sum()
print(f"Pares sin features de banda: {faltantes_banda} | sin features de libro: {faltantes_libro}")

pares_features.to_csv("pares_features.csv", index=False)
print(f"pares_features (v2): {pares_features.shape}")
