# -*- coding: utf-8 -*-
"""
Control de calidad post-extraccion: la busqueda por texto libre en Genius y
Open Library a veces devuelve el "mejor" resultado equivocado cuando el
nombre buscado es una palabra generica o ambigua (p.ej. "Death", "Sleep",
"Ghost", "Emperor" como nombres de banda; "En llamas" o "Cuarto Ala" como
titulo de libro). Este script no llama a ninguna API: solo compara el texto
buscado contra el texto devuelto (similaridad de cadenas) y marca con
bandera las filas que probablemente NO correspondan a la banda/libro real,
para revisarlas a mano antes de usarlas en modelado.
"""
import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd


def normaliza(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def similitud(a, b):
    return SequenceMatcher(None, normaliza(a), normaliza(b)).ratio()


# --- 1. Genius: banda_buscada vs artista_genius ---
genius = pd.read_csv("metadatos_genius.csv")
genius["similitud_artista"] = genius.apply(
    lambda r: similitud(r["banda_buscada"], r["artista_genius"]), axis=1
)
# umbral relativamente laxo: nombres con puntuacion distinta ("Blink-182" vs
# "blink-182") o con acentos siguen dando similitud alta; lo que se busca
# atrapar es un artista TOTALMENTE distinto.
genius["revisar"] = genius["similitud_artista"] < 0.5
genius.to_csv("metadatos_genius_qa.csv", index=False)

flag_genius = genius[genius["revisar"]][
    ["banda_buscada", "titulo_top", "artista_genius", "similitud_artista"]
]

# --- 2. Libros: titulo_buscado vs titulo (Open Library / Google Books) ---
libros = pd.read_csv("libros.csv")
libros["similitud_titulo"] = libros.apply(
    lambda r: similitud(r["titulo_buscado"], r["titulo"]), axis=1
)
# aqui el umbral tiene que ser MAS laxo todavia: es normal y correcto que el
# titulo buscado este en espanol ("El Senor de los Anillos") y el devuelto en
# ingles ("The Fellowship of the Ring") - eso NO es un error. Lo que se
# atrapa es cuando el resultado no tiene absolutamente nada que ver.
libros["revisar"] = (libros["similitud_titulo"] < 0.25) | libros["titulo"].isna()
libros.to_csv("libros_qa.csv", index=False)

flag_libros = libros[libros["revisar"]][
    ["titulo_buscado", "categoria_semilla", "titulo", "autor", "similitud_titulo"]
]

print(f"Genius: {genius['revisar'].sum()} de {len(genius)} filas para revisar")
print(flag_genius.to_string(index=False))
print()
print(f"Libros: {libros['revisar'].sum()} de {len(libros)} filas para revisar")
print(flag_libros.to_string(index=False))
