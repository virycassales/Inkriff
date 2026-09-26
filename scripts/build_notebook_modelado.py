# -*- coding: utf-8 -*-
"""Genera inkriff_modelado_similitud.ipynb — recomendador bidireccional libro<->banda
basado en similitud de contenido (sin entrenar una red desde cero, dado que solo
tenemos 105 pares etiquetados a mano). Compara dos enfoques: TF-IDF (baseline,
ligero, se puede correr en cualquier lado) y embeddings de oracion pre-entrenados
multilingues (necesita descargar un modelo -> requiere Colab, este sandbox no
tiene salida a huggingface.co)."""
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

cells.append(md("""# Inkriff — Modelado: recomendador bidireccional por similitud de contenido

Decisión de diseño (ya la platicamos): con solo **105 pares** etiquetados a mano no
tiene sentido entrenar una red desde cero — el riesgo de que memorice esos 105
pares en vez de aprender un patrón generalizable es alto. En vez de eso, este
notebook construye un **recomendador por similitud de contenido**: represento cada
libro y cada banda como un vector en un mismo espacio semántico, mido similitud
coseno, y uso los 105 pares **solo para evaluar** qué tan bien funciona (no para
entrenar).

Se prueban y comparan dos enfoques, de forma honesta — el primero no funciona muy
bien, y eso en sí mismo es un resultado que vale la pena documentar:

1. **TF-IDF** (bag-of-words) — rápido, no necesita descargar nada, pero solo
   detecta coincidencias de palabras EXACTAS. El problema: las palabras que
   describen una banda (tags de género musical: "power metal", "gothic") casi
   nunca aparecen literalmente en la sinopsis de un libro. Se prueba de todos
   modos como punto de referencia.
2. **Embeddings de oración pre-entrenados** (`sentence-transformers`, modelo
   multilingüe) — capturan similitud *semántica*, no solo léxica: entienden que
   "power metal, dragones, batallas medievales" se relaciona con "guerra, reino,
   espada, antiguo" aunque no compartan ni una palabra. Este paso **necesita
   descargar un modelo (~470 MB) de Hugging Face, así que solo corre en Colab**
   (el entorno donde arme este notebook no tiene salida a internet hacia ahí).

**Métricas de evaluación** (sobre los 105 pares conocidos): para cada par
libro-banda, ¿en qué lugar queda la banda correcta si ordeno todas las 81 bandas
por similitud con ese libro? (y viceversa). Se reporta el rank promedio (comparado
contra lo que daría el azar), Recall@5, Recall@10 y MRR (mean reciprocal rank).

Sube `bandas_completo.csv`, `libros_con_sinopsis.csv` y `pares_semilla_final.csv`
antes de correr.
"""))

cells.append(code("""!pip -q install scikit-learn sentence-transformers
"""))

cells.append(code('''import ast
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from google.colab import files
    print("Sube bandas_completo.csv, libros_con_sinopsis.csv y pares_semilla_final.csv:")
    files.upload()
except ImportError:
    pass

bandas = pd.read_csv("bandas_completo.csv")
libros = pd.read_csv("libros_con_sinopsis.csv")
pares = pd.read_csv("pares_semilla_final.csv")
print(f"bandas: {bandas.shape} | libros: {libros.shape} | pares: {pares.shape}")
'''))

cells.append(md("""## 0. Preparar el texto de cada lado

Mismo limpiador de texto que en el notebook de EDA (quita URLs/referencias de
Open Library y ruido editorial antes de tokenizar).
"""))

cells.append(code('''STOPWORDS_ES = {
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
    return re.sub(r"\\s+", " ", s).strip()


def limpia_texto(t):
    if not isinstance(t, str):
        return ""
    t = re.sub(r"https?://\\S+", " ", t)
    t = re.sub(r"\\[\\d+\\]:?", " ", t)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^a-z ]", " ", t.lower())
    palabras = [w for w in t.split() if len(w) > 2 and w not in STOPWORDS and w not in RUIDO_EDITORIAL]
    return " ".join(palabras)


bandas["tags_lista"] = bandas["tags_top"].dropna().apply(ast.literal_eval)
bandas["tags_lista"] = bandas["tags_lista"].apply(lambda x: x if isinstance(x, list) else [])

# Perfil de banda: subgenero (repetido, le da peso) + sus tags propios de MusicBrainz.
# A diferencia de agrupar solo por subgenero, aqui cada banda conserva SUS tags
# individuales -- si dos bandas comparten subgenero pero una tiene ademas el tag
# "folk" y otra "industrial", sus perfiles ya no son identicos.
bandas["perfil_texto"] = bandas.apply(
    lambda r: ((r["subgenero_semilla"] + " ") * 3) + " ".join(r["tags_lista"]), axis=1
)
bandas["perfil_texto"] = bandas["perfil_texto"].apply(limpia_texto)

libros["perfil_texto"] = libros.apply(
    lambda r: ((str(r["categoria_semilla"]) + " ") * 2) + limpia_texto(str(r["sinopsis"])), axis=1
)

print(bandas[["banda", "perfil_texto"]].head(3))
print(libros[["titulo_buscado", "perfil_texto"]].head(3))
'''))

cells.append(md("""## 0.1 Detectar libros con sinopsis vacía (perfiles degenerados)

Si un libro no tiene sinopsis, su `perfil_texto` queda como solo la categoría repetida
dos veces — un texto genérico y muy corto. En la práctica esto hace que ese libro
termine pareciéndose "un poco a todo" en el espacio de embeddings (es el fenómeno de
"hubness": un vector sin información específica cae cerca del centro del espacio
semántico y sale barato en similitud contra cualquier cosa). Estos libros se quedan en
la evaluación de métricas de abajo (no quiero inflar los números quitando casos
difíciles), pero se excluyen como candidatos en el recomendador final (sección 5) —
recomendar un libro basándome en cero información real de su trama no tiene sentido.
"""))

cells.append(code('''libros["tiene_sinopsis"] = libros["sinopsis"].fillna("").str.strip().str.len() > 0
n_sin_sinopsis = int((~libros["tiene_sinopsis"]).sum())
print(f"Libros sin sinopsis (perfil degenerado, solo la categoria repetida): {n_sin_sinopsis} de {len(libros)}")
if n_sin_sinopsis:
    print(libros.loc[~libros["tiene_sinopsis"], "titulo_buscado"].tolist())
'''))

cells.append(md("""## 1. Preparar la evaluación (rank, Recall@k, MRR)

Esta función se reusa para los dos enfoques, para que la comparación sea justa.
"""))

cells.append(code('''pares["banda"] = pares["banda_subgenero"].str.split(" — ").str[0]
pares["libro_titulo"] = pares["libro"].str.split(" — ").str[0]
pares["banda_key"] = pares["banda"].apply(normaliza)
pares["libro_key"] = pares["libro_titulo"].apply(normaliza)
bandas["banda_key"] = bandas["banda"].apply(normaliza)
libros["libro_key"] = libros["titulo_buscado"].apply(normaliza)

bk2i = dict(zip(bandas["banda_key"], range(len(bandas))))
lk2i = dict(zip(libros["libro_key"], range(len(libros))))


resultados = {}  # nombre_enfoque -> dict de metricas, para la tabla comparativa de la seccion 4


def evalua(sim, nombre):
    """sim: matriz (n_libros x n_bandas) de similitud. Imprime un resumen,
    guarda las metricas en `resultados` y regresa los ranks crudos por si se
    quieren graficar despues."""
    ranks_banda, ranks_libro = [], []
    for _, row in pares.iterrows():
        bi, li = bk2i.get(row["banda_key"]), lk2i.get(row["libro_key"])
        if bi is None or li is None:
            continue
        ranks_banda.append(int(np.where(np.argsort(-sim[li]) == bi)[0][0]) + 1)
        ranks_libro.append(int(np.where(np.argsort(-sim[:, bi]) == li)[0][0]) + 1)
    ranks_banda, ranks_libro = np.array(ranks_banda), np.array(ranks_libro)
    n_b, n_l = sim.shape[1], sim.shape[0]
    metricas = {
        "rank_prom_banda": ranks_banda.mean(), "azar_banda": n_b / 2,
        "recall5_banda": (ranks_banda <= 5).mean(), "recall10_banda": (ranks_banda <= 10).mean(),
        "mrr_banda": np.mean(1 / ranks_banda),
        "rank_prom_libro": ranks_libro.mean(), "azar_libro": n_l / 2,
        "recall5_libro": (ranks_libro <= 5).mean(), "recall10_libro": (ranks_libro <= 10).mean(),
        "mrr_libro": np.mean(1 / ranks_libro),
    }
    resultados[nombre] = metricas
    print(f"--- {nombre} ---")
    print(f"  Banda correcta -> rank promedio {metricas[\'rank_prom_banda\']:.1f} "
          f"(azar: {metricas[\'azar_banda\']:.1f}) | Recall@5 {metricas[\'recall5_banda\']:.1%} "
          f"| Recall@10 {metricas[\'recall10_banda\']:.1%} | MRR {metricas[\'mrr_banda\']:.3f}")
    print(f"  Libro correcto -> rank promedio {metricas[\'rank_prom_libro\']:.1f} "
          f"(azar: {metricas[\'azar_libro\']:.1f}) | Recall@5 {metricas[\'recall5_libro\']:.1%} "
          f"| Recall@10 {metricas[\'recall10_libro\']:.1%} | MRR {metricas[\'mrr_libro\']:.3f}")
    return ranks_banda, ranks_libro
'''))

cells.append(md("""## 2. Enfoque 1 — TF-IDF (baseline)
"""))

cells.append(code('''from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = pd.concat([bandas["perfil_texto"], libros["perfil_texto"]])
tfidf = TfidfVectorizer(max_features=500, min_df=2)
tfidf.fit(corpus)

X_bandas_tfidf = tfidf.transform(bandas["perfil_texto"])
X_libros_tfidf = tfidf.transform(libros["perfil_texto"])
sim_tfidf = cosine_similarity(X_libros_tfidf, X_bandas_tfidf)  # (n_libros, n_bandas)

_ = evalua(sim_tfidf, "TF-IDF")
'''))

cells.append(md("""**Resultado esperado:** apenas mejor que el azar (o parecido). Esto NO es un
error del código — es el resultado real de que el vocabulario de tags musicales
("power metal", "gothic", "djent") casi nunca coincide literalmente con las
palabras de una sinopsis de libro ("guerra", "reino", "espada"). TF-IDF solo
puede ver coincidencias EXACTAS de palabras, y aquí casi no las hay. Por eso
necesitamos un enfoque que entienda *significado*, no solo texto literal.
"""))

cells.append(md("""## 3. Enfoque 2 — Embeddings de oración pre-entrenados (multilingüe)

`sentence-transformers` convierte cualquier texto en un vector que captura su
significado — libros están en inglés/español y las descripciones de subgénero
también, así que uso un modelo **multilingüe** para que ambos lados queden en el
mismo espacio semántico sin importar el idioma. No se entrena nada: el modelo ya
viene pre-entrenado, solo lo usamos para codificar texto (por eso sigue
cumpliendo la idea de "sin entrenar una red desde cero").
"""))

cells.append(code('''from sentence_transformers import SentenceTransformer

modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

emb_bandas = modelo.encode(bandas["perfil_texto"].tolist(), show_progress_bar=True, normalize_embeddings=True)
emb_libros = modelo.encode(libros["perfil_texto"].tolist(), show_progress_bar=True, normalize_embeddings=True)

sim_embed = emb_libros @ emb_bandas.T  # coseno, ya que los embeddings estan normalizados
print(sim_embed.shape)

_ = evalua(sim_embed, "Embeddings multilingues")
'''))

cells.append(md("""## 3.1 Problema detectado: "hubs" — unos pocos libros dominan todas las recomendaciones

Al revisar a fondo las recomendaciones reales (banda → libro) de esta corrida, salió un
patrón preocupante: unos pocos libros se repetían en el top-5 de una fracción enorme de
bandas que no tienen nada que ver entre sí musicalmente. En una corrida real llegamos a
ver esto:

- "El Señor de los Anillos" en el top-5 de 41 de 81 bandas (50.6%)
- "La guerra de la amapola" en 34 de 81 (42.0%)
- "Trono de Cristal" en 29.6%, "Eleanor & Park" en 23.5%

Dos causas distintas, ambas reales:

1. **Perfiles degenerados**: algunos de esos libros ("La guerra de la amapola", "El
   priorato del naranjo", "Rey de cicatrices") tienen sinopsis vacía → su perfil de
   texto es solo la categoría repetida (ver 0.1). Ya quedan excluidos como candidatos
   en la sección 5.
2. **"Hubness" genuina**: otros ("El Señor de los Anillos", "Trono de Cristal",
   "Eleanor & Park") sí tienen sinopsis larga y real, y AÚN ASÍ son hubs. Este es un
   fenómeno bien documentado en espacios de embeddings de alta dimensión: unos pocos
   puntos terminan siendo "vecinos cercanos" de casi todo, sobre todo cuando los dos
   lados del espacio tienen textos de longitud muy distinta (perfiles de banda cortos,
   ~10-20 palabras, contra sinopsis de libro largas, ~100-300 palabras).

**Corrección aplicada**: una versión simplificada de "centrado" (parecida en espíritu a
CSLS, una técnica usada en matching cross-lingual de embeddings). A cada similitud le
resto el promedio de similitud de ESE libro contra TODAS las bandas, y el promedio de
ESA banda contra TODOS los libros. Un libro que en promedio "le cae bien a todo el
mundo" (promedio alto por fila) pierde puntos — ya no puede ganar solo por ser
genérico, tiene que superar su propio promedio para aparecer en un top-5.
"""))

cells.append(code('''media_por_libro = sim_embed.mean(axis=1, keepdims=True)   # que tanto "le cae bien" este libro a TODAS las bandas
media_por_banda = sim_embed.mean(axis=0, keepdims=True)   # que tanto "le cae bien" esta banda a TODOS los libros
media_global = sim_embed.mean()

sim_corregida = sim_embed - media_por_libro - media_por_banda + media_global

_ = evalua(sim_corregida, "Embeddings + correccion de hubness")
'''))

cells.append(code('''def top1_por_banda(sim):
    return [libros.iloc[np.argmax(sim[:, bi])]["titulo_buscado"] for bi in range(sim.shape[1])]

conteo_antes = pd.Series(top1_por_banda(sim_embed)).value_counts()
conteo_despues = pd.Series(top1_por_banda(sim_corregida)).value_counts()

print("Libro top-1 mas repetido ANTES de corregir hubness:")
print(conteo_antes.head(5))
print(f"-> el mas repetido aparece en {conteo_antes.iloc[0]} de {sim_embed.shape[1]} bandas "
      f"({conteo_antes.iloc[0] / sim_embed.shape[1]:.1%})")

print("\\nLibro top-1 mas repetido DESPUES de corregir hubness:")
print(conteo_despues.head(5))
print(f"-> el mas repetido aparece en {conteo_despues.iloc[0]} de {sim_corregida.shape[1]} bandas "
      f"({conteo_despues.iloc[0] / sim_corregida.shape[1]:.1%})")
'''))

cells.append(md("""### 3.2 Exportar embeddings de referencia (para búsquedas en vivo)

Guarda los embeddings crudos de las 81 bandas y 112 libros, más las constantes usadas en
la corrección de hubness (promedio por banda, promedio por libro, promedio global). Esto
NO hace falta para el recomendador por lotes de arriba, pero sí lo necesita el pipeline de
chat (`inkriff_chat_recomendador.ipynb`): cuando alguien pregunta por un libro o banda que
no está en este catálogo de 81/112, ese notebook calcula su embedding al vuelo y lo compara
contra estos mismos vectores guardados, aplicando la misma fórmula de corrección — así la
recomendación en vivo es consistente con todo lo evaluado arriba.
"""))

cells.append(code('''np.savez(
    "embeddings_referencia.npz",
    emb_bandas=emb_bandas,
    emb_libros=emb_libros,
    banda_col_means=sim_embed.mean(axis=0),
    libro_row_means=sim_embed.mean(axis=1),
    global_mean=np.array(sim_embed.mean()),
)
print("Guardado: embeddings_referencia.npz")
'''))

cells.append(md("""## 4. Comparación

Ahora comparamos tres enfoques: TF-IDF (baseline), embeddings pre-entrenados, y
embeddings con la corrección de hubness de arriba. La corrección de hubness no
necesariamente va a mejorar el rank/MRR promedio contra los 105 pares conocidos (esa
métrica no "ve" el problema de concentración) — su objetivo es otro: que el
recomendador final no repita los mismos 3-4 libros para medio universo de bandas. Por
eso en la sección 5 la decisión de cuál usar como `SIM_FINAL` toma en cuenta ambas
cosas (calidad de rank Y qué tanto se mantiene o se pierde al corregir hubness), no
solo la tabla de métricas.
"""))

cells.append(code('''tabla_comparativa = pd.DataFrame(resultados).T[
    ["rank_prom_banda", "recall10_banda", "mrr_banda", "rank_prom_libro", "recall10_libro", "mrr_libro"]
].round(3)
tabla_comparativa
'''))

cells.append(md("""## 5. Recomendador final: top-5 por libro y por banda

Elegimos `SIM_FINAL` automáticamente entre los enfoques evaluados: preferimos la
versión con corrección de hubness siempre que su MRR combinado (banda + libro) no
caiga más de un 15% contra los embeddings sin corregir — si cae más que eso, el
"costo" en calidad de match no vale la pena y nos quedamos con los embeddings crudos.
Además, sin importar cuál gane, los libros sin sinopsis (sección 0.1) quedan
excluidos como candidatos al construir el top-5 por banda — su perfil es demasiado
genérico como para recomendarlos con confianza.
"""))

cells.append(code('''mrr_embed = resultados["Embeddings multilingues"]["mrr_banda"] + resultados["Embeddings multilingues"]["mrr_libro"]
mrr_corr = resultados["Embeddings + correccion de hubness"]["mrr_banda"] + resultados["Embeddings + correccion de hubness"]["mrr_libro"]

if mrr_corr >= mrr_embed * 0.85:
    SIM_FINAL, ENFOQUE_FINAL = sim_corregida, "Embeddings + correccion de hubness"
else:
    SIM_FINAL, ENFOQUE_FINAL = sim_embed, "Embeddings multilingues"

print(f"Enfoque final elegido: {ENFOQUE_FINAL}")
print(f"  MRR combinado (banda+libro) -> corregido: {mrr_corr:.3f} | sin corregir: {mrr_embed:.3f}")
'''))

cells.append(code('''idx_libros_validos = np.where(libros["tiene_sinopsis"].values)[0]
n_excluidos = len(libros) - len(idx_libros_validos)
print(f"Libros excluidos como candidatos por sinopsis vacia: {n_excluidos} de {len(libros)}")

recomendaciones_por_libro = []
for li, titulo in enumerate(libros["titulo_buscado"]):
    top5_idx = np.argsort(-SIM_FINAL[li])[:5]
    for rank, bi in enumerate(top5_idx, start=1):
        recomendaciones_por_libro.append({
            "libro": titulo,
            "categoria_libro": libros.iloc[li]["categoria_semilla"],
            "libro_perfil_confiable": bool(libros.iloc[li]["tiene_sinopsis"]),
            "rank": rank,
            "banda_recomendada": bandas.iloc[bi]["banda"],
            "subgenero_banda": bandas.iloc[bi]["subgenero_semilla"],
            "similitud": round(float(SIM_FINAL[li, bi]), 4),
        })

df_recom_libros = pd.DataFrame(recomendaciones_por_libro)
df_recom_libros.to_csv("recomendaciones_libro_a_banda.csv", index=False)

recomendaciones_por_banda = []
for bi, banda in enumerate(bandas["banda"]):
    sims_validas = SIM_FINAL[idx_libros_validos, bi]
    top5_local = np.argsort(-sims_validas)[:5]
    top5_idx = idx_libros_validos[top5_local]
    for rank, li in enumerate(top5_idx, start=1):
        recomendaciones_por_banda.append({
            "banda": banda,
            "subgenero_banda": bandas.iloc[bi]["subgenero_semilla"],
            "rank": rank,
            "libro_recomendado": libros.iloc[li]["titulo_buscado"],
            "categoria_libro": libros.iloc[li]["categoria_semilla"],
            "similitud": round(float(SIM_FINAL[li, bi]), 4),
        })

df_recom_bandas = pd.DataFrame(recomendaciones_por_banda)
df_recom_bandas.to_csv("recomendaciones_banda_a_libro.csv", index=False)

print(f"Guardado: recomendaciones_libro_a_banda.csv ({df_recom_libros.shape}) "
      f"y recomendaciones_banda_a_libro.csv ({df_recom_bandas.shape})")
df_recom_libros.head(10)
'''))

cells.append(md("""### Verificación: ¿ya no hay tantos hubs en las recomendaciones finales?
"""))

cells.append(code('''conteo_final = df_recom_bandas["libro_recomendado"].value_counts()
print("Libros mas repetidos en el top-5 final, contando cuantas bandas distintas los recomiendan:")
print(conteo_final.head(10))
print(f"\\nEl libro mas repetido aparece en el top-5 de {conteo_final.iloc[0]} de {bandas.shape[0]} bandas "
      f"({conteo_final.iloc[0] / bandas.shape[0]:.1%}).")
'''))

cells.append(md("""## 6. Revisión manual rápida

Antes de dar por bueno el recomendador, vale la pena ver a ojo un puñado de
recomendaciones para libros/bandas conocidos y juzgar si "se sienten" correctas
(esto es evaluación cualitativa, complementa las métricas de arriba).
"""))

cells.append(code('''for titulo in ["El Senor de los Anillos", "Fourth Wing", "Dracula", "Los juegos del hambre"]:
    fila = df_recom_libros[df_recom_libros["libro"] == titulo]
    if len(fila):
        print(f"\\n{titulo} ({fila.iloc[0][\'categoria_libro\']}):")
        print(fila[["rank", "banda_recomendada", "subgenero_banda", "similitud"]].to_string(index=False))
'''))

cells.append(md("""También vale la pena checar bandas de subgéneros bien distintos entre sí — si el
recomendador ya no tiene el problema de hubs, sus top-5 de libros deberían verse
claramente diferentes entre ellas (antes de la corrección, varias de estas terminaban
recomendando casi los mismos 3-4 libros sin importar el subgénero).
"""))

cells.append(code('''muestra_bandas = bandas["banda"].sample(min(4, len(bandas)), random_state=42)
for banda_nombre in muestra_bandas:
    fila = df_recom_bandas[df_recom_bandas["banda"] == banda_nombre]
    if len(fila):
        print(f"\\n{banda_nombre} ({fila.iloc[0][\'subgenero_banda\']}):")
        print(fila[["rank", "libro_recomendado", "categoria_libro", "similitud"]].to_string(index=False))
'''))

cells.append(md("""## 7. Resumen

Archivos generados:

- `recomendaciones_libro_a_banda.csv` — top-5 bandas recomendadas por libro (112 libros × 5).
- `recomendaciones_banda_a_libro.csv` — top-5 libros recomendados por banda (81 bandas × 5).

Esto ya es un recomendador bidireccional funcional. Sobre la corrida anterior había
detectado un problema real de "hubs" (unos pocos libros dominando las recomendaciones
de medio universo de bandas) — este notebook ya incluye el diagnóstico (sección 3.1) y
dos correcciones: excluir libros sin sinopsis como candidatos (sección 0.1 / 5), y una
corrección de hubness por centrado sobre la matriz de similitud (sección 3.1). Revisa
los prints de la sección 5 ("Verificación") para confirmar que el libro más repetido ya
no acapara tantas bandas como antes.

Los siguientes pasos naturales serían: (a) afinar el perfil de texto de las bandas si
el Recall@10 no convence aún, (b) conseguir sinopsis para los libros que faltan (para
que puedan volver a ser candidatos), (c) armar el pipeline/interfaz (Gradio) que reciba
un libro o banda y muestre las recomendaciones, y (d) documentar todo esto
(metodología, limitaciones, resultados) para el reporte final del Diplomado.
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

with open("inkriff_modelado_similitud.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook generado: inkriff_modelado_similitud.ipynb")
