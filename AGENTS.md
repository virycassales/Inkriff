# AGENTS.md — Inkriff

Documento vivo de **decisiones, convenciones y trampas verificadas** del proyecto. Sirve para cualquier persona o asistente de código (Claude Code, Cursor, Copilot…) que trabaje en el repo: léelo **antes** de escribir código.

El prompt de arranque original del proyecto está en [`docs/Inkriff_Prompt_Arranque.md`](docs/Inkriff_Prompt_Arranque.md). Este archivo registra lo que **realmente** se decidió después, incluidos los cambios respecto a ese plan inicial.

---

## 1. Contexto

- **Qué es:** recomendador bidireccional rock/metal ↔ fantasía/romance, con explicación en lenguaje natural.
- **Para qué:** Trabajo Final del Diplomado en Ciencia de Datos (FES Acatlán, UNAM), presentado como una empresa ficticia de IA. **No es un producto comercial.**
- **Restricciones del Diplomado:** los datos deben venir de APIs o scraping (no de Kaggle), y todo tiene que correr en *free tier* (Colab, APIs gratuitas; la única excepción de pago es OpenAI).

---

## 2. Decisiones

| ID | Decisión | Por qué |
|----|----------|---------|
| **D1** | Similitud de contenido (embeddings pre-entrenados + coseno) en vez de entrenar un *dual encoder* | Con 105 pares etiquetados no se pueden separar entrenamiento y validación sin memorizar. *Cambio respecto al prompt de arranque*, que planteaba un dual encoder |
| **D2** | Los 105 pares curados se usan **solo para evaluar** | Si se usaran para ajustar, las métricas quedarían infladas |
| **D3** | Corrección de *hubness* por centrado: `sim − media_libro − media_banda + media_global` | Sin ella, "El Señor de los Anillos" aparecía en el top-5 de 50.6% de las bandas |
| **D4** | Los libros sin sinopsis se excluyen **como candidatos**, pero **siguen contando** en la evaluación | Evita perfiles degenerados sin maquillar las métricas |
| **D5** | Modelo `paraphrase-multilingual-MiniLM-L12-v2` | Multilingüe (el catálogo mezcla títulos en español e inglés) y ligero |
| **D6** | Sin letras de canciones; el *mood* musical sale de los tags de MusicBrainz | ToS de Genius y bloqueo 403 de Cloudflare a `lyricsgenius` |
| **D7** | Sin *audio features* de Spotify, solo el embed público | Esos endpoints están deprecados desde nov. 2024 |
| **D8** | LLM: OpenAI como principal y Groq como respaldo automático | Se cumple el plan (OpenAI) sin quedarse sin servicio si se agota la cuota |
| **D9** | 3 herramientas separadas (`libros_para_banda`, `bandas_para_libro`, `canciones_de_banda`) en vez de una con parámetro `tipo` | Con una sola, el LLM llamaba "bandas" a resultados que eran libros (bug real con Bad Omens) |
| **D10** | Gradio en un solo proceso, sin backend/frontend separados ni base de datos | Cabe en *free tier* y alcanza para el alcance académico |
| **D12** | Hosting en Hugging Face Spaces (CPU basic, gratis), **no Vercel** | Vercel solo corre funciones serverless cortas (~250 MB); el chat necesita un proceso persistente y torch + el modelo MiniLM |
| **D14** | Las sugerencias solo se guardan en un CSV local (versión final del notebook) | Más simple. Implica que en el Space se pierden al reiniciar (ver pendientes) |
| **D15** | Portadas y fotos se precalculan con `imagenes.yml` y se guardan en `app/` | El Space no las vuelve a buscar en cada reinicio; las fotos salen del oEmbed público de Spotify, sin API key |
| **D16** | Respaldo sin LLM y corrección de herramientas por código | Si el LLM falla o confunde banda con libro, igual se muestran las recomendaciones correctas |
| **D13** | El modelo se regenera con GitHub Actions (`modelo.yml`) en vez de Colab | Reproducible, sin subir archivos a mano, y verifica las 384 dims antes de hacer commit |
| **D11** | Ítems fuera del catálogo: búsqueda en vivo + embedding al vuelo con el **mismo** modelo y la **misma** corrección | Así una búsqueda en vivo es comparable con el catálogo precalculado |

---

## 3. Convenciones

1. **Los notebooks y `app/app.py` son generados.** Se editan los `scripts/build_*.py` y se regeneran (`make notebooks`, `make app`). Si editas el `.ipynb` directamente, el siguiente *build* borra tus cambios.
2. **Nombres en español**, sin acentos en identificadores (`recomendar_libros_para_banda`, `titulo_buscado`).
3. **Llave de cruce normalizada** entre tablas: minúsculas, sin acentos, sin puntuación (`normaliza()`). Nunca se cruza por el texto crudo.
4. **CSVs siempre en UTF-8.** Si se exporta desde Excel, hay que elegir "CSV UTF-8".
5. **Flujo de datos:** `data/raw` → `data/processed` → `make sync-app` → `app/`. La app nunca lee de `data/` directamente, porque tiene que poder subirse sola a Spaces.
6. **No hacer commits sin permiso** de la dueña del repo.
7. **Toda recomendación lleva su porqué.** Nunca se muestra solo un score.

---

## 4. Trampas verificadas

| # | Trampa | Solución |
|---|--------|----------|
| 4.1 | MusicBrainz devuelve 503 con frecuencia (límite de ~1 req/s compartido) | User-Agent propio + reintento con *backoff* 1-2-4-8-16 s |
| 4.2 | Open Library / Google Books devuelven 429 | Reintento exponencial en `_get_con_reintento` |
| 4.3 | Genius devuelve el artista equivocado con nombres genéricos ("Death" → Arctic Monkeys, "Ghost" → Kanye West): 10 de 81 bandas | `qa_check.py` marca `similitud < 0.5`; esa popularidad no se usa como señal |
| 4.4 | El QA de libros da falsos positivos por traducción ("El Nombre del Viento" → "The Name of the Wind") | Umbral más laxo (0.25) y revisión manual |
| 4.5 | Las sinopsis de Open Library traen URLs, `[3]:`, ISBN, "bestseller"… que se colaban al TF-IDF | Regex + stopwords completas de sklearn + lista de "ruido editorial" (`rebuild_features_v2.py`) |
| 4.6 | TF-IDF entre dominios ≈ azar: los tags musicales casi no comparten palabras con las sinopsis | Embeddings semánticos (D1/D5) |
| 4.7 | *Hubness*: unos pocos libros dominan todas las recomendaciones | D3 + D4 |
| 4.8 | El CSV de pares llegó con `�` en lugar de acentos (exportado desde Excel) | Reconstruido en `build_pares_final.py` |
| 4.9 | Groq devuelve `400 Tool choice is none, but model called a tool` | Mandar `tools=..., tool_choice="auto"` en **todas** las rondas |
| 4.10 | Un `<iframe>` de Spotify cortado a la mitad rompe el efecto de máquina de escribir | Los embeds se pegan completos al final del mensaje |
| 4.11 | El modelo de embeddings no se descarga en entornos sin salida a huggingface.co | Correr el notebook 4 en Colab |
| 4.12 | `embeddings_referencia.npz` tiene que tener **384 dims** (MiniLM). Si tiene otra dimensión, la búsqueda en vivo falla | `modelo.yml` lo regenera y falla si las dims no son 384 |
| 4.13 | La app usa la API de **Gradio 6** (`theme`/`css` en `launch()`) | `sdk_version: 6.28.0` en `app/README.md` y `gradio==6.28.0` en requirements |
| 4.14 | torch desde PyPI trae CUDA (~2 GB) y alarga el build del Space | `--extra-index-url .../whl/cpu` en `app/requirements.txt` |
| 4.16 | El LLM perdía el hilo porque solo veía el HTML de sus respuestas anteriores | Resumen invisible por respuesta + historial en texto limpio |
| 4.17 | Gradio 6 cambió el formato del historial del chat | Se normaliza antes de mandarlo al LLM |
| 4.15 | `make notebooks` sobreescribe el notebook 4 ejecutado (con resultados) por uno vacío | Después de regenerar, corre *Actions → Regenerar modelo* |

---

## 5. Pendientes

- [ ] Conseguir sinopsis de los 15 libros que no la tienen.
- [ ] Ampliar los 105 pares de evaluación.
- [x] Primera corrida de `Regenerar modelo`: reproduce al 100% la corrida de Colab del documento de metodología (MRR combinado 0.160 contra 0.133 sin corregir). Se borraron los `*_real.csv` duplicados.
- [x] Pestañas Catálogo y Sugerir y links de compra integradas en `scripts/build_chat_app.py` a partir del notebook final.
- [ ] Sugerencias permanentes en el Space (hoy se pierden al reiniciar): por ejemplo, un Dataset privado de HF.
- [ ] Presentación y video final del Diplomado.
