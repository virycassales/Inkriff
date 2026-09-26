# Prompt de Arranque — Inkriff

## Para qué sirve este documento

El PDF que compartiste ("InverAI - Prompt Arranque") es el mensaje inicial que el profe le da a un
asistente de código (Claude Code, Cursor, etc.) ANTES de escribir una sola línea: le explica el objetivo,
la arquitectura, el stack y los límites, para que todo lo que construya después sea consistente — en vez
de ir programando pedacitos sueltos sin un plan compartido.

Aquí está el equivalente para Inkriff. Lo vas a usar así: cuando lleguemos a la fase de construir el
pipeline/interfaz real (más adelante en la lista de tareas), pegas el bloque de abajo como primer mensaje
en tu asistente de código (o me lo compartes aquí mismo) para que arranque con todo el contexto de una vez,
en vez de que tengamos que explicarlo de nuevo cada vez.

Una diferencia a propósito frente al de InverAI: el suyo describe una plataforma de producción completa
(monorepo, Angular, Docker, un orquestador con 6 agentes especializados). Para nuestro Trabajo Final ya
decidimos un stack mucho más ligero y 100% en free tier (Gradio sobre Hugging Face Spaces, sin
frontend/backend separados, sin base de datos relacional) — así que este prompt está ajustado a ESE
alcance, no al de InverAI. Si en algún momento tu profe pide igualar la complejidad de InverAI, avísame y
lo ajustamos, pero por ahora mantenerlo simple es lo que nos conviene.

---

## Prompt (copiar y pegar tal cual)

Eres arquitecto full-stack senior en sistemas de recomendación y aplicaciones de IA. Construye "Inkriff":
una plataforma que recomienda bidireccionalmente entre música de rock/metal (y sus subgéneros: death,
power, symphonic, gothic, metalcore, etc.) y libros de fantasía/romance. NO escribas código todavía; lee
todo y ejecuta primero.

### OBJETIVO

Dado el historial de escucha de una persona (bandas, canciones, subgéneros de rock/metal), recomendar
libros de fantasía/romance afines en atmósfera/temática narrativa; y a la inversa, dado un libro o una
serie, recomendar bandas/canciones/subgéneros afines. Cada recomendación debe venir acompañada de una
explicación en lenguaje natural generada por un LLM de por qué encajan (tono, temática, atmósfera) — nunca
solo un score numérico.

Es un proyecto académico (Trabajo Final del Diplomado en Ciencia de Datos, FES Acatlán UNAM), enmarcado
como una empresa ficticia de IA — NO es un producto comercial real. No reproduce ni muestra letras
completas de canciones (riesgo de Términos de Servicio de Genius): solo usa features derivados de ellas.

### INTERFAZ DE USUARIO (vistas)

- Landing / Home: buscador de banda, canción o libro
- Asistente conversacional (chat): recibe un input musical o literario, devuelve la recomendación más la
  explicación generada
- Reproductor de sonido embebido (Spotify oEmbed) para la banda/canción recomendada o de referencia
- Vista "¿Cómo funciona?": muestra de forma simple qué tags/features se usaron para llegar a la
  recomendación (trazabilidad mínima; no requiere panel de administrador ni login)

### STACK

- **Datos**: MusicBrainz API (metadatos y géneros/subgéneros musicales, requiere User-Agent personalizado
  y respetar ~1 req/seg), Genius vía `lyricsgenius` (solo features de texto derivados de letras, nunca el
  texto completo), Open Library API con Google Books como respaldo (metadatos/sinopsis de libros)
- **Procesamiento/Modelado**: Python, spaCy/gensim para features de texto y embeddings, Google Colab como
  entorno principal de entrenamiento (GPU gratuita, sesiones ≤12h), Kaggle Notebooks como respaldo de GPU
  (usado solo como entorno de cómputo, nunca como fuente de datos)
- **Capa generativa**: API de OpenAI para las explicaciones en lenguaje natural, con Groq/Gemini/DeepSeek
  como respaldos gratuitos
- **Interfaz/Despliegue**: Gradio sobre Hugging Face Spaces (2 vCPU/16GB RAM free tier), con Streamlit
  Community Cloud como respaldo
- **Entorno de desarrollo**: VS Code/Cursor + Claude Code

### PIPELINE DEL SISTEMA (no es un proxy de las APIs)

1. **Extracción**: recolectar bandas/subgéneros (MusicBrainz + Genius) y libros/géneros (Open
   Library/Google Books) a partir del universo semilla curado
2. **Ingeniería de variables**: features de texto (densidad léxica, TTR, proporciones POS, densidad de
   entidades nombradas, proporción de pronombres) sobre letras y sinopsis; tags/géneros de MusicBrainz
   como proxy de "mood" musical (no se usan audio-features, esos endpoints de Spotify están deprecados
   desde nov. 2024)
3. **Modelado**: embeddings de libros y de música proyectados a un espacio semántico compartido (dual
   encoder entrenado sobre pares semilla curados), más clustering no supervisado para descubrir "vibras"
   compartidas entre ambos dominios
4. **Inferencia**: dado un input, buscar los vecinos más cercanos en el espacio compartido (embeddings
   precomputados + nearest-neighbor), devolver el top-N
5. **Explicación**: pasar el input y el resultado a la capa de OpenAI para generar la explicación en
   lenguaje natural de la recomendación (nunca solo el score)

### REGLAS DE TRABAJO

1. Mantén un archivo README/AGENTS.md vivo con decisiones, convenciones y pendientes; actualízalo al
   cerrar cada fase
2. Si una librería o API cambió (en particular Spotify, cuyos endpoints de audio-features/recomendaciones
   están deprecados desde nov. 2024), VERIFÍCALO en documentación oficial antes de asumir que funciona
3. Respeta el límite de tasa de MusicBrainz (~1 req/seg) y nunca redistribuyas letras completas de Genius
4. No hagas commits al repositorio sin permiso

### NO CONSTRUYAS

- Scraping no autorizado ni redistribución de contenido con derechos de autor (letras completas, texto
  íntegro de libros)
- Reproducción de audio real (solo enlaces/embeds oficiales vía Spotify oEmbed)
- Un sistema de login/autenticación de usuarios (fuera de alcance para este proyecto)
- Nada que dependa de infraestructura de pago fuera de la API de OpenAI (o sus respaldos gratuitos)
- Un backend/frontend separados tipo InverAI (FastAPI + Angular + Docker); nos quedamos con Gradio sobre
  Hugging Face Spaces por simplicidad y porque cabe en el free tier
