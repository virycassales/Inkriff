---
title: Inkriff
emoji: 🎸
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
short_description: Recomendador bidireccional rock/metal ↔ fantasía/romance
---

# Inkriff: app de chat

Esta carpeta es el **Space de Hugging Face**. El bloque de arriba (entre `---`) es la configuración que lee Hugging Face; no lo borres.

- Se despliega **automáticamente** desde GitHub con `.github/workflows/deploy-hf.yml` cada vez que cambia algo en `app/`.
- `app.py` se genera con `scripts/build_chat_app.py`. No lo edites a mano.
- Los CSV y el `.npz` se copian desde `data/` con `make sync-app`, o los copia el workflow de modelado.

Documentación completa: [github.com/virycassales/Inkriff](https://github.com/virycassales/Inkriff).
