# -*- coding: utf-8 -*-
"""Despliega la carpeta app/ como Space de Hugging Face.

Lo corre el workflow .github/workflows/deploy-hf.yml, pero también funciona a mano:

    HF_TOKEN=hf_xxx python scripts/deploy_hf.py

Variables de entorno:
    HF_TOKEN        (obligatoria) token de Hugging Face con permiso de escritura
    HF_SPACE        (opcional)    "usuario/nombre"; por defecto "<tu usuario>/inkriff"
    OPENAI_API_KEY  (opcional)    si viene, se guarda como secret del Space
    GROQ_API_KEY    (opcional)    igual, respaldo gratuito
    INKRIFF_SUGERENCIAS (opcional) "off" para no crear el Dataset de sugerencias
"""
import os
import sys

from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("Falta HF_TOKEN (créalo en https://huggingface.co/settings/tokens con permiso Write).")

api = HfApi(token=token)
usuario = api.whoami()["name"]
repo_id = os.environ.get("HF_SPACE") or f"{usuario}/inkriff"

# 1. Crear el Space si todavía no existe (gratis: CPU basic)
api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)

# 2. Pasar las API keys como secrets del Space (nunca quedan en el código)
for nombre in ("OPENAI_API_KEY", "GROQ_API_KEY"):
    valor = os.environ.get(nombre)
    if valor:
        api.add_space_secret(repo_id, nombre, valor)
        print(f"Secret {nombre} actualizado en el Space")

# 3. Dataset PRIVADO donde se guardan las sugerencias de la pestaña "Sugerir"
#    (sin esto la app funciona igual, pero las sugerencias se pierden al reiniciar el Space)
if os.environ.get("INKRIFF_SUGERENCIAS", "on").lower() != "off":
    dataset_id = f"{usuario}/inkriff-sugerencias"
    api.create_repo(dataset_id, repo_type="dataset", private=True, exist_ok=True)
    api.add_space_variable(repo_id, "INKRIFF_DATASET_ID", dataset_id)
    api.add_space_secret(repo_id, "HF_TOKEN", token)
    print(f"Sugerencias -> https://huggingface.co/datasets/{dataset_id} (privado)")

# 4. Subir el contenido de app/
sha = os.environ.get("GITHUB_SHA", "local")[:7]
api.upload_folder(
    folder_path="app",
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=["__pycache__/*", "*.pyc", ".env"],
    commit_message=f"Deploy desde GitHub ({sha})",
)

url = f"https://huggingface.co/spaces/{repo_id}"
print(f"Desplegado: {url}")
resumen = os.environ.get("GITHUB_STEP_SUMMARY")
if resumen:
    with open(resumen, "a", encoding="utf-8") as f:
        f.write(f"### 🎸 Inkriff desplegado\n\n{url}\n\nEl Space tarda unos minutos en construir la primera vez.\n")
