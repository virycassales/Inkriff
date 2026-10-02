# -*- coding: utf-8 -*-
"""Celda común para los notebooks: deja en la carpeta actual los archivos de datos que el
notebook necesita, sin tener que subirlos a mano a Colab.

Para cada archivo: si ya está en la carpeta actual, no hace nada; si se corre desde el repo,
lo copia de data/ o app/; si no (Colab), lo descarga del repositorio público en GitHub.
Los generadores de notebooks (build_*.py) importan celda_descarga() para insertar esa celda."""

# nombre del archivo en la carpeta de trabajo -> ruta dentro del repo
RUTAS = {
    "bandas_completo.csv": "data/raw/bandas_completo.csv",
    "libros.csv": "data/raw/libros.csv",
    "libros_con_sinopsis.csv": "data/processed/libros_con_sinopsis.csv",
    "pares_semilla_final.csv": "data/processed/pares_semilla_final.csv",
    "recomendaciones_banda_a_libro.csv": "data/processed/recomendaciones_banda_a_libro.csv",
    "recomendaciones_libro_a_banda.csv": "data/processed/recomendaciones_libro_a_banda.csv",
    "embeddings_referencia.npz": "data/processed/embeddings_referencia.npz",
    "portadas_libros.csv": "app/portadas_libros.csv",
    "fotos_bandas.csv": "app/fotos_bandas.csv",
}


def celda_descarga(nombres):
    faltan = [n for n in nombres if n not in RUTAS]
    assert not faltan, f"Sin ruta en el repo para: {faltan}"
    lineas = ",\n".join(f'    "{n}": "{RUTAS[n]}"' for n in nombres)
    return f'''# Datos de entrada: no hace falta subir nada a Colab.
# Si un archivo no está en la carpeta actual, se copia del repo (si el notebook se corre desde
# notebooks/) o se descarga del repositorio público en GitHub.
import os
import shutil
import urllib.request

REPO_RAW = "https://raw.githubusercontent.com/virycassales/Inkriff/main/"
ARCHIVOS = {{
{lineas},
}}
for nombre, ruta in ARCHIVOS.items():
    if os.path.exists(nombre):
        origen = "ya estaba en la carpeta"
    elif os.path.exists(os.path.join("..", ruta)):
        shutil.copy(os.path.join("..", ruta), nombre)
        origen = "copiado del repo local"
    else:
        urllib.request.urlretrieve(REPO_RAW + ruta, nombre)
        origen = "descargado de GitHub"
    print(f"{{nombre:38s}} {{origen}}")
'''
