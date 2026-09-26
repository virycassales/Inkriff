# Inkriff — comandos del proyecto
# Uso: make <objetivo>   (en Windows sin make, ver README > Puesta en marcha)

PY ?= python

.PHONY: help setup run notebooks app sync-app clean

help:          ## Lista los comandos disponibles
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  %-10s %s\n", $$1, $$2}'

setup:         ## Instala dependencias (notebooks + app)
	$(PY) -m pip install -r requirements.txt

run:           ## Arranca el chat en http://localhost:7860 (lee .env si existe)
	cd app && set -a && [ -f ../.env ] && . ../.env; set +a; $(PY) app.py

notebooks:     ## Regenera los notebooks desde scripts/ (no se editan a mano)
	cd notebooks && $(PY) ../scripts/build_notebook.py \
	  && $(PY) ../scripts/build_notebook_full.py \
	  && $(PY) ../scripts/build_notebook_eda.py \
	  && $(PY) ../scripts/build_notebook_modelado.py

app:           ## Regenera app/app.py y notebooks/inkriff_chat_recomendador.ipynb
	cd app && $(PY) ../scripts/build_chat_app.py && mv inkriff_chat_recomendador.ipynb ../notebooks/

sync-app:      ## Copia a app/ los datos que consume el chat
	cp data/raw/bandas_completo.csv \
	   data/processed/libros_con_sinopsis.csv \
	   data/processed/recomendaciones_banda_a_libro.csv \
	   data/processed/recomendaciones_libro_a_banda.csv \
	   data/processed/embeddings_referencia.npz app/

clean:         ## Borra cachés de Python y checkpoints
	find . -name __pycache__ -o -name .ipynb_checkpoints | xargs rm -rf
