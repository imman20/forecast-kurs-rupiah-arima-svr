.PHONY: install test lint pipeline notebooks app all

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

lint:
	ruff check src tests app

pipeline:
	PYTHONPATH=src python -m kurs_forecast.pipeline

notebooks:
	python notebooks/buat_notebook.py

app:
	streamlit run app/streamlit_app.py

all: install lint test pipeline notebooks
