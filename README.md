# Deforestation Detector

Pipeline de détection de déforestation par analyse d'images satellites.

## Installation

```bash
pip install -r requirements.txt
```

## Lancement de l'API

```bash
uvicorn src.api.app:app --reload
```

## Documentation API

Une fois lancée : http://localhost:8000/docs

## Structure

```
src/
  api/        → routes FastAPI
  core/       → logique métier
  utils/      → utilitaires (I/O, visualisation)
data/
  t0/         → images "avant"
  t1/         → images "après"
outputs/      → résultats générés
notebooks/    → démonstration
```

## Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| POST | /analyze | Histogrammes + stats initiales |
| POST | /preprocess | Prétraitement de l'image |
| POST | /segment | Segmentation K-means + masque végétation |
| POST | /compare | Comparaison t0/t1 + quantification |
| POST | /pipeline | Pipeline complet (t0 + t1) |
