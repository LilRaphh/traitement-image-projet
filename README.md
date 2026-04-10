# Deforestation Detector

Pipeline de détection de déforestation par analyse d'images satellites.

## Lecture Métier

Le projet compare deux images d'une même zone prises à deux dates différentes :

- `t0` : image la plus ancienne
- `t1` : image la plus récente

L'objectif est d'aider à répondre à une question simple : **la couverture végétale a-t-elle reculé entre t0 et t1 ?**

L'approche suivie est volontairement lisible :

1. on vérifie d'abord si les deux images sont comparables visuellement ;
2. on améliore légèrement leur lisibilité quand la lumière ou le bruit gênent l'analyse ;
3. on repère automatiquement les zones qui ressemblent le plus à de la végétation ;
4. on compare ensuite la surface végétalisée entre les deux dates ;
5. on classe enfin l'évolution observée ;
6. on produit enfin une carte et un pourcentage de perte.

Important :

- le résultat aide à orienter l'interprétation, mais ne remplace pas une expertise terrain ;
- des différences de saison, d'éclairage, d'ombre, de capteur ou de cadrage peuvent influencer la lecture ;
- les notebooks ont été commentés pour un public non spécialiste du traitement d'image, notamment pour un usage forestier.

## Installation

```bash
python3.12 -m .venv venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancement de l'API

```bash
uvicorn src.api.app:app --reload --port 8001
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
| POST | /compare | Comparaison t0/t1 + quantification + classification |
| POST | /pipeline | Pipeline complet (t0 + t1) avec classification finale |
