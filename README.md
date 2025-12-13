````markdown
# Virus Diagnosis – Machine Learning Pipeline

## Description
Ce projet consiste à développer un pipeline de **diagnostic de maladie** basé sur un modèle de **régression logistique**.  
L’objectif principal du TP est de mettre en pratique les **bonnes pratiques de génie logiciel**, notamment l’utilisation du **design pattern Builder** pour structurer un pipeline de machine learning.


## Architecture du projet

```text
virus_diag_test/
│── main.py
│── clinical_predictor.py
│── model.pkl
│
├── core/
│   │── dataset.py
│   │── logistic_regression.py
│   │── model.py
│
├── pipeline/
│   │── builder.py
│   │── clinical_builder.py
│   │── ml_pipeline.py
│   │── trainer.py
│   │── evaluate.py
│
├── utils/
│   │── preprocessing.py
│
├── data/
│   │── disease_diagnosis.csv
│
└── README.md ```

<!-- ```` -->


## Design Pattern utilisé : Builder

Le **design pattern Builder** est utilisé pour construire le pipeline de machine learning étape par étape, en séparant clairement :

* la **construction** du pipeline
* son **exécution**

### Rôles dans le pattern

| Élément          | Classe                    |
| ---------------- | ------------------------- |
| Product          | `MLPipeline`              |
| Builder abstrait | `PipelineBuilder`         |
| Builder concret  | `ClinicalPipelineBuilder` |
| Director         | `TrainerDirector`         |
| Client           | `main.py`                 |

### Avantages

* Séparation des responsabilités
* Code plus lisible et maintenable
* Facilité d’extension (autres modèles, autres pipelines)


## Fonctionnement du pipeline

1. Chargement des données
2. Prétraitement (encodage des variables catégorielles)
3. Séparation train / test
4. Construction du modèle
5. Entraînement et sauvegarde
6. Évaluation
7. Prédiction sur un nouveau patient


## Exécution du projet

### Prérequis

* Python 3.9+
* Bibliothèques :

  * `scikit-learn`
  * `pandas`
  * `joblib`

### Installation des dépendances

```bash
pip install -r requirements.txt
```

*(ou installer manuellement les bibliothèques)*


### Lancer le projet

```bash
python main.py
```


## Exemple de prédiction

Le fichier `clinical_predictor.py` permet d’utiliser le modèle entraîné pour prédire l’état d’un patient :

```python
predictor = ClinicalPredictor(model.model)
result = predictor.diagnose(sample)
print(result)
```


## Objectif pédagogique

Ce TP vise à :

* Appliquer le **design pattern Builder**
* Structurer un projet de machine learning selon les principes du **génie logiciel**
* Améliorer la lisibilité, la maintenabilité et l’évolutivité du code


## Auteur

* **Ezzahra RB**
* Master IA – Génie Logiciel

```
