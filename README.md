# diabetes-ml-predictor
Petit projet de machine learning réalisé en Python permettant de prédire le risque de diabète à partir de données médicales.  

# Diabetes Prediction with Machine Learning

## Description

Ce projet est une application simple de machine learning en Python permettant de prédire le risque de diabète à partir de données médicales.
Le modèle est entraîné à l'aide d'une régression logistique avec la bibliothèque scikit-learn.

Le script :

* charge un dataset médical
* entraîne un modèle de classification
* évalue ses performances
* affiche différentes visualisations (corrélations et matrice de confusion)

## Technologies utilisées

* Python
* pandas
* scikit-learn
* matplotlib
* seaborn

## Installation

Installer les dépendances :

```
python3 -m pip install pandas scikit-learn matplotlib seaborn
```

## Utilisation

Lancer le script :

```
python3 diabetes_model.py
```

Le programme :

1. charge les données
2. entraîne le modèle
3. affiche l’accuracy et un rapport de classification
4. affiche des visualisations des résultats

## Objectif du projet

Ce projet a été réalisé dans le but de pratiquer les bases du machine learning :

* prétraitement de données
* entraînement d’un modèle
* évaluation des performances
* visualisation des résultats
