## Introduction

Ce projet consiste en l'implémentation des algorithmes Arbre de decision et Reseau de neurones en Python, avec la possibilité d'utiliser des modèles préconstruits de la bibliothèque scikit-learn.

## Description des classes 

#### classifieur.py : 
Ce fichier contient une classe de base dont herite les classifieurs Arbre de decision et Reseau de neurones. Elle impletement la methode evaluate qui est utilise par ces classes.

#### DecisionTree.py : 
Ce fichier contient une classe implémentant l'algorithme d'arbre de decision. La classe a un paramètre sklearn qui permet d'initialiser un modèle scikit-learn si nécessaire.

#### NeuralNet.py : 
Ce fichier contient une classe implémentant l'algorithme des Reseau de neurones en integrant le SGD pour le backprop. La classe a deux methodes permetant d'utiliser un classifier sklearn pour determiner le nombre de couches caches et le nombre de neurones optimales.

#### load_datasets.py : 
Ce fichier les methodes permettant de charger les datasets

#### helpers.py : 
Ce fichier fournit des fonctions facilitant le chargement des données, la création des modèles et l'affichage du tableau récapitulatif.

#### entrainer_tester.ipynb : 
Ce fichier contient le code principale pour ce projet. Il permet de tester les differents algorithmes et reponds aux consignes de l'enonce du tp 4.

## Répartition des tâches

Ce travail a été réalisé seul.


## Difficultés rencontrés

Lors de l'entraînement des modèles avec notre implémentation de l'algorithme Arbre de decision, nous avons rencontré un problème au performances du modele avec la version utilisant l'elagage. Les performances baissent, au lieu d'augmenter.
Nous avons aussi eu des difficultes a implementer correctement la backprop dans notre version des reseau de neurones, ce qui justifie les faibles performances des modeles.
