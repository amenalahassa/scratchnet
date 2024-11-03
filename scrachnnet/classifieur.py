"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenir au moins les 3 méthodes definies ici bas, 
	* train 	: pour entraîner le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

import numpy as np
import scrachnnet.helpers
import time



# le nom de votre classe
# DecisionTree pour l'arbre de décision
# NeuralNet pour le réseau de neurones

class Classifier: #nom de la class à changer

	def __init__(self, sklearn = False, **kwargs):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		self.sklearn = sklearn
		
	def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le nombre d'attributs (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		"""
		pass
        
	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		pass

	def fit(self, x, y):
		return self.train(x, y)
	
	def cross_validate(self, model, X, y, k_fold=10, train_params=None):
		"""
		Cette méthode réalise une validation croisée k-fold pour évaluer le modèle

		Arguments :
		- X : numpy array de taille (n, m), où n est le nombre d'exemples de test et m est le nombre d'attributs (caractéristiques).
		- y : numpy array de taille (n, 1), contenant les étiquettes correspondant aux exemples de test.
		- k_fold : int, le nombre de sous-ensembles dans lesquels les données seront divisées pour la validation croisée. Par défaut, k_fold=10.


		Retourne :
		- moy_errors : int, l'erreur moyenne.
		- errors : list, une liste contenant les erreurs.

		"""
		# Initialiser la liste pour stocker les erreurs moyennes pour chaque valeur de k
		if train_params is None:
			train_params = {}

		errors = []

		# Diviser les données en k_fold sous-ensembles
		subset_size = len(X) // k_fold
		subsets_X = [X[i:i + subset_size] for i in range(0, len(X), subset_size)]
		subsets_y = [y[i:i + subset_size] for i in range(0, len(y), subset_size)]

		# Effectuer la validation croisée k_fold fois
		for i in range(k_fold):
			# Sélectionner l'ensemble de validation et l'ensemble d'apprentissage
			X_test = subsets_X[i]
			y_test = subsets_y[i]
			X_train = np.concatenate(subsets_X[:i] + subsets_X[i + 1:])
			y_train = np.concatenate(subsets_y[:i] + subsets_y[i + 1:])

			model.fit(X_train, y_train, **train_params)
			y_pred = model.predict(X_test)
			error = np.mean(y_pred != y_test)
			errors.append(error)

		return np.mean(errors), errors
	
	def evaluate(self, X, y):
		"""
		c'est la méthode qui va evaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)

		y : est une matrice numpy de taille nx1

		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
		start_time = time.time()  # Enregistrer le temps avant la prédiction
		y_pred = self.predict(X)
		conf_matrix = helpers.confusion_matrix(y, y_pred)
		accuracy = helpers.accuracy_score(y, y_pred)
		precision = helpers.precision_score(y, y_pred, average='macro')
		recall = helpers.recall_score(y, y_pred, average='macro')
		f1_score = helpers.f1_score(y, y_pred, avg='macro')
		end_time = time.time()  # Enregistrer le temps après la prédiction
		prediction_time = end_time - start_time  # Calculer le temps pris pour la prédiction

		return conf_matrix, accuracy, precision, recall, f1_score, prediction_time
	
	# Vous pouvez rajouter d'autres méthodes et fonctions,
	# il suffit juste de les commenter.