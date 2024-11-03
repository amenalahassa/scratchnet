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
from classifieur import Classifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from helpers import find_best_split
import time

class DecisionTree(Classifier):
	"""
	C'est un Initializer.
	Vous pouvez passer d'autre paramètres au besoin,
	c'est à vous d'utiliser vos propres notations
	"""
	def __init__(self, max_depth=None, min_samples_split=2 , prunning = False, sklearn = False, **kwargs):
		super().__init__(sklearn = sklearn)
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.sklearn = None
		self.prune = prunning

		if sklearn:
			self.sklearn = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)

	def _build_tree(self, train, train_labels, attributes, depth):
		if self.max_depth is not None and depth >= self.max_depth:
			return {'label': self._most_common_label(train_labels)}

		if len(np.unique(train_labels)) == 1:
			return {'label': train_labels[0]}

		if train.shape[0] < self.min_samples_split:
			return {'label': self._most_common_label(train_labels)}

		best_attribute = self._find_best_attribute(train, train_labels, attributes)
		if best_attribute is None:
			return {'label': self._most_common_label(train_labels)}

		children = self._split_data(train, train_labels, best_attribute)
		node = {'attribute': best_attribute}
		attributes.remove(best_attribute)
		for child, x, y in children:
			if len(y) == 0:
				node[child] = {'label': self._most_common_label(train_labels)}
			node[child] = self._build_tree(x, y, attributes, depth + 1)

		return node

	def _find_best_attribute(self, train, train_labels, attributes):
		# Calculate the entropy of the current node
		current_entropy = self._calculate_entropy(train_labels)

		# Initialize the best attribute and its information gain
		best_attribute = None
		best_info_gain = 0

		# Iterate over each attribute
		for attribute in attributes:
			# Calculate the information gain for this attribute
			info_gain = self._calculate_info_gain(train, train_labels, attribute, current_entropy)

			# If this attribute has a higher information gain, update the best attribute
			if info_gain > best_info_gain:
				best_attribute = attribute
				best_info_gain = info_gain

		return best_attribute

	def _calculate_entropy(self, labels):
		# Calculate the entropy of a given set of labels
		unique_labels, counts = np.unique(labels, return_counts=True)
		entropy = -np.sum((counts / len(labels)) * np.log2(counts / len(labels)))
		return entropy

	def _calculate_info_gain(self, train, train_labels, attribute, current_entropy):
		# Calculate the information gain for a given attribute
		unique_values, counts = np.unique(train[:, attribute], return_counts=True)
		info_gain = current_entropy
		for value, count in zip(unique_values, counts):
			subset_labels = train_labels[train[:, attribute] == value]
			info_gain -= (count / len(train)) * self._calculate_entropy(subset_labels)
		return info_gain

	def _split_data(self, train, train_labels, attribute):
		# Split the data on a given attribute
		unique_values = np.unique(train[:, attribute])
		child = []
		for value in unique_values:
			child.append((value, train[train[:, attribute] == value], train_labels[train[:, attribute] == value]))
		return child

	def _most_common_label(self, labels):
		# Return the most common label in a given set of labels
		unique_labels, counts = np.unique(labels, return_counts=True)
		return unique_labels[np.argmax(counts)]

	def _prune_tree(self, node, train, train_labels):
		if 'label' not in node:

			# unique_values = np.unique(train[:, node["attribute"]])
			children = self._split_data(train, train_labels, node["attribute"])
			for child, x, y in children:
				self._prune_tree(node[child], x, y)

			# Calculate error without pruning
			error_before_pruning = self._calculate_error(train, train_labels, self.root)

			# Backup the current node
			backup = node.copy()

			# Prune the current node
			node.clear()
			node['label'] = self._most_common_label(train_labels)

			# Calculate error after pruning
			error_after_pruning = self._calculate_error(train, train_labels, self.root)

			# If pruning doesn't increase error, keep the pruned node
			if error_after_pruning > error_before_pruning:
				node.update(backup)

	def _calculate_error(self, train, train_labels, tree):
		# Calculate misclassification error
		predictions = [self.predict_one(example) for example in train]
		misclassified = sum(y_pred != y_true for y_pred, y_true in zip(predictions, train_labels))
		error = misclassified / len(train_labels)
		return error

	def train(self, train, train_labels):
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le nombre d'attributs (le nombre de caractéristiques)

		train_labels : est une matrice numpy de taille nx1

		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire

		"""
		start_time = time.time()  # Enregistrer le temps avant la prédiction

		if not self.sklearn:
			self.attributes = list(range(train.shape[1]))
			self.root = self._build_tree(train, train_labels, self.attributes,0)

			if self.prune:
				# Prune the tree after building
				self._prune_tree(self.root, train, train_labels)
		else:
			self.sklearn.fit(train, train_labels)

		end_time = time.time()  # Enregistrer le temps après la prédiction
		prediction_time = end_time - start_time  # Calculer le temps pris pour la prédiction

		return prediction_time

	def predict_one(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		# Use the trained tree to predict the label of a given example
		node = self.root
		while 'label' not in node:
			node = node[x[node['attribute']]]
		return node['label']

	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		pred = []
		start_time = time.time()  # Enregistrer le temps avant la prédiction

		if self.sklearn:
			pred = self.sklearn.predict(x)

		else:
			for row in x:
				pred.append(self.predict_one(row))

			pred = np.array(pred)

		end_time = time.time()  # Enregistrer le temps après la prédiction
		prediction_time = end_time - start_time  # Calculer le temps pris pour la prédiction

		return pred, prediction_time
	def fit_transform(self, X, y):
		X = pd.DataFrame(X)

		columns_threshold = {}
		X_processed = X.copy()
		# Boucle sur chaque colonne pour trouver les valeurs de coupe optimales
		for column in X.columns:
			# Si la colonne est numérique
			if X[column].dtype in ['int64', 'float64']:
				# Trouver la meilleure valeur de coupe pour cette colonne
				split_value = find_best_split(X[column].values, y)
				# Discrétiser la colonne en utilisant la valeur de coupe trouvée
				X_processed[column] = np.where(X[column] <= split_value, 0, 1)

				columns_threshold[column] = split_value

		self.columns_threshold = columns_threshold

		return X_processed.values, y

	def transform(self, X, y):
		X = pd.DataFrame(X)

		X_processed = X.copy()
		# Boucle sur chaque colonne pour trouver les valeurs de coupe optimales
		for column, split_value in self.columns_threshold.items():
			X_processed[column] = np.where(X[column] <= split_value, 0, 1)

		return X_processed.values, y
