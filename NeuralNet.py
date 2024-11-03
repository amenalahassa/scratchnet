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

import helpers
from classifieur import Classifier
from helpers import softmax, sigmoid, min_max_scaling, initialize_weights_xavier
from sklearn.neural_network import MLPClassifier
import pandas as pd
import time


# le nom de votre classe
# DecisionTree pour l'arbre de décision
# NeuralNet pour le réseau de neurones
class Layer:
    def __init__(self, input_size, output_size, activation_function = None):
        """
        Initialisation d'une couche du réseau de neurones.

        Args:
        input_size (int): Nombre de neurones en entrée.
        output_size (int): Nombre de neurones en sortie.
        activation_function (callable): Fonction d'activation à utiliser.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function

        # Initialisation des poids
        self.weights = initialize_weights_xavier(input_size, output_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, x):
        """
        Forward pass à travers la couche.

        Args:
        x (ndarray): Entrée de la couche de forme (n, input_size).

        Returns:
        ndarray: Sortie de la couche de forme (n, output_size).
        """
        self.input = x
        if not self.activation_function:
            self.output = np.dot(x, self.weights) + self.bias
            return self.output
        self.output = self.activation_function(np.dot(x, self.weights) + self.bias)
        return self.output

    def backward(self, delta, learning_rate):
        """
        Backpropagation à travers la couche.

        Args:
        delta (ndarray): Gradient en sortie de la couche de forme (n, output_size).
        learning_rate (float): Taux d'apprentissage.

        Returns:
        ndarray: Gradient en entrée de la couche de forme (n, input_size).
        """
        if self.activation_function:
            derivative = self.activation_function(self.output, derivative=True)
            delta *= derivative
        self.weights += np.dot(self.input.T, delta) * learning_rate
        self.bias += np.sum(delta, axis=0) * learning_rate
        return np.dot(delta, self.weights.T)

class NeuralNet(Classifier):
    def __init__(self, input_size, hidden_size, output_size, activation_function, output_activation_function, n_layer = 1, dataset=None, **kwargs):
        """
        C'est un Initializer.
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layer = n_layer
        self.activation_function = activation_function
        self.output_function = output_activation_function
        self.dataset = dataset

        # Initialisation des couches
        self.layers = [Layer(input_size, hidden_size, activation_function)]
        self.layers.extend([Layer(hidden_size, hidden_size, activation_function) for _ in range(n_layer)])
        self.layers.append(Layer(hidden_size, output_size))

    def train(self, train_data, train_labels, test, loss_function, learning_rate=0.1, epochs=1000):
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

        losses = []
        labels = []
        predicted = []

        test_predicted = []
        for epoch in range(epochs):
            # Forward pass
            output = train_data
            for layer in self.layers:
                output = layer.forward(output)

            # Calcul de la perte
            loss, grad = loss_function(output, train_labels)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

            # Backpropagation
            for layer in reversed(self.layers):
                grad = layer.backward(grad, learning_rate)

            losses.append(loss)
            labels.append(train_labels)

            # print(output.shape)
            if self.output_function == sigmoid:
                output = (output > 0.5).flatten()

            if self.output_function == softmax:
                output = np.argmax(output, axis=-1)

            # print(output.shape, train_labels)
            predicted.append(output)

            test_predicted.append(self.predict(test))

        end_time = time.time()  # Enregistrer le temps après la prédiction
        prediction_time = end_time - start_time  # Calculer le temps pris pour la prédiction

        return losses, labels, predicted, test_predicted, prediction_time
    def predict(self, x):
        """
        Prédire la classe d'un exemple x donné en entrée
        exemple est de taille 1xm
        """
        for layer in self.layers:
            x = layer.forward(x)

        if self.output_function == sigmoid:
            x = (x > 0.5).flatten()

        if self.output_function == softmax:
            x = np.argmax(x, axis=-1)
        return x

    def optimal_hidden_size(self, X, y, k = 7, sizes = (10,), learning_rate=1e-2, epoch=100, batch_size=10):

        errors = []

        for size in sizes:
            model = MLPClassifier(
                hidden_layer_sizes=(size,),
                activation="logistic", solver='sgd', learning_rate="adaptive",
                learning_rate_init=learning_rate,
                max_iter=epoch, batch_size=batch_size, verbose = False, early_stopping=True,
            )

            error, _ = self.cross_validate(model, X, y, k)

            errors.append(error)

        best_index = errors.index(min(errors))

        return sizes[best_index], errors

    def optimal_depth(self, X, y, hidden_size, k=7, depths=(2,), learning_rate=1e-2, epoch=100, batch_size=10):

        errors = []

        for depth in depths:
            model = MLPClassifier(
                hidden_layer_sizes=tuple([hidden_size] * depth),
                activation="logistic", solver='sgd', learning_rate="adaptive",
                learning_rate_init=learning_rate,
                max_iter=epoch, batch_size=batch_size, verbose=False, early_stopping=True,
            )

            error, _ = self.cross_validate(model, X, y, k)

            errors.append(error)

        best_index = errors.index(min(errors))

        return depths[best_index], errors

    def fit_transform(self, X, y):
        X = pd.DataFrame(X)

        X_processed = X.copy()

        columns_scales = {}
        # Boucle sur chaque colonne pour trouver les valeurs de coupe optimales
        for column in X.columns:
            # Si la colonne est numérique
            if X[column].dtype in ['int64', 'float64']:
                min_vals = X[column].min()
                max_vals = X[column].max()
                X_processed[column] = min_max_scaling(X[column].values, min_vals, max_vals)
                columns_scales[column] = (min_vals, max_vals)

        self.columns_scales = columns_scales

        return X_processed.values, y

    def transform(self, X, y):
        X = pd.DataFrame(X)

        X_processed = X.copy()
        for column, scale in self.columns_scales.items():
            X_processed[column] = min_max_scaling(X[column].values, scale[0], scale[1])

        return X_processed.values, y

    def __str__(self):
        print(f"Architecture du Neural net pour dataset: {self.dataset}")
        print(f"Nombre de feature (Taille de l'entree) : {self.input_size}")
        print(f"Nombre de couche cachee : {self.n_layer}")
        print(f"Taille des couches cachees : {self.hidden_size}")
        print(f"Activation : {self.activation_function.__name__}")
        print(f"Activation en sortie: {self.output_function.__name__}")
        print("\n\n")
        return ""