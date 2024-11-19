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
from scrachnnet.classifieur import Classifier
from scrachnnet.helpers import softmax, sigmoid, min_max_scaling, initialize_weights_xavier
from sklearn.neural_network import MLPClassifier
import pandas as pd
import time
import torch


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
        self.bias = torch.zeros(output_size)

    def forward(self, x):
        """
        Forward pass through the layer.

        Args:
        x (torch.Tensor): Input to the layer of shape (n, input_size).

        Returns:
        torch.Tensor: Output of the layer of shape (n, output_size).
        """
        self.input = x  # Store the input for use in backprop
        self.output = x @ self.weights + self.bias  # Matrix multiplication for forward pass

        # Apply activation function if provided
        if self.activation_function:
            self.activation = self.activation_function(self.output)
            return self.activation

        return self.output

    def backward(self, delta, learning_rate):
        """
        Backpropagation through the layer.

        Args:
        delta (torch.Tensor): Gradient at the layer output of shape (n, output_size).
        learning_rate (float): Learning rate.

        Returns:
        torch.Tensor: Gradient at the layer input of shape (n, input_size).
        """
        # Apply activation function's derivative if necessary
        if self.activation_function:
            derivative = self.activation_function(self.output, derivative=True)
            delta = delta * derivative

        # Calculate gradient of the loss with respect to inputs
        grad_input = delta @ self.weights.T

        # Update weights and biases using gradients
        self.weights.grad = None  # Clear existing gradients (optional for manual updates)
        self.bias.grad = None

        self.weights -= ((self.input.T @ delta) / self.input.shape[0]) * learning_rate
        self.bias -= (delta.sum(dim=0) / self.input.shape[0]) * learning_rate

        return grad_input

class NeuralNet(Classifier):
    def __init__(self, input_size, hidden_size, output_size, activation_function, n_layer = 1, dataset=None, **kwargs):
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
        self.dataset = dataset
        self.output_activation_function = softmax if output_size > 1 else sigmoid

        # Initialisation des couches
        self.layers = [Layer(input_size, hidden_size, activation_function)]
        self.layers.extend([Layer(hidden_size, hidden_size, activation_function) for _ in range(n_layer)])
        self.layers.append(Layer(hidden_size, output_size, self.output_activation_function))

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

            # Backpropagation
            for layer in reversed(self.layers):
                grad = layer.backward(grad, learning_rate)

            losses.append(loss)
            labels.append(train_labels.numpy())

            if self.output_size == 1:
                output = (output > 0.5).flatten().to(torch.float)
            else:
                output = torch.argmax(output, dim=-1)

            predicted.append(output)

            test_predicted.append(self.predict(test))

        end_time = time.time()  # Enregistrer le temps après la prédiction
        prediction_time = end_time - start_time  # Calculer le temps pris pour la prédiction

        return losses, labels, predicted, test_predicted, prediction_time

    def predict(self, x, model = None):
        """
        Prédire la classe d'un exemple x donné en entrée
        exemple est de taille 1xm
        """
        if model is not None:
            x = model(x)
        else:
            for layer in self.layers:
                x = layer.forward(x)

        if self.output_size == 1:
            x = (x > 0.5).flatten().to(torch.float)
        else:
            x = torch.argmax(x, dim=-1)

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


    def as_pytorch(self):
        """
        Implémentation du réseau de neurones avec PyTorch.
        Using the same architecture as the one defined in the NeuralNet class.
        """

        activation = torch.nn.ReLU
        if self.activation_function == sigmoid:
            activation = torch.nn.Sigmoid

        # Define the model
        # Initiate the weights and biases with NeuralNet's weights and biases
        model = torch.nn.Sequential()
        model.add_module("input", torch.nn.Linear(self.input_size, self.hidden_size))
        model.add_module("activation", activation())
        for i in range(self.n_layer):
            model.add_module(f"hidden_{i}", torch.nn.Linear(self.hidden_size, self.hidden_size))
            model.add_module(f"activation_{i}", activation())

        model.add_module("output", torch.nn.Linear(self.hidden_size, self.output_size))

        # Copy the weights and biases
        layers = [layer for layer in model if isinstance(layer, torch.nn.Linear)]
        for i, layer in enumerate(layers):
            layer.weight.data = self.layers[i].weights.T
            layer.bias.data = self.layers[i].bias.flatten()

        for param in model.parameters():
            param.requires_grad = True

        model.output_activation_function = self.output_activation_function
        return model


    def pytorch_train(self, model, train_data, train_labels, test, loss_function, learning_rate=0.1, epochs=1000):
        """
        Training the PyTorch model using PyTorch.
        """
        start_time = time.time()

        # Define the optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        losses = []
        labels = []
        predicted = []
        test_predicted = []

        for epoch in range(epochs):
            model.train()

            # Forward pass
            output = model(train_data)
            output = model.output_activation_function(output)

            # Calculate the loss
            loss, _ = loss_function(output, train_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            labels.append(train_labels.numpy())

            if self.output_size == 1:
                output = (output > 0.5).flatten().to(torch.float)
            else:
                output = torch.argmax(output, dim=-1)

            predicted.append(output)

            # Predict the test set
            model.eval()
            val_predicted = model(test)

            if self.output_size == 1:
                val_predicted = (val_predicted > 0.5).flatten().to(torch.float)
            else:
                val_predicted = torch.argmax(val_predicted, dim=-1)

            test_predicted.append(val_predicted)

        end_time = time.time()
        prediction_time = end_time - start_time

        return losses, labels, predicted, test_predicted, prediction_time


    def __str__(self):
        print(f"Architecture du Neural net pour dataset: {self.dataset}")
        print(f"Nombre de feature (Taille de l'entree) : {self.input_size}")
        print(f"Nombre de couche cachee : {self.n_layer}")
        print(f"Taille des couches cachees : {self.hidden_size}")
        print(f"Activation : {self.activation_function.__name__}")
        print(f"Nombre de classes (Taille de la sortie) : {self.output_size}")
        print("\n\n")
        return ""