import numpy as np
import pandas as pd
from scrachnnet.load_datasets import load_iris_dataset, load_wine_dataset, load_abalone_dataset
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import torch


LOADERS = {
    'load_iris_dataset': load_iris_dataset,
    'load_wine_dataset': load_wine_dataset,
    'load_abalone_dataset': load_abalone_dataset
}


def dataset_loader(names, train_ratio = .8):
    """
    Load datasets and preprocess them.

    Parameters:
    names : list of str
        List of dataset names to load.
    train_ratio : float, optional
        Ratio of data to use for training, default is 0.8.

    Returns:
    datasets : dict
        Dictionary containing loaded datasets and their preprocessed versions.
    """
    datasets = {}
    for d in names:
        loader = f'load_{d}_dataset'
        train, train_labels, test, test_labels = LOADERS[loader](train_ratio)

        # Encoder la variable categorielle de abalone
        if d == 'abalone':
            oe = OneHotEncoder(sparse_output=False)
            # L'attribut sexe a trois valeurs possibles F, I, M.
            # Encoder cet attribut en one-hot vecteur va donner trois colonnes pour chaque valeur
            encoded = oe.fit_transform(train[:, 0].reshape(-1, 1))
            # On ne va utiliser que la colonne des valeurs F et I
            train = np.concatenate((encoded[:, [0, 1]], train[:, 1:]), axis=1)
            encoded = oe.transform(test[:, 0].reshape(-1, 1))
            # On procède de la meme manière pour les données test
            test = np.concatenate((encoded[:, [0, 1]], test[:, 1:]), axis=1)

        labels = np.unique(np.concatenate((train_labels, test_labels))).astype(int)
        datasets[d] = (train, train_labels, test, test_labels, labels)

    return datasets


def display_results(history):
    """
    Display evaluation results in a summary table.

    Parameters:
    history : list of tuples
        List of tuples containing data, model, and evaluation results.

    Returns:
    None
    """
    data, models, results = zip(*history)
    conf_matrix, accuracy, precision, recall, f1_score = zip(*results)
    df = pd.DataFrame(data = {
        'Jeu de donnée ': data, 'Model': models, 'Accuracy': accuracy,
        'Precision': precision, 'Recall': recall, 'F1-score': f1_score,
        'Confusion matrix': conf_matrix
    })
    print("Tableau récapitulatif")
    print(df)


def plot(x, y, title, xlabel='Epoch', ylabel='Accuracy'):
    plt.plot(x, y, 'b', label=ylabel)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plot_train_test(x, y, title, xlabel='Epoch', ylabel='Accuracy'):
    plt.plot(x, y[0], label=f"{ylabel} on Training")
    plt.plot(x, y[1], label=f"{ylabel} on Test")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plot_confusion_matrix(cm, labels=None):
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def train_test_split(X, y, test_size=0.25, random_state=None):
    """
    Split the data into train and test sets.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        The input data.
    y : array-like, shape (n_samples,)
        The target labels.
    test_size : float or int, default=0.25
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the training and testing indices.

    Returns:
    X_train : array-like, shape (n_train_samples, n_features)
        The training input data.
    X_test : array-like, shape (n_test_samples, n_features)
        The testing input data.
    y_train : array-like, shape (n_train_samples,)
        The training target labels.
    y_test : array-like, shape (n_test_samples,)
        The testing target labels.
    """
    # Set random seed if specified
    if random_state is not None:
        np.random.seed(random_state)

    # Convert test_size to integer if it represents absolute number of test samples
    if isinstance(test_size, float):
        test_size = int(test_size * len(X))

    # Shuffle indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Split indices into train and test sets
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    # Split data based on indices
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

def min_max_scaling(X, min_vals, max_vals):
    X_scaled = (X - min_vals) / (max_vals - min_vals)
    return X_scaled

def find_best_split(feature, target):
    # Trier les valeurs de la caractéristique et les cibles correspondantes
    sorted_indices = np.argsort(feature)
    sorted_feature = feature[sorted_indices]
    sorted_target = target[sorted_indices]

    best_gain = 0
    best_split_value = None

    # Parcours des valeurs triées pour trouver les points de coupure
    for i in range(1, len(sorted_feature)):
        if sorted_feature[i] != sorted_feature[i - 1]:
            # Valeur de coupure potentielle
            split_value = (sorted_feature[i] + sorted_feature[i - 1]) / 2

            # Division des cibles en deux groupes selon la valeur de coupure
            left_targets = sorted_target[sorted_feature <= split_value]
            right_targets = sorted_target[sorted_feature > split_value]

            # Calcul du gain d'information
            gain = information_gain(target, left_targets, right_targets)

            # Mise à jour du meilleur gain et de la valeur de coupure correspondante
            if gain > best_gain:
                best_gain = gain
                best_split_value = split_value

    return best_split_value

def information_gain(parent, left_child, right_child):
    # Calcul de l'entropie du parent
    entropy_parent = entropy(parent)

    # Calcul de l'entropie des enfants
    entropy_left = entropy(left_child)
    entropy_right = entropy(right_child)

    # Calcul du gain d'information
    gain = entropy_parent - (len(left_child) / len(parent)) * entropy_left - (len(right_child) / len(parent)) * entropy_right

    return gain

def entropy(y):
    # Calcul de l'entropie
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Fonction principale pour préparer les données
def prepare_data_for_decision_tree(X, y):
    # Création d'une copie des données
    X_processed = X.copy()

    # Boucle sur chaque colonne pour trouver les valeurs de coupe optimales
    for column in X.columns:
        # Si la colonne est numérique
        if X[column].dtype in ['int64', 'float64']:
            # Trouver la meilleure valeur de coupe pour cette colonne
            split_value = find_best_split(X[column].values, y)
            # Discrétiser la colonne en utilisant la valeur de coupe trouvée
            X_processed[column] = np.where(X[column] <= split_value, 0, 1)

    return X_processed

def confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix

    Parameters:
    y_true : array-like, true labels
    y_pred : array-like, predicted labels

    Returns:
    conf_matrix : numpy array, confusion matrix
    """
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    num_classes = len(unique_labels)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        conf_matrix[int(true_label), int(pred_label)] += 1

    return conf_matrix

def precision_score(y_true, y_pred, average='macro'):
    """
    Calculate precision score

    Parameters:
    y_true : array-like, true labels
    y_pred : array-like, predicted labels
    average : string, optional (default='macro'), type of averaging

    Returns:
    precision : float, precision score
    """
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    num_classes = len(unique_labels)
    precision = 0
    # print(y_true, y_pred)
    for label in unique_labels:
        # print((y_true == label).shape, (y_pred == label).shape, y_true.shape)
        true_positive = np.sum((y_true == label) & (y_pred == label))
        false_positive = np.sum((y_true != label) & (y_pred == label))
        if true_positive + false_positive != 0:
            precision += true_positive / (true_positive + false_positive)

    if average == 'macro':
        precision /= num_classes
    elif average == 'micro':
        precision = np.sum(precision) / len(y_true)

    return precision

def recall_score(y_true, y_pred, average='macro'):
    """
    Calculate recall score

    Parameters:
    y_true : array-like, true labels
    y_pred : array-like, predicted labels
    average : string, optional (default='macro'), type of averaging

    Returns:
    recall : float, recall score
    """
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    num_classes = len(unique_labels)
    recall = 0

    for label in unique_labels:
        true_positive = np.sum((y_true == label) & (y_pred == label))
        false_negative = np.sum((y_true == label) & (y_pred != label))
        if true_positive + false_negative != 0:
            recall += true_positive / (true_positive + false_negative)

    if average == 'macro':
        recall /= num_classes
    elif average == 'micro':
        recall = np.sum(recall) / len(y_true)

    return recall

def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy score

    Parameters:
    y_true : array-like, true labels
    y_pred : array-like, predicted labels

    Returns:
    accuracy : float, accuracy score
    """
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total
    return accuracy

def f1_score(y_true, y_pred, avg = "macro"):
    precision = precision_score(y_true, y_pred, avg)
    recall = recall_score(y_true, y_pred, avg)
    return 2 * (precision * recall) / (precision + recall)

def metrics_per_class(cm):
    """
    Calculate accuracy, precision, recall, and F1-score for each class from confusion matrix.

    Parameters:
    cm : array-like
        Confusion matrix.

    Returns:
    accuracy_per_class : array-like
        Accuracy for each class.
    precision_per_class : array-like
        Precision for each class.
    recall_per_class : array-like
        Recall for each class.
    f1_score_per_class : array-like
        F1-score for each class.
    """
    # Calculate total true positives per class
    tp_per_class = np.diag(cm)

    # Calculate total predicted positives per class
    predicted_positives_per_class = np.sum(cm, axis=0)

    # Calculate total actual positives per class
    actual_positives_per_class = np.sum(cm, axis=1)

    predicted_positives_per_class[predicted_positives_per_class == 0] = 1
    actual_positives_per_class[actual_positives_per_class == 0] = 1

    # Calculate accuracy per class
    accuracy_per_class = tp_per_class / actual_positives_per_class
    accuracy_per_class = np.nan_to_num(accuracy_per_class)  # Handle division by zero

    # Calculate precision per class
    precision_per_class = tp_per_class / predicted_positives_per_class
    precision_per_class = np.nan_to_num(precision_per_class)  # Handle division by zero

    # Calculate recall per class
    recall_per_class = tp_per_class / actual_positives_per_class
    recall_per_class = np.nan_to_num(recall_per_class)  # Handle division by zero

    # copied_recall_per_class = np.copy(recall_per_class)
    # copied_precision_per_class= np.copy(precision_per_class)

    precision_recall = precision_per_class + recall_per_class

    precision_recall[precision_recall == 0] = 1

    # recall_per_class[recall_per_class == 0] = 1
    # recall_per_class[recall_per_class == 0] = 1

    # Calculate F1-score per class
    f1_score_per_class = 2 * (precision_per_class * recall_per_class) / precision_recall
    f1_score_per_class = np.nan_to_num(f1_score_per_class)  # Handle division by zero

    return accuracy_per_class, precision_per_class, recall_per_class, f1_score_per_class


def print_metric_per_class(cm):
    # Calculate metrics per class
    accuracy_per_class, precision_per_class, recall_per_class, f1_score_per_class = metrics_per_class(cm)

    # Print metrics per class
    for i in range(len(accuracy_per_class)):
        print(f"Class {i}:")
        print("Accuracy:", accuracy_per_class[i])
        print("Precision:", precision_per_class[i])
        print("Recall:", recall_per_class[i])
        print("F1-score:", f1_score_per_class[i])
        print()


def initialize_weights_xavier(input_size, output_size):
    variance = 1 / (input_size + output_size)
    std_dev = torch.sqrt(torch.tensor(variance, dtype=torch.float32))
    weights = torch.randn(input_size, output_size) * std_dev
    return weights.to(dtype=torch.float32)


def binary_cross_entropy_loss(predicted, labels):
    epsilon = 1e-15  # To avoid log(0) issues

    predicted = torch.clamp(predicted, epsilon, 1 - epsilon)  # Clamp predictions to avoid log(0)

    # Compute the binary cross-entropy loss manually
    loss = -torch.mean(labels * torch.log(predicted) + (1 - labels) * torch.log(1 - predicted))

    # Calculate gradients with respect to the loss (difference between predicted and labels)
    grad = predicted - labels

    return loss, grad

def categorical_cross_entropy_loss(predicted, labels):
    epsilon = 1e-15  # To avoid log(0) issues
    predicted = torch.clamp(predicted, epsilon, 1 - epsilon)  # Clamp predictions to avoid log(0)

    # Ensure labels are of type LongTensor for one-hot encoding
    labels = labels.long()

    # Convert labels to one-hot encoding
    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=predicted.shape[1]).float()

    # Calculate categorical cross-entropy loss
    loss = -torch.mean(torch.sum(labels_one_hot * torch.log(predicted), dim=1))

    # Compute the gradient (difference between predicted and labels_one_hot)
    grad = predicted - labels_one_hot

    return loss, grad

def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)  # Assumes x is the sigmoid output
    return 1 / (1 + torch.exp(-x))

def softmax(x, derivative=False):

    if derivative:
        return torch.full(x.shape, 1)

    exp_vals = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)
    s = exp_vals / torch.sum(exp_vals, dim=1, keepdim=True)
    return s