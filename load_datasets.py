import numpy as np
import random

def load_iris_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples qui vont etre attribués à l'entrainement,
        le reste des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisés
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    
    # Vous pouvez utiliser des valeurs numériques pour les différents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
    
    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/bezdekIris.data', 'r')

    dataset = f.readlines()
    f.close()

    # Mélanger les exemples du dataset
    random.shuffle(dataset)

    data = []
    labels = []

    for line in dataset:
        if line.strip() != "":
            instance = line.strip().split(',')
            data.append([float(x) for x in instance[:4]])
            labels.append(conversion_labels[instance[4]])

    # Convertir en numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Séparer les données en train et test
    train_size = int(len(data) * train_ratio)
    train = data[:train_size]
    train_labels = labels[:train_size]
    test = data[train_size:]
    test_labels = labels[train_size:]

    return (train, train_labels, test, test_labels)
	
	
	
def load_wine_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Binary Wine quality

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/binary-winequality-white.csv', 'r')

    dataset = f.readlines()
    f.close()

    data = []
    labels = []

    for line in dataset:
        if line.strip() != "":
            instance = line.strip().split(',')
            data.append([float(x) for x in instance[:-1]])
            labels.append(int(instance[-1]))

    # Convertir en numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Mélanger les exemples du dataset
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data[:], labels[:] = zip(*combined)

    # Séparer les données en train et test
    train_size = int(len(data) * train_ratio)
    train = data[:train_size]
    train_labels = labels[:train_size]
    test = data[train_size:]
    test_labels = labels[train_size:]

    return (train, train_labels, test, test_labels)

def load_abalone_dataset(train_ratio):
    """
    Cette fonction a pour but de lire le dataset Abalone-intervalles

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    f = open('datasets/abalone-intervalles.csv', 'r') # La fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.
    dataset = f.readlines()
    f.close()

    data = []
    labels = []

    for line in dataset:
        if line.strip() != "":
            instance = line.strip().split(',')
            row = [instance[0]]
            row.extend([float(x) for x in instance[1:len(instance) - 1]])
            data.append(row)
            labels.append(float(instance[-1]))

    # Convertir en numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Mélanger les exemples du dataset
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data[:], labels[:] = zip(*combined)

    # Séparer les données en train et test
    train_size = int(len(data) * train_ratio)
    train = data[:train_size]
    train_labels = labels[:train_size]
    test = data[train_size:]
    test_labels = labels[train_size:]

    return (train, train_labels, test, test_labels)