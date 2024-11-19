import torch

import scrachnnet.NeuralNet as NeuralNet
import scrachnnet.helpers as helpers


def main():

    hidden_size = 5
    num_layer = 2
    activation_function = helpers.sigmoid
    output_activation_function = helpers.sigmoid
    learning_rate = 1e-3
    epochs = 1000
    hidden_size_range = (10, 50, 100, 300)
    depths = (2, 5, 7, 10)
    dataset_names = ['iris', 'wine', 'abalone']
    train_size = .7

    datasets = helpers.dataset_loader(dataset_names, train_ratio=train_size)
    ds_names = "iris"
    # ds_names = "abalone"
    dataset = datasets[ds_names]

    train, train_labels, test, test_labels, labels = dataset

    input_size = train.shape[1]
    output_size = 1 if len(labels) == 2 else len(labels)
    loss_function = helpers.binary_cross_entropy_loss if output_size == 1 else helpers.categorical_cross_entropy_loss
    mlp = NeuralNet.NeuralNet(input_size, hidden_size, output_size, activation_function, n_layer=num_layer, dataset=ds_names)

    # print(mlp)

    train, train_labels = mlp.fit_transform(train.astype(float), train_labels.astype(int))
    test, test_labels = mlp.transform(test.astype(float), test_labels.astype(int))

    X_train, X_test, y_train, y_test = helpers.train_test_split(train, train_labels, test_size=0.2, random_state=42)

    # Pytorch
    pt_model  = mlp.as_pytorch()
    pt_losses, pt_train_labels, pt_train_predicted, pt_test_predicted, pt_training_time = mlp.pytorch_train(pt_model, X_train, y_train, X_test, loss_function, learning_rate=learning_rate, epochs=epochs)

    losses, train_labels, train_predicted, test_predicted, training_time = mlp.train(X_train, y_train, X_test, loss_function, learning_rate=learning_rate, epochs=epochs)

    train_accuracy, train_f1_score = [], []
    test_accuracy, test_f1_score = [], []
    for label, predicted, test_pred in zip(train_labels, train_predicted, test_predicted):
        train_accuracy.append(helpers.accuracy_score(label, predicted.numpy()))
        test_accuracy.append(helpers.accuracy_score(y_test.numpy(), test_pred.numpy()))
        train_f1_score.append(helpers.f1_score(label, predicted.numpy()))
        test_f1_score.append(helpers.f1_score(y_test.numpy(), test_pred.numpy()))

    # Pytorch
    pt_train_accuracy, pt_train_f1_score = [], []
    pt_test_accuracy, pt_test_f1_score = [], []
    for label, predicted, test_pred in zip(pt_train_labels, pt_train_predicted, pt_test_predicted):
        pt_train_accuracy.append(helpers.accuracy_score(label, predicted.numpy()))
        pt_test_accuracy.append(helpers.accuracy_score(y_test.numpy(), test_pred.numpy()))
        pt_train_f1_score.append(helpers.f1_score(label, predicted.numpy()))
        pt_test_f1_score.append(helpers.f1_score(y_test.numpy(), test_pred.numpy()))

    helpers.plot_train_test(range(epochs), (train_accuracy, test_accuracy), title="Accuracy - Train vs Test")
    helpers.plot_train_test(range(epochs), (train_f1_score, test_f1_score), title="F1-score - Train vs Test")

    helpers.plot_train_test(range(epochs), (pt_train_accuracy, pt_test_accuracy), title="Accuracy - Pytorch Train vs Test")
    helpers.plot_train_test(range(epochs), (pt_train_f1_score, pt_test_f1_score), title="F1-score - Pytorch Train vs Test")

    # Plot losses
    helpers.plot(range(epochs), losses, title="Loss - Train", ylabel="Loss")
    helpers.plot(range(epochs), pt_losses, title="Loss - Pytorch Train", ylabel="Loss")

    conf_matrix, accuracy, precision, recall, f1_score, _ = mlp.evaluate(torch.tensor(test, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.float32))
    print("Résultat global sur le test set:")
    print("Accuracy:", accuracy)
    print("Précision:", precision)
    print("Rappel:", recall)
    print("F1-score:", f1_score)
    print("\n\n")

    print("Résultat par class sur le test set:")
    helpers.print_metric_per_class(conf_matrix)
    helpers.plot_confusion_matrix(conf_matrix, labels)

    conf_matrix, accuracy, precision, recall, f1_score, _ = mlp.evaluate(torch.tensor(test, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.float32), model=pt_model)
    print("Pytorch -- Résultat global sur le test set:")
    print("Accuracy:", accuracy)
    print("Précision:", precision)
    print("Rappel:", recall)
    print("F1-score:", f1_score)
    print("\n\n")

    print("Résultat par class sur le test set:")
    helpers.print_metric_per_class(conf_matrix)
    helpers.plot_confusion_matrix(conf_matrix, labels)

if __name__ == '__main__':
    main()