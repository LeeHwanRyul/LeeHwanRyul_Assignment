import pickle
import numpy as np

from mnist_data import MnistData
from mnist_net import MnistNet


# hyperparameters
EPOCH_NUM = 3
BATCH_SIZE = 100
LEARNING_RATE = 0.1
MODEL_NAME = 'Lee_mnist_model.pkl'


def train(net, x_train, y_train, x_test, y_test):
    train_size = x_train.shape[0]

    for epoch in range(EPOCH_NUM):
        indices = np.random.permutation(train_size)
        x_train = x_train[indices]
        y_train = y_train[indices]

        losses = []

        for i in range(0, train_size, BATCH_SIZE):
            x_batch = x_train[i:i + BATCH_SIZE]
            y_batch = y_train[i:i + BATCH_SIZE]

            grads = net.gradient(x_batch, y_batch)
            losses.append(net.last_layer.loss)

            for key in net.params.keys():
                net.params[key] -= LEARNING_RATE * grads[key]

        train_acc = net.accuracy(x_train, y_train)
        test_acc = net.accuracy(x_test, y_test)
        avg_loss = np.mean(losses)

        print(
            f'Epoch {epoch + 1}/{EPOCH_NUM}, '
            f'Loss: {avg_loss:.4f}, '
            f'Train Accuracy: {train_acc:.4f}, '
            f'Test Accuracy: {test_acc:.4f}'
        )


if __name__ == '__main__':
    np.random.seed(1)

    mnist = MnistData()
    x_train, y_train, x_test, y_test = mnist.load_data()

    net = MnistNet(hidden1_size=100, hidden2_size=50)
    train(net, x_train, y_train, x_test, y_test)

    with open(MODEL_NAME, 'wb') as f:
        pickle.dump(net.params, f)

    print('Model saved:', MODEL_NAME)
