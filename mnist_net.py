from collections import OrderedDict
import numpy as np

from layer import Layer


INPUT_SIZE = 784
OUTPUT_SIZE = 10


class MnistNet:
    def __init__(self, hidden1_size=100, hidden2_size=50):
        self.params = {}

        # ReLU를 사용하므로 He initialization을 사용했다.
        w1_scale = np.sqrt(2.0 / INPUT_SIZE)
        w2_scale = np.sqrt(2.0 / hidden1_size)
        w3_scale = np.sqrt(2.0 / hidden2_size)

        self.params['w1'] = w1_scale * np.random.randn(INPUT_SIZE, hidden1_size)
        self.params['b1'] = np.zeros(hidden1_size)
        self.params['w2'] = w2_scale * np.random.randn(hidden1_size, hidden2_size)
        self.params['b2'] = np.zeros(hidden2_size)
        self.params['w3'] = w3_scale * np.random.randn(hidden2_size, OUTPUT_SIZE)
        self.params['b3'] = np.zeros(OUTPUT_SIZE)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Layer.Affine(self.params['w1'], self.params['b1'])
        self.layers['Relu1'] = Layer.Relu()
        self.layers['Affine2'] = Layer.Affine(self.params['w2'], self.params['b2'])
        self.layers['Relu2'] = Layer.Relu()
        self.layers['Affine3'] = Layer.Affine(self.params['w3'], self.params['b3'])

        self.last_layer = Layer.SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, y):
        y_hat = self.predict(x)
        return self.last_layer.forward(y_hat, y)

    def accuracy(self, x, y):
        y_hat = self.predict(x)
        y_pred = np.argmax(y_hat, axis=1)

        if y.ndim == 2:
            y_true = np.argmax(y, axis=1)
        else:
            y_true = y

        return np.mean(y_pred == y_true)

    def gradient(self, x, y):
        self.loss(x, y)

        dout = self.last_layer.backward(1)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['w1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['w2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db
        grads['w3'] = self.layers['Affine3'].dw
        grads['b3'] = self.layers['Affine3'].db

        return grads
