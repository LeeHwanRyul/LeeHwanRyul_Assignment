import gzip
import os
import struct
import urllib.request

import numpy as np


class MnistData:
    URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    FALLBACK_URL = 'https://raw.githubusercontent.com/fgnt/mnist/master/'

    FILES = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }

    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _download(self, file_name):
        file_path = os.path.join(self.data_dir, file_name)
        if os.path.exists(file_path):
            return file_path

        print('Downloading', file_name)
        try:
            urllib.request.urlretrieve(self.URL + file_name, file_path)
        except Exception:
            print('Trying another download address...')
            urllib.request.urlretrieve(self.FALLBACK_URL + file_name, file_path)
        return file_path

    def _load_images(self, file_name):
        file_path = self._download(file_name)

        with gzip.open(file_path, 'rb') as f:
            magic, number, rows, cols = struct.unpack('>IIII', f.read(16))
            if magic != 2051:
                raise ValueError('Image file format is not correct.')
            images = np.frombuffer(f.read(), dtype=np.uint8)

        return images.reshape(number, rows * cols)

    def _load_labels(self, file_name):
        file_path = self._download(file_name)

        with gzip.open(file_path, 'rb') as f:
            magic, number = struct.unpack('>II', f.read(8))
            if magic != 2049:
                raise ValueError('Label file format is not correct.')
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        return labels.reshape(number)

    def _one_hot(self, labels):
        one_hot = np.zeros((labels.size, 10))
        one_hot[np.arange(labels.size), labels] = 1
        return one_hot

    def load_data(self, normalize=True, one_hot_label=True):
        x_train = self._load_images(self.FILES['train_images'])
        y_train = self._load_labels(self.FILES['train_labels'])
        x_test = self._load_images(self.FILES['test_images'])
        y_test = self._load_labels(self.FILES['test_labels'])

        if normalize:
            x_train = x_train.astype(np.float32) / 255.0
            x_test = x_test.astype(np.float32) / 255.0

        if one_hot_label:
            y_train = self._one_hot(y_train)
            y_test = self._one_hot(y_test)

        return x_train, y_train, x_test, y_test
