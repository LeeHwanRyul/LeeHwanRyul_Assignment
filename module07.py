import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from mnist_net import MnistNet


MODEL_NAME = 'Lee_mnist_model.pkl'
IMAGE_DIR = 'images'


def preprocess_image(file_path):
    image = Image.open(file_path).convert('L')

    # 흰 종이에 검은 글씨로 쓴 경우 MNIST 형태로 반전한다.
    if np.array(image).mean() > 127:
        image = ImageOps.invert(image)

    # 글자가 있는 부분만 잘라낸 뒤 20x20 안에 맞춘다.
    bbox = image.getbbox()
    if bbox is not None:
        image = image.crop(bbox)

    image.thumbnail((20, 20))
    canvas = Image.new('L', (28, 28), 0)
    left = (28 - image.width) // 2
    top = (28 - image.height) // 2
    canvas.paste(image, (left, top))

    x = np.array(canvas, dtype=np.float32) / 255.0
    return x.reshape(1, 784), canvas


def test_handwritten_images(net):
    results = []

    for digit in range(10):
        for number in range(1, 6):
            file_path = os.path.join(IMAGE_DIR, f'{digit}_{number}.png')

            if not os.path.exists(file_path):
                print('Missing image:', file_path)
                continue

            x, image = preprocess_image(file_path)
            score = net.predict(x)
            pred = int(np.argmax(score, axis=1)[0])
            results.append((file_path, digit, pred, image))
            print(f'{file_path}: expected={digit}, predicted={pred}')

    if len(results) == 0:
        print('No handwritten images were found.')
        return

    correct = sum(true == pred for _, true, pred, _ in results)
    print(f'Handwritten image accuracy: {correct}/{len(results)} = {correct / len(results):.4f}')

    # 앞의 10개 결과만 확인한다.
    show_count = min(10, len(results))
    fig, axes = plt.subplots(2, 5, figsize=(9, 4))
    axes = axes.ravel()

    for i in range(10):
        axes[i].axis('off')
        if i < show_count:
            _, true, pred, image = results[i]
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'T:{true}, P:{pred}')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if not os.path.exists(MODEL_NAME):
        raise FileNotFoundError('Run train.py first to create the model file.')

    with open(MODEL_NAME, 'rb') as f:
        saved_params = pickle.load(f)

    net = MnistNet(hidden1_size=100, hidden2_size=50)
    for key in net.params.keys():
        net.params[key][...] = saved_params[key]

    test_handwritten_images(net)
