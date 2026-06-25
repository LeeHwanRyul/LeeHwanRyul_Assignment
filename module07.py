import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mnist_net import MnistNet


MODEL_NAME = 'Lee_mnist_model.pkl'
IMAGE_DIR = 'images'


def _shift_without_wrap(image_array, shift_y, shift_x):
    """Move the digit without wrapping pixels around the opposite edge."""
    shifted = np.zeros_like(image_array)

    src_y1 = max(0, -shift_y)
    src_y2 = min(image_array.shape[0], image_array.shape[0] - shift_y)
    src_x1 = max(0, -shift_x)
    src_x2 = min(image_array.shape[1], image_array.shape[1] - shift_x)

    dst_y1 = max(0, shift_y)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x1 = max(0, shift_x)
    dst_x2 = dst_x1 + (src_x2 - src_x1)

    if src_y2 > src_y1 and src_x2 > src_x1:
        shifted[dst_y1:dst_y2, dst_x1:dst_x2] = image_array[src_y1:src_y2, src_x1:src_x2]

    return shifted


def preprocess_image(file_path):
    image = Image.open(file_path).convert('L')
    image_array = np.asarray(image, dtype=np.uint8)

    # Decide the background color using the outer border of the image.
    border = np.concatenate([
        image_array[0, :], image_array[-1, :],
        image_array[:, 0], image_array[:, -1]
    ])

    # MNIST uses a black background and a bright digit.
    if border.mean() > 127:
        image_array = 255 - image_array

    # Remove weak background noise. Small noise pixels made getbbox() include
    # nearly the entire original image, so the digit became too small.
    image_array = image_array.copy()
    image_array[image_array < 35] = 0

    points = np.argwhere(image_array > 0)
    if points.size == 0:
        raise ValueError(f'No digit was found in {file_path}')

    y_min, x_min = points.min(axis=0)
    y_max, x_max = points.max(axis=0) + 1
    digit = image_array[y_min:y_max, x_min:x_max]

    # Increase contrast after cropping.
    if digit.max() > 0:
        digit = digit.astype(np.float32)
        digit = digit / digit.max() * 255.0
        digit = digit.astype(np.uint8)

    digit_image = Image.fromarray(digit, mode='L')

    # Fit the digit inside a 20 x 20 area, similar to MNIST.
    width, height = digit_image.size
    scale = 20.0 / max(width, height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    digit_image = digit_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    left = (28 - new_width) // 2
    top = (28 - new_height) // 2
    canvas[top:top + new_height, left:left + new_width] = np.asarray(digit_image)

    # Move the brightness center to the center used by MNIST images.
    total = canvas.sum()
    if total > 0:
        yy, xx = np.indices(canvas.shape)
        center_y = (yy * canvas).sum() / total
        center_x = (xx * canvas).sum() / total
        shift_y = int(round(13.5 - center_y))
        shift_x = int(round(13.5 - center_x))
        canvas = _shift_without_wrap(canvas, shift_y, shift_x)

    x = canvas.astype(np.float32) / 255.0
    return x.reshape(1, 784), Image.fromarray(canvas)


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
    print(
        f'Handwritten image accuracy: '
        f'{correct}/{len(results)} = {correct / len(results):.4f}'
    )

    show_count = min(10, len(results))
    fig, axes = plt.subplots(2, 5, figsize=(9, 4))
    axes = axes.ravel()

    for i in range(10):
        axes[i].axis('off')
        if i < show_count:
            _, true, pred, image = results[i]
            axes[i].imshow(image, cmap='gray', vmin=0, vmax=255)
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
