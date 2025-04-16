
# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# model = nn.models.Model_MLP()
model = nn.models.Model_CNN_v2_1()
model.load_model(r'.\saved_models\model_v2_1_1\best_model.pickle')

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()
test_imgs = test_imgs.reshape(-1, 1, 28, 28)

confusion_matrix = np.zeros((10, 10))

for i in range(10):
    idx = np.where(test_labs == i)[0]
    imgs = test_imgs[idx]
    labels = test_labs[idx]

    num = len(imgs)

    pred = model(imgs)
    pred_labels = np.argmax(pred, axis=1)
    for j in range(10):
        confusion_matrix[i][j] = np.sum(pred_labels == j)
    confusion_matrix[i][i] = 0
    crt = np.sum(pred_labels == labels)
    wrg = num - crt
    acc = crt / num * 100
    print(f'The {i}-th class has {num} samples, {crt} correct predictions, {wrg} wrong predictions, and accuracy of {acc:.2f}%.')

# 绘制混淆矩阵并在图上标注数量
plt.figure(figsize=(10, 10))
plt.imshow(confusion_matrix, cmap='Blues')
for i in range(10):
    for j in range(10):
        plt.text(j, i, int(confusion_matrix[i][j]), ha='center', va='center', color='black')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion matrix')
plt.colorbar()
plt.show()