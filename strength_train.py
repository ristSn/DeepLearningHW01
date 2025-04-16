import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
    pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# 将train_imgs和valid_imgs转化为[batch_size, channel, height, width]的格式
train_imgs = train_imgs.reshape(-1, 1, 28, 28)
valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

# 将各个类别的样本划分开来
train_imgs_classes = []
valid_imgs_classes = []
for i in range(10):
    idx = np.where(train_labs == i)[0]
    train_imgs_classes.append(train_imgs[idx])

    idx = np.where(valid_labs == i)[0]
    valid_imgs_classes.append(valid_imgs[idx])

# 统计各个类别的样本数量
crt_classes_count = [len(train_imgs_classes[i]) for i in range(10)]
train_size = 50000

# 网络结构
model = nn.models.Model_CNN_v2_1()

optimizer = nn.optimizer.MomentGD(init_lr=1e-2, model=model, mu=0.9)
scheduler = nn.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.6)
loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

for _ in range(10):
    runner.train([train_imgs, train_labs], [valid_imgs, valid_labs],
                num_epochs=2, log_iters=100, save_dir=r'./saved_models/model_v2_1_1')

    wrong_pred_samples_indices = []
    for i in range(10):
        train_class_imgs = train_imgs_classes[i]
        pred = model(train_class_imgs)
        pred_labels = np.argmax(pred, axis=1)
        wrong_idx = np.where(pred_labels != i)[0]
        wrong_pred_samples_indices.append(wrong_idx)
        print(f'class {i} has {len(wrong_idx)} wrong pred samples')

    # 重采样
    add_new_samples_num = 100
    for i in range(10):
        train_imgs_class = train_imgs_classes[i]
        wrong_idx = wrong_pred_samples_indices[i]
        if len(wrong_idx) == 0:
            continue
        add_new_idx = np.random.choice(wrong_idx, add_new_samples_num, replace=True)
        new_imgs = train_imgs_class[add_new_idx]
        new_labs = np.ones(add_new_samples_num, dtype=np.int32)*i

        train_imgs = np.concatenate([train_imgs, new_imgs], axis=0)
        train_labs = np.concatenate([train_labs, new_labs], axis=0)

    crt_sample_size = len(train_imgs)
    print(f"current train set size: {crt_sample_size}")

    # 把训练集重新打乱
    idx = np.random.permutation(np.arange(len(train_imgs)))
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]

