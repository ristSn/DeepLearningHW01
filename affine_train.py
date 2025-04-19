# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(20250405)

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

# 在train_imgs里重采样，并对重采样的样本做随机仿射变换
resample_num = 50000
for _ in range(resample_num):
    if _ % 1000 == 0:
        print(f"sample has been prepared {_} for {resample_num}")
    rand_idx = np.random.randint(0, train_imgs.shape[0])
    img = train_imgs[rand_idx][0]
    label = train_labs[rand_idx]

    r_theta = np.random.normal(0, np.pi/12)

    r_x_shear = np.random.normal(0, 0.2)
    r_y_shear = np.random.normal(0, 0.2)

    Q = np.array([[np.cos(-r_theta), -np.sin(-r_theta)], [np.sin(-r_theta), np.cos(-r_theta)]])

    Sx = np.array([[1, -r_x_shear], [0, 1]])
    Sy = np.array([[1, 0], [-r_y_shear, 1]])


    img_new = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            idx0 = np.array([i-img.shape[0]//2, j-img.shape[1]//2])
            i_init, j_init = np.dot(Sx, np.dot(Sy, np.dot(Q, idx0))) + np.array([img.shape[0]//2, img.shape[1]//2])
            i_down, j_down = int(i_init), int(j_init)
            i_f = i_init - i_down
            j_f = j_init - j_down
            if i_down >= 0 and i_down < img.shape[0]-1 and j_down >= 0 and j_down < img.shape[1]-1:
                img_new[i][j] = (1-i_f)*(1-j_f)*img[i_down][j_down] \
                                + (1-i_f)*j_f*img[i_down][j_down+1] \
                                + i_f*(1-j_f)*img[i_down+1][j_down] \
                                + i_f*j_f*img[i_down+1][j_down+1]

    train_imgs = np.concatenate((train_imgs, img_new.reshape(1, 1, 28, 28)), axis=0)
    train_labs = np.concatenate((train_labs, np.array([label])))

# 保存一下
# np.save(r'./dataset/affine_imgs/train_imgs.npy', train_imgs)
# np.save(r'./dataset/affine_imgs/train_labs.npy', train_labs)
# np.save(r'./dataset/affine_imgs/valid_imgs.npy', valid_imgs)
# np.save(r'./dataset/affine_imgs/valid_labs.npy', valid_labs)


# train_imgs = np.load(r'./dataset/affine_imgs/train_imgs.npy')
# train_labs = np.load(r'./dataset/affine_imgs/train_labs.npy')
# valid_imgs = np.load(r'./dataset/affine_imgs/valid_imgs.npy')
# valid_labs = np.load(r'./dataset/affine_imgs/valid_labs.npy')

idx = np.random.permutation(np.arange(train_imgs.shape[0]))
# save the index.
with open('idx_affine.pickle', 'wb') as f:
    pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]

for i in range(10):
    count = np.sum(train_labs == i)
    print(f"class {i} has {count} samples in training set.")

for i in range(100):
    img = train_imgs[i][0]
    label = train_labs[i]
    plt.subplot(10, 10, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(label)
    plt.axis('off')
plt.show()

# linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])              6666
# optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
# scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
# loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)
#
# runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
#
# runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'./best_models')

model = nn.models.Model_CNN_v2_1()

optimizer = nn.optimizer.MomentGD(init_lr=1e-2, model=model, mu=0.9)
scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=1000, gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
print("start training...")
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs],
             num_epochs=5, log_iters=100, save_dir=r'./saved_models/model_affine_v2_2')
print("training done.")

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()