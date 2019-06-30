# coding=utf-8
import os

import pandas as pd
import numpy as np

DEBUG = True

# reference
# https://www.kaggle.com/corochann/deep-learning-cnn-with-chainer-lb-0-99700

# Load data
print('Loading data...')

# CSV を読み込み
train = pd.read_csv(os.path.join('./train.csv'))
test = pd.read_csv(os.path.join('./test.csv'))

# train 行を全て取得、列を1列以降全て取得（画像データ）
train_x = train.iloc[:, 1:].values.astype('float32')

# train 行を全て取得、列を1列以降全て取得（数字のデータ）
train_y = train.iloc[:, 0].values.astype('int32')

# test 行列全てのデータを取得（画像データー）
test_x = test.values.astype('float32')


# データ数 42000, 各データのピクセル数 784
print('train_x', train_x.shape)
# train_x (42000, 784)

# 答えのデータ
print('train_y', train_y.shape)
# train_y (42000,)

# データ数 28000, 各データのピクセル数 784
print('test_x', test_x.shape)
# test_x (28000, 784)

# reshape and rescale value
# 28 * 28 の配列に変換する
# 255 で割る事で値を 0-1.0 の値に変換
train_imgs = train_x.reshape((-1, 1, 28, 28)) / 255.
test_imgs = test_x.reshape((-1, 1, 28, 28)) / 255.
print('train_imgs', train_imgs.shape, 'test_imgs', test_imgs.shape)
# train_imgs (42000, 1, 28, 28) test_imgs (28000, 1, 28, 28)

# %matplotlib inline
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

# 画像を表示する関数
# matplot で表示する
def show_image(img):
    plt.figure(figsize=(1.5, 1.5))
    plt.axis('off')
    if img.ndim == 3:
        img = img[0, :, :]
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()

# 画像を表示する
# 0 番目は 1
# 1 番目は 0
# 2 番目は 1
# 3 番目は 4
# といった感じ

# print('index0, label {}'.format(train_y[0]))
# show_image(train_imgs[0])
# print('index1, label {}'.format(train_y[1]))
# show_image(train_imgs[1])
# show_image(train_imgs[2])
# show_image(train_imgs[3])


#
#
# import chainer
# import chainer.links as L
# import chainer.functions as F
# from chainer.dataset.convert import concat_examples
#
#
# class CNNMedium(chainer.Chain):
#     def __init__(self, n_out):
#         super(CNNMedium, self).__init__()
#         with self.init_scope():
#             self.conv1 = L.Convolution2D(None, 16, 3, 1)
#             self.conv2 = L.Convolution2D(16, 32, 3, 1)
#             self.conv3 = L.Convolution2D(32, 32, 3, 1)
#             self.conv4 = L.Convolution2D(32, 32, 3, 2)
#             self.conv5 = L.Convolution2D(32, 64, 3, 1)
#             self.conv6 = L.Convolution2D(64, 32, 3, 1)
#             self.fc7 = L.Linear(None, 30)
#             self.fc8 = L.Linear(30, n_out)
#
#     def __call__(self, x):
#         h = F.leaky_relu(self.conv1(x), slope=0.05)
#         h = F.leaky_relu(self.conv2(h), slope=0.05)
#         h = F.leaky_relu(self.conv3(h), slope=0.05)
#         h = F.leaky_relu(self.conv4(h), slope=0.05)
#         h = F.leaky_relu(self.conv5(h), slope=0.05)
#         h = F.leaky_relu(self.conv6(h), slope=0.05)
#         h = F.leaky_relu(self.fc7(h), slope=0.05)
#         h = self.fc8(h)
#         return h
#
#     def _predict_batch(self, x_batch):
#         with chainer.no_backprop_mode(), chainer.using_config('train', False):
#             h = self.__call__(x_batch)
#             return F.softmax(h)
#
#     def predict_proba(self, x, batchsize=32, device=-1):
#         if device >= 0:
#             chainer.cuda.get_device_from_id(device).use()
#             self.to_gpu()  # Copy the model to the GPU
#
#         y_list = []
#         for i in range(0, len(x), batchsize):
#             x_batch = concat_examples(x[i:i + batchsize], device=device)
#             y = self._predict_batch(x_batch)
#             y_list.append(chainer.cuda.to_cpu(y.data))
#         y_array = np.concatenate(y_list, axis=0)
#         return y_array
#
#     def predict(self, x, batchsize=32, device=-1):
#         proba = self.predict_proba(x, batchsize=batchsize, device=device)
#         return np.argmax(proba, axis=1)
#
#
# if DEBUG:
#     print('DEBUG mode, reduce training data...')
#     # Use only first 1000 example to reduce training time
#     train_x = train_x[:1000]
#     train_imgs = train_imgs[:1000]
#     train_y = train_y[:1000]
# else:
#     print('No DEBUG mode')
#
# from chainer import iterators, training, optimizers, serializers
# from chainer.datasets import TupleDataset
# from chainer.training import extensions
#
# # -1 indicates to use CPU,
# # positive value indicates GPU device id.
# device = -1  # If you use CPU.
# # device = 0  # If you use GPU. (You need to install chainer & cupy with CUDA/cudnn installed)
# batchsize = 16
# class_num = 10
# out_dir = '.'
# if DEBUG:
#     epoch = 5  # This value is small. Change to more than 20 for Actual running.
# else:
#     epoch = 20
#
#
# def train_main(train_x, train_y, val_x, val_y, model_path='cnn_model.npz'):
#     # 1. Setup model
#     model = CNNMedium(n_out=class_num)
#     classifier_model = L.Classifier(model)
#     if device >= 0:
#         chainer.cuda.get_device(device).use()  # Make a specified GPU current
#         classifier_model.to_gpu()  # Copy the model to the GPU
#
#     # 2. Setup an optimizer
#     optimizer = optimizers.Adam()
#     # optimizer = optimizers.MomentumSGD(lr=0.001)
#     optimizer.setup(classifier_model)
#
#     # 3. Load the dataset
#     # train_data = MNISTTrainImageDataset()
#     train_dataset = TupleDataset(train_x, train_y)
#     val_dataset = TupleDataset(val_x, val_y)
#
#     # 4. Setup an Iterator
#     train_iter = iterators.SerialIterator(train_dataset, batchsize)
#     # train_iter = iterators.MultiprocessIterator(train, args.batchsize, n_prefetch=10)
#     val_iter = iterators.SerialIterator(val_dataset, batchsize, repeat=False, shuffle=False)
#
#     # 5. Setup an Updater
#     updater = training.StandardUpdater(train_iter, optimizer,
#                                        device=device)
#     # 6. Setup a trainer (and extensions)
#     trainer = training.Trainer(updater, (epoch, 'epoch'), out=out_dir)
#
#     # Evaluate the model with the test dataset for each epoch
#     trainer.extend(extensions.Evaluator(val_iter, classifier_model, device=device), trigger=(1, 'epoch'))
#
#     trainer.extend(extensions.dump_graph('main/loss'))
#     trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
#     trainer.extend(extensions.LogReport())
#     trainer.extend(extensions.PlotReport(
#         ['main/loss', 'validation/main/loss'],
#         x_key='epoch', file_name='loss.png'))
#     trainer.extend(extensions.PlotReport(
#         ['main/accuracy', 'validation/main/accuracy'],
#         x_key='epoch',
#         file_name='accuracy.png'))
#
#     try:
#         # Use extension library, chaineripy's PrintReport & ProgressBar
#         from chaineripy.extensions import PrintReport, ProgressBar
#         trainer.extend(ProgressBar(update_interval=5))
#         trainer.extend(PrintReport(
#             ['epoch', 'main/loss', 'validation/main/loss',
#              'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
#
#     except:
#         print('chaineripy is not installed, run `pip install chaineripy` to show rich UI progressbar')
#         # Use chainer's original ProgressBar & PrintReport
#         # trainer.extend(extensions.ProgressBar(update_interval=5))
#         trainer.extend(extensions.PrintReport(
#             ['epoch', 'main/loss', 'validation/main/loss',
#              'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
#
#     # Resume from a snapshot
#     # serializers.load_npz(args.resume, trainer)
#
#     # Run the training
#     trainer.run()
#     # Save the model
#     serializers.save_npz('{}/{}'
#                          .format(out_dir, model_path), model)
#     return model
#
#
# import numpy as np
# from sklearn.model_selection import train_test_split
#
# seed = 777
# model_simple = 'cnn_model_simple.npz'
#
# train_idx, val_idx = train_test_split(np.arange(len(train_x)),
#                                       test_size=0.20, random_state=seed)
# print('train size', len(train_idx), 'val size', len(val_idx))
# train_main(train_imgs[train_idx], train_y[train_idx], train_imgs[val_idx], train_y[val_idx], model_path=model_simple)
#
# class_num = 10
#
#
# def predict_main(model_path='cnn_model.npz'):
#     # 1. Setup model
#     model = CNNMedium(n_out=class_num)
#     classifier_model = L.Classifier(model)
#     if device >= 0:
#         chainer.cuda.get_device(device).use()  # Make a specified GPU current
#         classifier_model.to_gpu()  # Copy the model to the GPU
#
#     # load trained model
#     serializers.load_npz(model_path, model)
#
#     # 2. Prepare the dataset --> it's already prepared
#     # test_imgs
#
#     # 3. predict the result
#     t = model.predict(test_imgs, device=device)
#     return t
#
#
# def create_submission(submission_path, t):
#     result_dict = {
#         'ImageId': np.arange(1, len(t) + 1),
#         'Label': t
#     }
#     df = pd.DataFrame(result_dict)
#     df.to_csv(submission_path,
#               index_label=False, index=False)
#     print('submission file saved to {}'.format(submission_path))
#
#
# predict_label = predict_main(model_path=model_simple)
# print('predict_label = ', predict_label, predict_label.shape)
#
# create_submission('submission_simple.csv', predict_label)
