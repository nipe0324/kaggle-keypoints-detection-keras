# モデル8
#
# [WIP] 各データ数に応じてモデルの特化
# ref: https://elix-tech.github.io/ja/2016/06/02/kaggle-facial-keypoints-ja.html
#

import time
from datetime import datetime
from load_data import load2d
from saver import save_arch, save_history
from plotter import plot_hist, plot_model_arch
import pickle

import numpy as np
from collections import OrderedDict
# This module will be removed in 0.20.
from sklearn.cross_validation import train_test_split

from data_generator import FlippedImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping


# 定数
SPECIALIST_SETTINGS = [
    dict(
        id='1',
        columns=(
            'left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x', 'right_eye_center_y',
            ),
        flip_indices=((0, 2), (1, 3)),
        ),

    dict(
        id='2',
        columns=(
            'nose_tip_x', 'nose_tip_y',
            ),
        flip_indices=(),
        ),

    dict(
        id='3',
        columns=(
            'mouth_left_corner_x', 'mouth_left_corner_y',
            'mouth_right_corner_x', 'mouth_right_corner_y',
            'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
            ),
        flip_indices=((0, 2), (1, 3)),
        ),

    dict(
        id='4',
        columns=(
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y',
            ),
        flip_indices=(),
        ),

    dict(
        id='5',
        columns=(
            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
            ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),

    dict(
        id='6',
        columns=(
            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
            ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),
    ]

# 変数
model_name = 'model8'
nb_epoch = 2 # 5000
validation_split = 0.2
lr = 0.01
start = 0.03
stop = 0.001
learning_rates = np.linspace(start, stop, nb_epoch)
patience = 100 # EarlyStoppingでn回連続でエラーの最小値が更新されなかったらストップさせる
momentum = 0.9
nesterov = True
loss_method = 'mean_squared_error'


# データ読み込み
X, y = load2d()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)


# モデル定義
def fit_specialists():
    model7 = load_arch('model/model7-arch-5000.json')
    specialists = OrderedDict()

    for setting in SPECIALIST_SETTINGS:
        # 特定のカラムのデータの取得
        cols = setting['columns']
        X, y = load2d(cols=cols)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = model_from_json(model7.to_json()) # アーキテクチャのみを取り出す
        model.layers.pop() # 出力層を取り除く
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        model.add(Dense(len(cols))) # 新しい出力層を追加

        save_arch(model, arch_path) # モデルを保存しておく

        # トレーニングの準備
        sgd = SGD(lr=start, momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        plot(model, to_file="model_{}.png".format(cols[0]), show_shapes=True)

        flipgen = FlippedImageDataGenerator()
        flipgen.flip_indices = setting['flip_indices']
        early_stop = EarlyStopping(patience=100)
        learning_rates = np.linspace(start, stop, nb_epoch)
        change_lr = LearningRateScheduler(lambda epoch: float(learning_rates[epoch]))

        print("Training model for columns {} for {} epochs".format(cols, nb_epoch))

        hist8 = model.fit_generator(flipgen.flow(X_train, y_train),
                                     samples_per_epoch=X_train.shape[0],
                                     nb_epoch=nb_epoch,
                                     validation_data=(X_val, y_val),
                                     callbacks=[change_lr, early_stop])

        save_arch(model, arch_path) # モデルを保存しておく
        specialists[cols] = model

        # # 保存
        # save(model, 'model')

        # # プロット
        # plot_hist(hist7)


fit_specialists()

