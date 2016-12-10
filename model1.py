# モデル1
#
# 中間層1層
#

import time
from datetime import datetime
from load_data import load
from saver import save_arch
from plotter import plot_hist, plot_model_arch

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

# 変数
model_name = 'model1'
nb_epoch = 100
validation_split = 0.2
lr = 0.01
momentum = 0.9
nesterov = True
loss_method = 'mean_squared_error'
arch_path = 'model/' + model_name + '-arch-' + str(nb_epoch) + '.json'
weights_path = 'model/' + model_name + '-weights-' + str(nb_epoch) + '.hdf5'

# データ読み込み
X, y = load()

# モデル
model = Sequential()
model.add(Dense(100, input_dim=9216))
model.add(Activation('relu'))
model.add(Dense(30))

save_arch(model, arch_path) # モデルを保存しておく

# トレーニングの準備
checkpoint_collback = ModelCheckpoint(filepath = weights_path,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_best_only=True,
                                      mode='auto')

sgd = SGD(lr=lr, momentum=momentum, nesterov=nesterov)
model.compile(loss=loss_method, optimizer=sgd)

# トレーニング
start_time = time.time()
print('start_time: %s' % (datetime.now()))
hist = model.fit(X, y, nb_epoch=nb_epoch, validation_split=validation_split, callbacks=[checkpoint_collback])
print('end_time: %s, duracion(min): %d' % (datetime.now(), int(time.time()-start_time) / 60))

# プロットしてファイルとして保存する
plot_hist(hist, model_name)
plot_model_arch(model, model_name)
