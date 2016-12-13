import os

import matplotlib.pyplot as plt

def plot_hist(history, model_name=None):
  plt.plot(history['loss'], linewidth=3, label='train')
  plt.plot(history['val_loss'], linewidth=3, label='valid')
  plt.grid()
  plt.legend()
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.ylim(1e-4, 1e-2)
  plt.yscale('log')
  if model_name:
    path = os.path.join('images', model_name + '-loss.png')
    plt.savefig(path)
  else:
    plt.show()


def plot_model_arch(model, model_name):
  from keras.utils.visualize_util import plot
  path = os.path.join('images', model_name + '.png')
  plot(model, to_file=path, show_shapes=True)


def plot_samples(X, y):
  fig = plt.figure(figsize=(6, 6))
  fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
  for i in range(16):
    axis = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    _plot_sample(X[i], y[i], axis)
  plt.show()

def _plot_sample(x, y, axis):
  img = x.reshape(96, 96)
  axis.imshow(img, cmap='gray')
  axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)
