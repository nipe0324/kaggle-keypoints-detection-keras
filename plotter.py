import os

import matplotlib.pyplot as plt

def plot_hist(hist, model_name):
  plt.plot(hist.history['loss'], linewidth=3, label='train')
  plt.plot(hist.history['val_loss'], linewidth=3, label='valid')
  plt.grid()
  plt.legend()
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.ylim(1e-3, 1e-2)
  plt.yscale('log')
  path = os.path.join('images', model_name + '-loss.png')
  plt.savefig(path)

def plot_model_arch(model, model_name):
  from keras.utils.visualize_util import plot
  path = os.path.join('images', model_name + '.png')
  plot(model, to_file=path, show_shapes=True)
