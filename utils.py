from keras import backend as K

def reshape2d_by_image_dim_ordering(X):
  """
  image_dim_orderring に合わせて2D画像のshapeを変える
  """
  if K.image_dim_ordering() == 'th':
      X = X.reshape(X.shape[0], 1, 96, 96)
      input_shape = (1, 96, 96)
  else:
      X = X.reshape(X.shape[0], 96, 96, 1)
      input_shape = (96, 96, 1)

  return X, input_shape
