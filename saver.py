from keras.models import model_from_json

def save_arch(model, path):
  """
    モデルのグラフをファイルに保存する
    model: 保存をするモデル
    path: 保存先のファイル名
  """
  json_string = model.to_json()
  open(path, 'w').write(json_string)


def load_arch(path):
  """
    モデルのグラフをファイルから取得する
    path: 取得先のファイル名
  """
  return model_from_json(open(path).read())
