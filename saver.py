from keras.models import model_from_json
import pickle

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


def save_history(hist, model_name):
  with open('history/' + model_name +'.pickle', mode='wb') as f:
    pickle.dump(hist.history, f)
