import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from utils import reshape2d_by_image_dim_ordering
from keras import backend as K

FTRAIN = 'data/training.csv'
FTEST = 'data/test.csv'

def load(test=False, cols=None):
    """testがTrueの場合はFTESTからデータを読み込み、Falseの場合はFTRAINから読み込みます。
    colsにリストが渡された場合にはそのカラムに関するデータのみ返します。
    """

    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname)) # pandasのdataframeを使用

    # スペースで句切られているピクセル値をnumpy arrayに変換
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # カラムに関連するデータのみを抽出
        df = df[list(cols) + ['Image']]

    print(df.count())  # カラム毎に値が存在する行数を出力
    df = df.dropna()  # データが欠けている行は捨てる

    X = np.vstack(df['Image'].values) / 255.  # 0から1の値に変換
    X = X.astype(np.float32)

    if not test:  # ラベルが存在するのはFTRAINのみ
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # -1から1の値に変換
        X, y = shuffle(X, y, random_state=42)  # データをシャッフル
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


def load2d(test=False, cols=None):
    X, y = load(test, cols)
    # image_dim_orderring に合わせて2D画像のshapeを変える
    X, _ = reshape2d_by_image_dim_ordering(X)
    return X, y


if __name__ == '__main__':
    X, y = load()
    print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
        X.shape, X.min(), X.max()))
    print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
        y.shape, y.min(), y.max()))
