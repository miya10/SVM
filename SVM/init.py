# 実行に必要な関数のまとまり
import argparse
import numpy as np

"""
--主要変数の説明--
filename:読み込みデータのファイル名
kernel:選択されたカーネル
n:データ分割数
data:テキストファイルから直接読み込んだデータの配列
"""

# コマンドライン引数の設定
def set_parser():
    parser = argparse.ArgumentParser(usage='SVM実装プログラム', add_help=True)
    parser.add_argument('--filename', '-f', type=str, help='訓練データのファイルパス', required=True)
    parser.add_argument('--kernel_type', '-k', type=str, help='カーネルの指定, [None] or [gaussian_kernel] or [polynomial_kernel]', choices=['gaussian_kernel', 'polynomial_kernel', 'sigmoid_kernel', None], default=None)
    parser.add_argument('--division', '-n', type=int, help='交差検定の分割数n', default=None)
    
    args = parser.parse_args()
    filename = args.filename
    kernel = args.kernel_type
    n = args.division
    return filename, kernel, n

# 入力データを正規化（使わないかも）
def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

# txtファイルからxy座標と教師信号をnumpy配列で取得
def load_data(filename):
    if 'txt' in filename:
        data = np.loadtxt(filename, delimiter=",")
    else:
        data = np.loadtxt(filename, dtype=float)
    x, y = data[:,0:-1], data[:,-1]
    #x = min_max(x)
    return x, y