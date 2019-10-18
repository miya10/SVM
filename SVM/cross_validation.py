# 交差検定
import numpy as np
from svm import *
from init import *

"""
--主要変数の説明--
x:読み込みデータのn次元特徴を表す配列
y:正解ラベルの配列
train_x:訓練データの特徴を表す配列
train_y:訓練データの正解ラベルの配列
test_x:テストデータの特徴を表す配列
test_y:テストデータの正解ラベルの配列
predict_y:test_xに対して行った予測結果
correct_num:test_yとpredict_yを比較した時の値が一致している数
alphas:2次計画問題の解の配列
accuracy:交差検定を行う際に途中結果の正解率
total_accuracy:accuracyの平均値
f:識別器の計算結果
他の変数はグラフのプロット用もしくは計算途中の変数
"""

# 入力データを訓練用と検証用に分割
def split_data(x, y, n, i):
    x = np.array_split(x, n)
    test_x = x[i]
    train_x = np.delete(x, i, 0)
    train_x = np.reshape(train_x, (-1, 2))
    y = np.array_split(y, n)
    test_y = y[i]
    train_y = np.delete(y, i, 0)
    train_y = np.reshape(train_y, (-1, 1))
    return train_x, test_x, train_y, test_y

# カーネルなしで交差検定を行う関数
def cross_validate_no_kernel(test_x, test_y, w, b, total_accuracy):
    predict_y = np.empty(0)
    for j in range(len(test_x)):
        f = func_no_kernel(test_x[j][0], w, b) - test_x[j][1]
        if f < 0:
            predict_y = np.append(predict_y, -1)
        else:
            predict_y = np.append(predict_y, 1)
    correct_num = np.sum(predict_y == test_y)
    accuracy = float(correct_num) / len(test_x)
    print("accurcy = %e" % (accuracy))
    total_accuracy += accuracy
    return total_accuracy

# カーネルありで交差検定を行う関数
def cross_validate_kernel(train_x, test_x, train_y, test_y, alphas, b, kernel, total_accuracy):
    """# プロット用の格子点作成
    split = 52
    x1 = np.linspace(-1, 50, split)
    mesh_x, mesh_y = np.meshgrid(x1, x1)
    mesh_lst = np.array([mesh_x.ravel(), mesh_y.ravel()]).T"""
    predict_y = np.empty(0)
    for j in range(len(test_x)):
        f = func_kernel(train_x, test_x[j], alphas, train_y, b, kernel)
        if f < 0:
            predict_y = np.append(predict_y, -1)
        else:
            predict_y = np.append(predict_y, 1)
    correct_num = np.sum(predict_y == test_y)
    accuracy = correct_num / len(test_x)
    print("accuracy = %e" % (accuracy))
    total_accuracy += accuracy
    return total_accuracy
    """# レポート用の結果を出力するための部分
    f = np.empty(0)
    for j in range(len(mesh_lst)):
        f_j = func_kernel(train_x, mesh_lst[j], alphas, train_y, b, kernel)
        f = np.append(f, f_j)
    f = f.reshape(split, split)
    filename = 'results/poly_'+str(i)+'_accuracy='+str(accuracy)+'.png'
    draw_graph_kernel(x, y, mesh_x, mesh_y, f, filename)"""
    

def main():
    filename, kernel, n = set_parser()
    x, y = load_data(filename)
    total_accuracy = 0
    for i in range(n):
        train_x, test_x, train_y, test_y = split_data(x, y, n, i)
        alphas, w, b = fit(train_x, train_y, kernel)
        if kernel == None:
            total_accuracy = cross_validate_no_kernel(test_x, test_y, w, b, total_accuracy)
        else:
            total_accuracy = cross_validate_kernel(train_x, test_x, train_y, test_y, alphas, b, kernel, total_accuracy)
    print('total accuracy = %e' % (total_accuracy / n))

if __name__ == '__main__':
    main()