# 交差検定
import numpy as np

from svm import *


# 入力データを分割
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

def cross_validate():
    N = 2

if __name__ == '__main__':
    filename, kernel, n = set_parser()
    x, y = load_data(filename)

    for i in range(n):
        train_x, test_x, train_y, test_y = split_data(x, y, n, i)
        alphas, w, b = fit(train_x, train_y, kernel)
        if kernel == None:
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
        else:
            split = 52
            x1 = np.linspace(-1, 50, split)
            mesh_x, mesh_y = np.meshgrid(x1, x1)
            mesh_lst = np.array([mesh_x.ravel(), mesh_y.ravel()]).T
            predict_y = np.empty(0)
            for i in range(len(test_x)):
                f = func_kernel(train_x, test_x[i], alphas, train_y, b, kernel)
                if f < 0:
                    predict_y = np.append(predict_y, -1)
                else:
                    predict_y = np.append(predict_y, 1)
            correct_num = np.sum(predict_y == test_y)
            accuracy = correct_num / len(test_x)
            print("accuracy = %e" % (accuracy))