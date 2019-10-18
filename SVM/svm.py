# coding
from cvxopt import matrix, solvers
import numpy as np
import matplotlib.pyplot as plt
from kernel import *
from init import *

"""
--主要変数の説明--
x:読み込みデータのn次元特徴を表す配列
y:正解ラベルの配列
alphas:2次計画問題の解の配列
f:識別器の計算結果
他の変数はグラフのプロット用もしくは計算途中の変数
"""

# SVM実装モデル
def fit(x, y, kernel):
    # cvxotの形式にデータを変換
    m, n = x.shape
    y = y.reshape(-1, 1)
    data_size = len(x)
    if kernel == None:
        x_tmp = y * x
        K_no_kernel = np.dot(x_tmp, x_tmp.T)
        P = matrix(K_no_kernel)
    else:
        K_kernel = np.zeros((data_size, data_size))
        for i in range(data_size):
            for j in range(data_size):
                K_kernel[i,j] = y[i] * y[j] * eval(kernel)(x[i], x[j])
        P = matrix(K_kernel)
    q = matrix(-np.ones((m, 1)))    # 全て1 (m*1)
    G = matrix(-np.eye(m))          # 対角成分が-1 (m*m)
    h = matrix(np.zeros(m))         # 全て0 (m*1)
    A = matrix(y.reshape(1, -1))    # 分類ラベル (1*m)
    b = matrix(np.zeros(1))         # スカラー

    # solverのパラメータの調整 
    solvers.options['show_progress'] = False
    solvers.options['abstol'] = 1e-10
    solvers.options['reltol'] = 1e-10
    solvers.options['feastol'] = 1e-10

    # solverを実行
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    w = ((y * alphas).T @ x).reshape(-1,1)
    S = (alphas > 1e-6).flatten()
    if kernel == None:      # カーネルなしの時の閾値計算
        b = y[S] - np.dot(x[S], w)
        b = np.average(b)
    else:                   # カーネルありの時の閾値計算
        b = 0
        S_index = np.where(S==True)
        for i in range(len(x[S])):
            sum_tmp_b = 0
            for j in range(len(x[S])):
                tmp_b = alphas[S_index[0][j]] * y[S][j] * eval(kernel)(x[S][j], x[S][i])
                sum_tmp_b += tmp_b
            b_i = -sum_tmp_b + y[S][i]
            b += b_i
        b = - b / len(x[S])
    return alphas, w ,b

# カーネルなしの識別器計算
def func_no_kernel(x1, w, b):
    return - (w[0] / w[1]) * x1 - (b / w[1])

# カーネルありの識別器計算
def func_kernel(x, mesh_lst, alphas, y, b, kernel):
    sum_tmp = 0
    for i in range(len(x)):
        sum_tmp += alphas[i] * y[i] * eval(kernel)(x[i], mesh_lst)
    return sum_tmp - b

# カーネルなしの時のグラフを描写
def draw_graph(x, y, f):
    for n in range(len(x)):
        if y[n] == 1:
            plt.scatter(x[n,0], x[n,1], s = 10, color = 'r')
        else:
            plt.scatter(x[n,0], x[n,1], s = 10, color = 'b')
    x1 = np.linspace(-1, 50, 1000)
    x2 = f
    plt.plot(x1, x2, 'g-')
    plt.xlim(-1, 50)
    plt.ylim(-1, 50)
    #plt.savefig('results/カーネる.png')
    plt.show()

# カーネルありのグラフ描写
def draw_graph_kernel(x, y, mesh_x, mesh_y, f):
    for n in range(len(x)):
        if y[n] == 1:
            plt.scatter(x[n,0], x[n,1], s = 10, color = 'r')
        else:
            plt.scatter(x[n,0], x[n,1], s = 10, color = 'b')

    plt.contour(mesh_x, mesh_y, f, [0.0], colors='k', linewidths=1, origin='lower')
    plt.xlim(-1, 50)
    plt.ylim(-1, 50)
    filename = 'results/gaussian_sigma=5.png'
    plt.savefig(filename)
    plt.show()

# メイン関数
def main():
    filename, kernel, n = set_parser()
    x, y = load_data(filename)
    alphas, w, b = fit(x, y, kernel)
    if kernel == None:
        x1 = np.linspace(-1, 50, 1000)
        f = func_no_kernel(x1, w, b)
        draw_graph(x, y, f)
    else:
        split = 50
        x1 = np.linspace(-1, 50, split)
        mesh_x, mesh_y = np.meshgrid(x1, x1)
        mesh_lst = np.array([mesh_x.ravel(), mesh_y.ravel()]).T
        f = np.empty(0)
        for i in range(len(mesh_lst)):
            f_i = func_kernel(x, mesh_lst[i], alphas, y, b, kernel)
            f = np.append(f, f_i)
        f = f.reshape(split, split)
        draw_graph_kernel(x, y, mesh_x, mesh_y, f)

    print("----結果出力----\n解：α=%r\n重み：w=%r\n閾値：θ=%e" % (alphas, alphas.flatten()*y, b))

if __name__ == '__main__':
    main()