from cvxopt import matrix, solvers
import numpy as np
import matplotlib.pyplot as plt
import argparse
from kernel import polynomial_kernel, gaussian_kernel, sigmoid_kernel

# 入力データを正規化（使わないかも）
def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

# txtファイルからxy座標と教師信号をnumpy配列で取得
def load_data(filename):
    data = np.loadtxt(filename, delimiter=",")
    x, y = data[:,0:2], data[:,2]
    #x = min_max(x)
    return x, y

def fit(x, y, kernel):
    # 初期設定
    m, n = x.shape
    y = y.reshape(-1, 1)
    data_size = len(x)
    
    # cvxotの形式にデータを変換
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
    if kernel == None:
        b = y[S] - np.dot(x[S], w)
        b = np.average(b)
    else:
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

def func_no_kernel(x1, w, b):
    return - (w[0] / w[1]) * x1 - (b / w[1])

def func_kernel(x, mesh_lst, alphas, y, b, kernel):
    sum_tmp = 0
    for i in range(len(x)):
        sum_tmp += alphas[i] * y[i] * eval(kernel)(x[i], mesh_lst)
    return sum_tmp - b

# グラフを描写
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

def draw_graph_kernel(x, y, mesh_x, mesh_y, f):
    for n in range(len(x)):
        if y[n] == 1:
            plt.scatter(x[n,0], x[n,1], s = 10, color = 'r')
        else:
            plt.scatter(x[n,0], x[n,1], s = 10, color = 'b')

    plt.contour(mesh_x, mesh_y, f, [0.0], colors='k', linewidths=1, origin='lower')
    plt.xlim(-1, 50)
    plt.ylim(-1, 50)
    #plt.savefig('results/sigmoid_circle.png')
    plt.show()

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

if __name__ == '__main__':
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
        print(f)
        draw_graph_kernel(x, y, mesh_x, mesh_y, f)