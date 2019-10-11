#coding:utf-8

# 非線形SVM（ソフトマージン）
# cvxoptのQuadratic Programmingを解く関数を使用

import numpy as np
from scipy.linalg import norm
import cvxopt
import cvxopt.solvers
from pylab import *

N = 100         # データ数
C = 0.5         # スラック変数を用いて表されるペナルティとマージンのトレードオフパラメータ
SIGMA = 0.3     # ガウスカーネルのパラメータ

# 多項式カーネル
def polynomial_kernel(x, y):
    return (1 + np.dot(x, y)) ** P

# ガウスカーネル
def gaussian_kernel(x, y):
    return np.exp(-norm(x-y)**2 / (2 * (SIGMA ** 2)))

# どちらのカーネル関数を使うかここで指定
kernel = gaussian_kernel

def f(x, a, t, X, b):
    sum = 0.0
    for n in range(N):
        sum += a[n] * t[n] * kernel(x, X[n])
    return sum + b

if __name__ == "__main__":
    # 訓練データをロード
    data = np.genfromtxt("sample_circle.txt")
    X = data[:,0:2]
    t = data[:,2] # 教師信号を-1 or 1に変換
    
    # ラグランジュ乗数を二次計画法（Quadratic Programming）で求める
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = t[i] * t[j] * kernel(X[i], X[j])
    
    m, n = X.shape
    t = t.reshape(-1, 1)
    x_tmp = t * X
    K_no_kernel = np.dot(x_tmp, x_tmp.T)
    data_size = len(X)
    K_kernel = np.zeros((data_size, data_size))
    for i in range(data_size):
        for j in range(data_size):
            K_kernel[i,j] = t[i] * t[j] * kernel(X[i], X[j])
    
    # cvxotの形式にデータを変換
    if kernel == None:
        P = matrix(no_kernel)
    else:
        P = matrix(K_kernel)
    q = matrix(-np.ones((m, 1)))    # 全て1 (m*1)
    G = matrix(-np.eye(m))          # 対角成分が-1 (m*m)
    h = matrix(np.zeros(m))         # 全て0 (m*1)
    A = matrix(t.reshape(1, -1))    # 分類ラベル (1*m)
    b = matrix(np.zeros(1))         # スカラー

    # solverのパラメータの調整 
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['abstol'] = 1e-10
    cvxopt.solvers.options['reltol'] = 1e-10
    cvxopt.solvers.options['feastol'] = 1e-10

    # solverを実行
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    a = array(sol['x']).reshape(N)
    
    # サポートベクトルのインデックスを抽出
    S = []
    M = []
    for n in range(len(a)):
        if 0 < a[n]:
            S.append(n)
        if 0 < a[n] < C:
            M.append(n)
    
    # bを計算
    sum = 0
    for n in M:
        temp = 0
        for m in S:
            temp += a[m] * t[m] * kernel(X[n], X[m])
        sum += (t[n] - temp)
    b = sum / len(M)
    
    
    # 訓練データを描画
    for n in range(N):
        if t[n] > 0:
            scatter(X[n,0], X[n,1], c='b', marker='o')
        else:
            scatter(X[n,0], X[n,1], c='r', marker='o')
    
    # サポートベクトルを描画
#    for n in S:
#        scatter(X[n,0], X[n,1], s=80, c='c', marker='o')
    
    # 識別境界を描画
    X1, X2 = meshgrid(linspace(-2,2,50), linspace(-2,2,50))
    w, h = X1.shape
    X1.resize(X1.size)
    X2.resize(X2.size)
    Z = array([f(array([x1, x2]), a, t, X, b) for (x1, x2) in zip(X1, X2)])
    X1.resize((w, h))
    X2.resize((w, h))
    Z.resize((w, h))
    CS = contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    
    for n in S:
        print (f(X[n], a, t, X, b))
    
    xlim(-2, 2)
    ylim(-2, 2)
    show()