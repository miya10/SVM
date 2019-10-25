from cvxopt import matrix, solvers
import sys
sys.path.append('../SVM')
import init, svm
from kernel import *

# SVR実装モデル
def fit(x, y, kernel, C=1000.0):
    eps = 0.1
    # cvxoptの形式にデータを変換
    m, n = x.shape
    y = y.reshape(-1, 1)
    data_size = len(x)
    
    P = np.zeros((m * 2, m * 2))
    for i in range(m):
        for j in range(m):
            if kernel == None:
                P[i][j] = np.inner(x[i], x[j])
            else:
                P[i][j] = eval(kernel)(x[i], x[j])
            P[m + i][m + j] = P[i][j]
            P[i + m][j] = - 1.0 * P[i][j]
            P[i][m + j] = - 1.0 * P[i][j]
    P = matrix(P)
    q = np.zeros(m * 2)
    for i in range(m):
        q[i] = 1.0 * y[i] + eps
        q[i + m] = -1.0 * y[i] + eps
    q = matrix(q)
    tmp1 = - 1.0 * np.diag(np.ones(m * 2))
    tmp2 = np.diag(np.ones(m * 2))
    G = matrix(np.vstack((tmp1, tmp2)))
    tmp1 = np.zeros(m * 2)
    tmp2 = np.ones(m * 2) * C
    h = matrix(np.hstack((tmp1, tmp2)))
    A = matrix(np.append(np.ones(m), - 1.0 * np.ones(m)), (1, m * 2))
    b = matrix(np.zeros(1))

    # solverのパラメータの調整 
    solvers.options['show_progress'] = False
    solvers.options['abstol'] = 1e-10
    solvers.options['reltol'] = 1e-10
    solvers.options['feastol'] = 1e-10

    # solverを実行
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    alphas = np.where(alphas < 1e-6, 0.0, alphas)
    S = (alphas > 1e-6).flatten()
    diff = (alphas[m:] - alphas[:m]).ravel()
    support_vector = np.arange(m)[(diff > eps) | (diff < -eps)]
    w = np.zeros(x.shape[1])
    for i in range(len(x)):
        tmp_w = (alphas[m+i] - alphas[i]) * x[i]
        w += tmp_w

    b = np.zeros(len(x))
    for i in range(len(x)):
        tmp_b_sum = 0
        for j in range(len(x)):
            tmp_b = diff[j] * np.dot(x[i], x[j])
            if kernel == None:
                tmp_b = diff[j] * np.dot(x[i], x[j])
            else:
                tmp_b = diff[j] * eval(kernel)(x[i], x[j])
            tmp_b_sum += tmp_b
        if diff[i] > 0:
            b[i] = - y[i] + tmp_b_sum + eps
        else:
            b[i] = - y[i] + tmp_b_sum - eps
    b = np.average(b[support_vector])
    return alphas, w, b

def predict(x, w, b, kernel):
    result = np.zeros(len(x))
    if kernel == None:
        for i in range(len(x)):
            result[i] = np.dot(w, x[i]) - b
    else:
        for i in range(len(x)):
            result[i] = eval(kernel)(w, x[i]) - b
    return result

# メイン関数
def main():
    filename, kernel, n = init.set_parser()
    x, y = init.load_data(filename)
    alphas, w, b = fit(x, y, kernel)
    result = predict(x, w, b, kernel)
    print('alpha：',alphas.flatten())
    print('重み：', w)
    print('閾値：', b)
    print('result = ', result)

if __name__ == '__main__':
    main()