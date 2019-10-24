from cvxopt import matrix, solvers
import sys
sys.path.append('../SVM')
import init, svm
from kernel import *

# SVR実装モデル
def fit(x, y, kernel):
    eps = 0.1
    C = 1000.0
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
            P[i + m][j] = -1.0 * P[i][j]
            P[i][m + j] = -1.0 * P[i][j]
    P = matrix(P)
    q = np.zeros(m * 2)
    for i in range(m):
        q[i] = 1.0 * y[i] + eps
        q[i + m] = -1.0 * y[i] + eps
    q = matrix(q)
    tmp1 = -1.0 * np.diag(np.ones(m * 2))
    tmp2 = np.diag(np.ones(m * 2))
    G = matrix(np.vstack((tmp1, tmp2)))
    tmp1 = np.zeros(m * 2)
    tmp2 = np.ones(m * 2) * C
    h = matrix(np.hstack((tmp1, tmp2)))
    A = matrix(np.append(np.ones(m), -1.0*np.ones(m)), (1, m*2))
    b = matrix(np.zeros(1))

    # solverのパラメータの調整 
    solvers.options['show_progress'] = False
    solvers.options['abstol'] = 1e-10
    solvers.options['reltol'] = 1e-10
    solvers.options['feastol'] = 1e-10

    # solverを実行
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    w = (alphas[:m] - alphas[m:]).ravel()
    """
    b_array = y - _inner_mat(self.x) + eps
    self._b = np.mean(b_array[self.support_vector_id])"""
    print(alphas)
    print(w.T)
    print(x[0])

    f = np.dot(w.T,x[0])
    print(f)
    return alphas, w

# メイン関数
def main():
    filename, kernel, n = init.set_parser()
    x, y = init.load_data(filename)
    alphas, w = fit(x, y, kernel)

if __name__ == '__main__':
    main()