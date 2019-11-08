from cvxopt import matrix, solvers
import sys
sys.path.append('../SVM')
import init, svm
from kernel import *
import data_organize
import matplotlib.pyplot as plt

"""
--主要変数の説明--
x:読み込みデータのn次元特徴を表す配列
y:価格の配列
kernel:指定されたカーネル
train_x:学習用の特徴量
test_x:テスト用の特徴量
train_y:学習用の価格データ
test_y:テスト用の価格データ
alphas:2次計画問題の解の配列
w:重み
b:閾値
diff:α_k - α_k^* の配列
predict_y:価格の予測結果
他の変数はグラフのプロット用もしくは計算途中の変数
"""

# 訓練データとテストデータに分割
def split_data(x, y, n, i):
    test_len = len(x) // n
    feature_num = x.shape[1]
    test_x = x[len(x) - (i+1) * test_len:len(x) - i * test_len,]
    train_x = np.delete(x, slice(len(x) - (i+1) * test_len, len(x) - i * test_len), 0)
    test_y = y[len(x) - (i+1) * test_len:len(x) - i * test_len,]
    train_y = np.delete(y, slice(len(x) - (i+1) * test_len, len(x) - i * test_len), 0)
    return train_x, test_x, train_y, test_y

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
    return alphas, w, b, diff

# 回帰式を計算
def predict(train_x, test_x, w, b, diff, kernel):
    result = np.zeros(len(train_x))
    if kernel == None:
        for i in range(len(x)):
            result[i] = np.dot(w, test_x[i]) - b
    else:
        for i in range(len(train_x)):
            result[i] =  diff[i] * eval(kernel)(train_x[i], test_x)
        result = np.sum(result) - b
    return result

# グラフを描写
def draw_graph(test_y, predict_y):
    fig, ax = plt.subplots(facecolor="w")
    ax.plot(test_y, color='b', label="real")
    ax.plot(predict_y, color='r', label='predict')
    ax.legend()
    filename = 'results/gaussian_root5_seikika.png'
    #plt.savefig(filename)
    plt.show()

# メイン関数
def main_for_sample():
    filename, kernel, n = init.set_parser()
    x, y = init.load_data(filename)
    alphas, w, b, diff = fit(x, y, kernel)
    if kernel == None:
        predict_y = predict(x, x, w, b, diff, kernel)
    else:
        predict_y = np.zeros(0)
        for i in range(len(x)):
            result = predict(x, x[i], w, b, diff, kernel)
            predict_y = np.append(predict_y, result)
    draw_graph(y, predict_y)
    print('alpha：',alphas.flatten())
    print('重み：', w)
    print('閾値：', b)
    print('result = ', predict_y)

def main():
    filename, kernel, n = init.set_parser()
    x, y = data_organize.data_organize(filename)
    size = 500
    x = x[0:size,]
    y = y[0:size,]
    for i in range(x.shape[1]):
        #x[:, i] = init.standardization(x[:, i])
        x[:, i] = init.min_max(x[:, i])
    train_x, test_x, train_y, test_y = split_data(x, y, n, n-1)
    alphas, w, b, diff = fit(train_x, train_y, kernel)
    if kernel == None:
        predict_y = predict(train_x, test_x, w, b, diff, kernel)
    else:
        predict_y = np.zeros(0)
        for i in range(len(test_x)):
            result = predict(train_x, test_x[i], w, b, diff, kernel)
            predict_y = np.append(predict_y, result)
    #draw_graph(test_y, predict_y)
    """print('alpha：',alphas.flatten())
    print('重み：', w)
    print('閾値：', b)
    print('result = ', predict_y)"""
    return test_y, predict_y

if __name__ == '__main__':
    main()
    #main_for_sample()