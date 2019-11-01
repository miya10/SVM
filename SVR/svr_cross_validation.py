import sys
sys.path.append('../SVM')
from init import *
import svr
from data_organize import data_organize



def cross_validate(train_x, test_x, train_y, test_y, kernel, C):
    alphas, w, b, diff = svr.fit(train_x, train_y, kernel, C)
    if kernel == None:
        predict_y = svr.predict(train_x, test_x, w, b, diff, kernel)
    else:
        predict_y = np.zeros(0)
        for i in range(len(test_x)):
            result = svr.predict(train_x, test_x[i], w, b, diff, kernel)
            predict_y = np.append(predict_y, result)
    mse = 0
    for i in range(len(test_y)):
        tmp = (test_y[i] - predict_y[i]) ** 2
        mse += tmp
    return mse / len(test_y)

def main():
    filename, kernel, n = set_parser()
    #x, y = load_data(filename)
    x, y = data_organize(filename)
    x = x[0:400,]
    y = y[0:400,]
    c_list = [1.0, 10.0, 100.0, 1000.0]
    result = []
    for j in range(len(c_list)):
        print('---- C = '+str(c_list[j])+' ----')
        total_mse = 0
        for i in range(n):
            train_x, test_x, train_y, test_y = svr.split_data(x, y, n, i)
            mse = cross_validate(train_x, test_x, train_y, test_y, kernel, c_list[j])
            print('MSE = ', mse)
            total_mse += mse
        total_mse = total_mse / n
        result.append(total_mse)
        print('評価結果：', total_mse)
    best_index = result.index(min(result))
    print('C = %e で最も良い精度が観測されました．' % (c_list[best_index]))

if __name__ == '__main__':
    main()