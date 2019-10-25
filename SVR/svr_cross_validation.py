import sys
sys.path.append('../SVM')
from init import *
from cross_validation import split_data
import svr

def cross_validate(train_x, test_x, train_y, test_y, kernel, C):
    alphas, w, b = svr.fit(train_x, train_y, kernel, C)
    predict_arr = svr.predict(test_x, w, b, kernel)
    mse = 0
    for i in range(len(test_y)):
        tmp = (test_y[i] - predict_arr[i]) ** 2
        mse += tmp
    return mse / len(test_y)

def main():
    filename, kernel, n = set_parser()
    x, y = load_data(filename)
    c_list = [1.0, 10.0, 100.0, 1000.0]
    result = []
    for j in range(len(c_list)):
        print('---- C = '+str(c_list[j])+' ----')
        total_mse = 0
        for i in range(n):
            train_x, test_x, train_y, test_y = split_data(x, y, n, i)
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