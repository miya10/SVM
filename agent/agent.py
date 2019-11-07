import sys
import numpy as np
from decimal import Decimal
sys.path.append('../SVR')
import svr

# スコアを計算
def calculate_score(test_y, predict_y):
    #predict_y = predict_y * 0.80
    score_arr = np.zeros(0)
    arr = np.arange(0, 1.0, 0.01)
    for j in range(len(arr)):
        #new_predict_y = predict_y - j
        new_predict_y = predict_y * arr[j]
        score = 0.0
        correct_score = 0
        for i in range(len(test_y)):
            if new_predict_y[i] < test_y[i]:
                correct_score += 1
                score += new_predict_y[i]
            elif new_predict_y[i] == test_y[i]:
                score += new_predict_y[i] * 0.5
            else:
                continue
        final_score = score / np.sum(test_y)
        print(str(arr[j]) + '-効率:' + str(final_score) + ' 正解数:' + str(correct_score) + ' 収益:' + str(score))
        score_arr = np.append(score_arr, final_score)
    best_score = np.max(score_arr)
    best_index = np.argmax(score_arr)
    print(str(best_index)+'でscore:'+str(best_score))
    return best_score

def average_agent(test_y):
    new_predict_y = np.full(len(test_y), np.average(test_y))
    score = 0.0
    correct_score = 0
    for i in range(len(test_y)):
        
        if new_predict_y[i] < test_y[i]:
            correct_score += 1
            score += new_predict_y[i]
        elif new_predict_y[i] == test_y[i]:
            score += new_predict_y[i] * 0.5
        else:
            continue
    final_score = score / np.sum(test_y)
    print('平均値agent\n効率:' + str(final_score) + ' 正解数:' + str(correct_score) + ' 収益:' + str(score))

# メイン関数
def main():
    test_y, predict_y = svr.main()
    average_agent(test_y)
    calculate_score(test_y, predict_y)

if __name__ == '__main__':
    main()