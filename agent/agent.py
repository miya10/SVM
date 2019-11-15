import sys
import numpy as np
from decimal import Decimal
sys.path.append('../SVR')
import svr

# スコアを計算
def simple_agent(test_y, predict_y):
    best_benefits = 0
    new_predict_y = predict_y - 50
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
    best_score = score
    best_benefits = final_score
    best_correct_score = correct_score
    print('効率=%0.3f, 提供成功回数=%d/%d, 収益=%d/%d' % (best_benefits, best_correct_score, len(test_y), best_score, np.sum(test_y)))


# スコアを計算
def nomal_agent(test_y, predict_y):
    arr = np.arange(0.5, 1.0, 0.01)
    best_benefits = 0
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
        if final_score > best_benefits:
            best_param = arr[j]
            best_score = score
            best_benefits = final_score
            best_correct_score = correct_score
    print('param=%0.2f, 効率=%0.3f, 提供成功回数=%d/%d, 収益=%d/%d' % (best_param, best_benefits, best_correct_score, len(test_y), best_score, np.sum(test_y)))

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
    print('効率=%0.3f, 提供成功回数=%d/%d, 収益=%d/%d' % (final_score, correct_score, len(test_y), score, np.sum(test_y)))

def random_agent(test_y, predict_y):
    score_arr = np.zeros(0)
    arr = np.arange(0.5, 1.0, 0.01)
    random_arr = np.random.rand(len(test_y))
    best_benefits = 0
    for j in range(len(arr)):
        new_predict_y = predict_y * arr[j]
        price_sum = 0.0
        score = 0.0
        correct_score = 0
        sample_num = 0
        for i in range(len(test_y)):
            if random_arr[i] < 0.5:
                sample_num += 1
                price_sum += test_y[i]
                if new_predict_y[i] < test_y[i]:
                    correct_score += 1
                    score += new_predict_y[i]
                elif new_predict_y[i] == test_y[i]:
                    score += new_predict_y[i] * 0.5
                else:
                    continue
        final_score = score / price_sum
        if final_score > best_benefits:
            best_param = arr[j]
            best_score = score
            best_benefits = final_score
            best_correct_score = correct_score
    print('param=%0.2f, 効率=%0.3f, 提供成功回数=%d/%d, 収益=%d/%d' % (best_param, best_benefits, best_correct_score, sample_num, best_score, price_sum))

def outlier_agent(test_y, predict_y):
    score_arr = np.zeros(0)
    arr = np.arange(0.5, 1.0, 0.01)
    best_benefits = 0
    for j in range(len(arr)):
        new_predict_y = predict_y * arr[j]
        sorted_predict_y = np.sort(predict_y)
        price_sum = 0.0
        score = 0.0
        correct_score = 0
        sample_num = 0
        for i in range(len(test_y)):
            if predict_y[i] < sorted_predict_y[int(len(predict_y)*0.75)]:
                sample_num += 1
                price_sum += test_y[i]
                if new_predict_y[i] < test_y[i]:
                    correct_score += 1
                    score += new_predict_y[i]
                elif new_predict_y[i] == test_y[i]:
                    score += new_predict_y[i] * 0.5
                else:
                    continue
        final_score = score / price_sum
        if final_score > best_benefits:
            best_param = arr[j]
            best_score = score
            best_benefits = final_score
            best_correct_score = correct_score
    print('param=%0.2f, 効率=%0.3f, 提供成功回数=%d/%d, 収益=%d/%d' % (best_param, best_benefits, best_correct_score, sample_num, best_score, price_sum))

def compare_agent(test_y, predict_y):
    score_arr = np.zeros(0)
    arr = np.arange(0.5, 1.0, 0.01)
    best_benefits = 0
    for j in range(len(arr)):
        new_predict_y = predict_y * arr[j]
        sorted_predict_y = np.sort(predict_y)
        price_sum = 0.0
        score = 0.0
        correct_score = 0
        sample_num = 0
        for i in range(len(test_y)):
            if predict_y[i] < sorted_predict_y[int(len(predict_y)*0.75)]:
                sample_num += 1
                price_sum += test_y[i]
                if new_predict_y[i] < test_y[i]:
                    correct_score += 1
                    score += new_predict_y[i]
                elif new_predict_y[i] == test_y[i]:
                    score += new_predict_y[i] * 0.5
                else:
                    continue
        final_score = score / price_sum

# メイン関数
def main():
    test_y, predict_y = svr.main()
    print('-average agent-')
    average_agent(test_y)
    print('-simple agent-')
    simple_agent(test_y, predict_y)
    print('-nomal agent-')
    nomal_agent(test_y, predict_y)
    print('-random agent-')
    random_agent(test_y, predict_y)
    print('-outlier agent-')
    outlier_agent(test_y, predict_y)

if __name__ == '__main__':
    main()