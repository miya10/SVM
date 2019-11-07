import numpy as np
import pandas as pd
from scipy import stats
import sys
sys.path.append('../SVM')
import init

"""
--主要変数の説明--
df:pandas形式のデータフレーム
data:dfをnumpy形式に変形したもの
x:読み込みデータのn次元特徴を表す配列
y:価格の配列
"""

# csvファイルの読み込み
def data_organize(filename):
    df = pd.read_csv(filename, header=0)
    #df = df[['accommodates', 'bathrooms', 'bedrooms', 'beds', 'number_of_reviews', 'review_scores_rating', 'review_scores_value', 'reviews_per_month', 'price']]
    df = df[['accommodates', 'bathrooms', 'bedrooms', 'beds', 'latitude', 'calculated_host_listings_count', 'longitude', 'zipcode', 'number_of_reviews', 'review_scores_rating', 'review_scores_value', 'reviews_per_month', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'price']]
    df = df.dropna(how='any')
    table = str.maketrans({',': '', '$': '',})
    df['price'] = df['price'].str.translate(table).astype(np.float)
    data = df.values
    x, y = data[:,0:-1], data[:,-1]
    return x, y