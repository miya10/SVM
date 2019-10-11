import numpy as np
from scipy.linalg import norm

# 多項式カーネル
def polynomial_kernel(x, y):
    return (1 + np.dot(x, y)) ** 2   # 2次元だから2

# ガウスカーネル
def gaussian_kernel(x, y):
    sigma = 10
    return np.exp(- norm(x-y) ** 2 / (2 * (sigma ** 2)))

# シグモイドカーネル
def sigmoid_kernel(x, y):
    a = 3
    b = 3
    return np.tanh(a * np.dot(x, y) + b)
