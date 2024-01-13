import numpy as np
import matplotlib.pyplot as plt

def calc_cov(x_a, x_b, a_mean = None, b_mean = None):
    if a_mean == None:
        a_mean = np.mean(x_a)
    elif b_mean == None:
        x_b_mean = np.mean(x_b)
    s = 0
    for i in range(len(x_a)):
        s = s + ((x_a[i] - x_a_mean) * (x_b[i] - x_b_mean))
    return s / x_a.shape[0]

def est_cov_matrix(data, custom_means = None):
    n = data.shape[0]
    cov_matrix = np.zeros((n, n))
    
    if custom_means == None:
        custom_means = []
        for i in range(n):
            custom_means.append(None)

    for i in range(n):
        for j in range(n):
            cov_matrix[i][j] = calc_cov(data[i], data[j], custom_means[i], custom_means[j])
    
    return cov_matrix

def generate_data(seed, m):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, ))

def get_means(data):
    means = np.zeros(data.shape[0])
    for i in range(len(means)):
        means[i] = np.mean(data[i])
    return means