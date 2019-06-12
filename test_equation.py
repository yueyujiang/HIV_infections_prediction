import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def test_function(infection, diagnosis, alpha):
    min_len = min(len(infection), len(diagnosis))
    infection = infection[0: min_len]
    diagnosis = diagnosis[0: min_len]
    alpha = np.concatenate((alpha, np.ones((min_len-len(alpha),)).astype(float)))
    infection_change = infection.copy()
    infection_change[1:] = infection[1:] - infection[:-1]
    predicted_diagnosis = np.zeros(diagnosis.shape)
    for i in range(len(diagnosis)):
        try:
            predicted_diagnosis[i] = np.sum(infection_change[i:: -1] * alpha[0: i + 1])
        except:
            break
    plt.plot(diagnosis)
    plt.plot(predicted_diagnosis)
    plt.plot(infection)
    plt.legend(['diagnosis', 'approximated diagnosis', 'infection'])
    plt.savefig('f6.png')
    return predicted_diagnosis

idx = 7999
r = 500
degree = 12
infection = np.load('infection.npy')
diagnosis = np.load('diagnosis.npy')
alpha = np.load('alpha.npy')
error = 0
# for idx in range(diagnosis.shape[0]):
#     i = infection[idx, :r]
#     d = diagnosis[idx, :r]
#     diagnosis_appr = test_function(i, d, alpha)
#     start = 0
#     while d[start] == 0:
#         start += 1
#     error += np.sum(np.abs(diagnosis_appr[start:] - d[start:]) / np.abs(d[start:])) / (r - start)
#     print(error)
#     print(idx)
# print(error / diagnosis.shape[0])

i = infection[idx, :r]
d = diagnosis[idx, :r]
diagnosis_appr = test_function(i, d, alpha)