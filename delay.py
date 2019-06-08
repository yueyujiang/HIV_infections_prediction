import matplotlib.pyplot as plt
import numpy as np
import torch


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
    # plt.plot(diagnosis)
    # plt.plot(predicted_diagnosis)
    # plt.plot(infection)
    # plt.legend(['diagnosis', 'perdicted_diagnosis', 'infection'])
    # plt.savefig('plot.png')
    return predicted_diagnosis
    # print(diagnosis, predicted_diagnosis)

def test_G_0_known(G0, diagnosis, alpha, infection, diagonsis_tmp):
    max_number = 500
    min_len = min(len(infection), len(diagnosis))
    diagnosis = diagnosis[0: min_len]
    # diagnosis = diagnosis[::10]
    if len(alpha) < min_len:
        alpha = np.concatenate((alpha, np.ones((min_len - len(alpha),)).astype(float)))
    # alpha = alpha[::10]
    # infection = infection[::10]
    # min_len = len(diagnosis)
    alpha_diff = alpha.copy()
    alpha_diff[1:] = alpha[1:] - alpha[:-1]
    infection_pred = np.zeros(diagnosis.shape)
    infection_pred[0] = G0
    for i in range(1, min_len):
        tmp = diagnosis[i] - alpha[i] * G0
        for j in range(1, i):
            tmp -= alpha[i - j] * (infection_pred[j] - infection_pred[j - 1])
        infection_pred[i] = min(max(float(tmp) / alpha[0] + infection_pred[i - 1], infection_pred[i-1]), max_number)
        # infection_pred[i] = float(tmp) / alpha[0] + infection_pred[i - 1]
    # for i in range(0, len(infection_pred), 50):
    #     infection_pred[i * 50: i * 50 + 50] = infection_pred[i * 50: i * 50 + 50].mean()
    plt.plot(diagnosis)
    plt.plot(infection_pred)
    plt.plot(infection)
    plt.plot(diagonsis_tmp)
    plt.legend(['fit_diagnosis', 'perdicted_infection', 'infection', 'original_diagnosis'])
    # plt.legend(['diagnosis', 'infection'])
    plt.savefig('plot1.png')
    return infection_pred

def G_0_known(G0, diagnosis, alpha):
    max_number = 500
    min_len = diagnosis.shape[1]
    # diagnosis = diagnosis[0: min_len]
    # diagnosis = diagnosis[::10]
    if len(alpha) < min_len:
        alpha = np.concatenate((alpha, np.ones((min_len - len(alpha),)).astype(float)))
    # alpha_diff = alpha.copy()
    # alpha_diff[1:] = alpha[1:] - alpha[:-1]
    infection_pred = torch.zeros(diagnosis.shape)
    infection_pred[:, 0] = G0
    for i in range(1, min_len):
        tmp = diagnosis[:, i] - alpha[i].float() * G0
        for j in range(1, i):
            tmp -= alpha[i - j].float() * (infection_pred[:, j] - infection_pred[:, j - 1])
        tmp = torch.cat(((tmp.float() / alpha[0].float() + infection_pred[:, i - 1]).unsqueeze(1), infection_pred[:, i-1].unsqueeze(1)), dim=1)
        tmp = torch.max(tmp,dim=1, keepdim=True)[0]
        tmp = torch.cat((tmp, torch.ones(tmp.shape) * max_number), dim=1)
        infection_pred[:, i] = torch.min(tmp, dim=1)[0]
    return torch.tensor(infection_pred[:, 0: min_len])

# idx = 299
# r = 500
# degree = 12
# infection = np.load('infection.npy')
# diagnosis = np.load('diagnosis.npy')
# alpha = np.load('alpha.npy')
# error = 0
# for idx in range(500):
#     diagnosis_tmp = diagnosis[idx, :r].copy()
#     X = np.arange(0, len(diagnosis_tmp)).reshape((-1, 1))
#     y = np.ravel(diagnosis_tmp)
#     model = make_pipeline(PolynomialFeatures(degree), Ridge())
#     model.fit(X, y)
#     diagnosis_appr = model.predict(X)
#     diagnosis_appr[diagnosis_appr < 0] = 0
#     # reg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000).fit(X, y)
#     # diagnosis = reg.predict(X)
#     # diagnosis = test_function(infection[:, 255], diagnosis[:, 255], alpha)
#     infection_appr = test_G_0_known(np.array(infection[idx, 0]), np.array(diagnosis_appr), np.array(alpha), infection[idx, :r], diagnosis_tmp)
#     start = 0
#     while infection[idx, start] == 0:
#         start += 1
#     error += np.sum(np.abs(infection_appr[start:] - infection[idx, start:r]) / (np.abs(infection[idx, start:r]))) / (r - start)
#     print(error)
#     print(idx)
# print(error / diagnosis.shape[0])