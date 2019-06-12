import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge


diagnosis = np.load('infection.npy')
ans = np.zeros(diagnosis.shape)
for idx in range(diagnosis.shape[0]):
    diagnosis_tmp = diagnosis[idx, :].copy()
    X = np.arange(0, len(diagnosis_tmp)).reshape((-1, 1))
    y = np.ravel(diagnosis_tmp)
    model = make_pipeline(PolynomialFeatures(12), Ridge())
    model.fit(X, y)
    diagnosis_appr = model.predict(X)
    ans[idx, :] = diagnosis_appr

np.save('infection_appr.npy', ans)