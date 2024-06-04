import numpy as np
import sklearn.metrics as sm
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


def RMSE(y_true, y_pred):
    return np.sqrt(sm.mean_squared_error(y_true, y_pred))

def MAE(y_true, y_pred):
    return sm.mean_absolute_error(y_true, y_pred)

def PR(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

def SD(y_true, y_pred):
    y_pred = y_pred.reshape((-1, 1))
    lr = LinearRegression().fit(y_pred, y_true)
    y_ = lr.predict(y_pred)
    return np.sqrt(np.square(y_true - y_).sum() / (len(y_pred) - 1))
