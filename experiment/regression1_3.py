import pandas as pd
# 线性回归
from sklearn import datasets, linear_model, cross_validation
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


import numpy as np
def load_data(fea_path = '../feature_FK.pkl'):
    with open(fea_path, 'rb') as f:
        fea_df = pd.DataFrame(pickle.load(f))
    fea_df = fea_df[['WORD_TOTAL_READING_TIME','WORD']+list(fea_df.columns)[-9:]]
    fea_df = fea_df.replace('.', 0)
    fea_df = fea_df.dropna()
    data_Y = fea_df.iloc[:,0].values
    data_X = fea_df.iloc[:,2:].values
    return data_X, data_Y
data_X, data_Y = load_data('../feature_FK.pkl')
regr = linear_model.LinearRegression()
scores = cross_validation.cross_val_score(regr, data_X, data_Y, scoring='neg_mean_squared_error', cv=10)
print('Linear Regression score:', np.sqrt(-scores.mean()))
ridge = linear_model.Ridge(alpha=100000)
scores = cross_validation.cross_val_score(ridge, data_X, data_Y, scoring='neg_mean_squared_error', cv=10)
print('Ridge Regression score:', np.sqrt(-scores.mean()))
print(data_Y.std())
print(np.sqrt(-scores.mean())/data_Y.std())