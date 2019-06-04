import pandas as pd
# 线性回归
from sklearn import datasets, linear_model, cross_validation
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

import numpy as np
def normalize_fea(old_df, fea_num):
    new_fea = preprocessing.scale(old_df.iloc[:, -fea_num:].values, axis=0)
    new_df = old_df.copy()
    new_df.iloc[:, -fea_num:] = new_fea
    return new_df
def load_data(fea_path = '../feature_FK.pkl', fea_num=11):
    with open(fea_path, 'rb') as f:
        fea_df = pd.DataFrame(pickle.load(f))
    fea_df = fea_df[['fixation_duration','word']+list(fea_df.columns)[-fea_num:]]
    fea_df = fea_df.replace('.', 0)
    fea_df = fea_df.dropna()
    fea_df = normalize_fea(fea_df, fea_num)
    data_Y = fea_df.iloc[:,0].values
    data_X = fea_df.iloc[:,2:].values
    return data_X, data_Y
data_X, data_Y = load_data('../data/Mishra/all_feature.pkl', fea_num=11)
regr = linear_model.LinearRegression()
scores = cross_validation.cross_val_score(regr, data_X, data_Y, scoring='neg_mean_squared_error', cv=10)
print('Linear Regression score:', np.sqrt(-scores.mean()))
ridge = linear_model.Ridge(alpha=1)
scores = cross_validation.cross_val_score(ridge, data_X, data_Y, scoring='neg_mean_squared_error', cv=10)
print('Ridge Regression score:', np.sqrt(-scores.mean()))
print(data_Y.std())
print(np.sqrt(-scores.mean())/data_Y.std())