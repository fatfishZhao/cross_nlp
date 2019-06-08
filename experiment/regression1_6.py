import pandas as pd
# 线性回归
from sklearn import datasets, linear_model, cross_validation
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

import numpy as np

'''
    nomalize the features and reading time.
'''

def normalize_time(old_df):
    people_list = list(old_df.drop_duplicates(subset='PP_NR', inplace=False)['PP_NR'])
    new_df = old_df.copy()
    for one_people in people_list:
        tmp_df = old_df[old_df['PP_NR']==one_people]
        sum_time = tmp_df['WORD_TOTAL_READING_TIME'].sum()
        normalize_time = tmp_df['WORD_TOTAL_READING_TIME'].values/sum_time
        new_df.loc[list(new_df[new_df['PP_NR']==one_people].index), ['WORD_TOTAL_READING_TIME']]=normalize_time
        # new_df[new_df['PP_NR']==one_people]['WORD_TOTAL_READING_TIME']=normalize_time
    return new_df
def normalize_fea(old_df, fea_num):
    new_fea = preprocessing.scale(old_df.iloc[:, -fea_num:].values, axis=0)
    new_df = old_df.copy()
    new_df.iloc[:, -fea_num:] = new_fea
    return new_df

def load_data(fea_path, fea_num):
    with open(fea_path, 'rb') as f:
        fea_df = pd.DataFrame(pickle.load(f))
    fea_df = fea_df.replace('.', 0)
    fea_df = fea_df.dropna()
    fea_df = normalize_time(old_df=fea_df)
    fea_df = normalize_fea(old_df=fea_df, fea_num=fea_num)

    fea_df = fea_df[['WORD_TOTAL_READING_TIME','WORD']+list(fea_df.columns)[-fea_num:]]

    data_Y = fea_df.iloc[:,0].values
    data_X = fea_df.iloc[:,2:].values
    return data_X, data_Y
data_X, data_Y = load_data('../data/EGCO/all_feature.pkl', fea_num=11)
regr = linear_model.LinearRegression()
linear_scores = cross_validation.cross_val_score(regr, data_X, data_Y, scoring='neg_mean_squared_error', cv=10)
print('Linear Regression score:', np.sqrt(-linear_scores.mean()))
ridge = linear_model.Ridge(alpha=0.1)
ridge_scores = cross_validation.cross_val_score(ridge, data_X, data_Y, scoring='neg_mean_squared_error', cv=10)
print('Ridge Regression score:', np.sqrt(-ridge_scores.mean()))
print(data_Y.std())
print(data_Y.mean())
print(np.sqrt(-linear_scores.mean())/data_Y.std())