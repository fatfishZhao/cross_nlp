import pandas as pd
# 线性回归
from sklearn import datasets, linear_model, cross_validation
import pickle
from sklearn.metrics import mean_squared_error, r2_score


import numpy as np
def load_data(fea_path = '../feature.pkl'):
    with open(fea_path, 'rb') as f:
        fea_df = pd.DataFrame(pickle.load(f))
    fea_df = fea_df[['PP_NR', 'WORD_TOTAL_READING_TIME']+list(fea_df.columns)[-9:]]
    fea_df = fea_df.replace('.', 0)
    fea_df = fea_df.dropna()
    return fea_df

fea_data = load_data('../feature.pkl')
people_list = list(fea_data.drop_duplicates(subset='PP_NR', inplace = False)['PP_NR'])
score_dict = {}
for people_id in people_list:
    tmp_df = fea_data[fea_data['PP_NR']==people_id]
    data_X = tmp_df.iloc[:,2:].values
    data_Y = tmp_df.iloc[:,1].values

    regr = linear_model.LinearRegression()
    scores = cross_validation.cross_val_score(regr, data_X, data_Y, scoring='neg_mean_squared_error', cv=10)
    # print(people_id+': ', scores.mean())
    # print('std^2 is ', data_Y.std()**2)
    print('regression/std^2:', -scores.mean()/(data_Y.std()**2))
    score_dict.update({people_id:scores.mean()})