import pandas as pd
# 线性回归
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score, KFold
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
def load_data(fea_df, fea_num=11):
    fea_df = fea_df[['fixation_duration','word', 'text_id']+list(fea_df.columns)[-fea_num:]]
    fea_df = fea_df.replace('.', 0)
    fea_df = fea_df.dropna()
    fea_df = normalize_fea(fea_df, fea_num)
    return fea_df
with open('../data/Mishra/all_feature.pkl', 'rb') as f:
    fea_df = pd.DataFrame(pickle.load(f))
fea_num=11
fea_df = load_data(fea_df, fea_num)
regr = linear_model.LinearRegression()
kf = KFold(n_splits=10, shuffle=True, random_state=233)
text_ids = list(set(fea_df['text_id']))
for train_text_i, test_text_i in kf.split(text_ids):
    train_index = [text_id in train_text_i for text_id in fea_df['text_id']]
    test_index = [text_id in test_text_i for text_id in fea_df['text_id']]
    train_df = fea_df[train_index]; test_df = fea_df[test_index]
    train_X = train_df.iloc[:, -fea_num:].values; train_Y = train_df.iloc[:,0].values
    test_X = test_df.iloc[:, -fea_num:].values; test_Y = test_df.iloc[:,0].values
    regr.fit(train_X, train_Y)
    predict_Y = regr.predict(test_X)
    rmse = np.sqrt(mean_squared_error(test_Y, predict_Y))
    print("rmse is %f, Y std is %f, Y mean is %f"%(rmse, test_Y.std(), test_Y.mean()))