import pandas as pd
# 线性回归
from sklearn import datasets, linear_model, cross_validation
import pickle
import os
import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


import numpy as np
def load_data(data_df, fea_num, word2s):
    sentence_set = list(pd.Series(list(set(word2s.values()))).dropna())
    data_df = data_df.replace('.', 0)
    data_df = data_df.dropna()
    train_sen, test_sen = train_test_split(sentence_set, test_size=0.1)
    train_index = [word2s[word_id] in train_sen for word_id in data_df['WORD_ID']]
    test_index = [word2s[word_id] in test_sen for word_id in data_df['WORD_ID']]

    data_df = data_df[['WORD_TOTAL_READING_TIME','WORD']+list(data_df.columns)[-fea_num:]]

    train_df = data_df[train_index]
    test_df = data_df[test_index]
    train_X = train_df.iloc[:,2:].values
    train_Y = train_df.iloc[:,0].values
    test_X = test_df.iloc[:, 2:].values
    test_Y = test_df.iloc[:, 0].values

    return train_X, train_Y, test_X, test_Y
# 映射word_id -> sentence_id
def get_word2sen(data_df, material_file='../data/ECGO/EnglighMaterial_all.pkl', save_file = '../data/ECGO/word2sen.pkl'):
    if os.path.exists(save_file):
        with open(save_file, 'rb') as f:
            word2s = dict(pickle.load(f))
    else:
        with open(material_file, 'rb') as f:
            english_matrial = pd.DataFrame(pickle.load(f))
        word2s = {}
        for i in range(english_matrial.shape[0]):
            word2s.update({english_matrial['WORD_ID'][i]: english_matrial['SENTENCE_ID'][i]})
        sentence_id = '1-1'
        for i in tqdm.tqdm(range(data_df.shape[0])):
            if not data_df['WORD_ID'][i] in word2s.keys():
                word2s.update({data_df['WORD_ID'][i]: sentence_id})
            elif word2s[data_df['WORD_ID'][i]] != sentence_id:
                sentence_id = word2s[data_df['WORD_ID'][i]]
        with open(save_file, 'wb') as f:
            pickle.dump(word2s, f)
    return word2s
with open('../data/ECGO/all_feature.pkl', 'rb') as f:
    data_df = pd.DataFrame(pickle.load(f))
word2sen = get_word2sen(data_df)

train_X, train_Y, test_X, test_Y = load_data(data_df, fea_num=11, word2s=word2sen)

regr = linear_model.LinearRegression()
regr.fit(train_X, train_Y)
predict_Y = regr.predict(test_X)
rmse = np.sqrt(mean_squared_error(test_Y, predict_Y))
print('rmse is ', rmse)
print('std is ', test_Y.std())
