from scipy.stats import pearsonr
import pickle
import pandas as pd

with open('../data/EGCO/all_feature.pkl', 'rb') as f:
    feature_all = pd.DataFrame(pickle.load(f))
feature_all = feature_all.replace('.', 0)
feature_all = feature_all.dropna()
fea_num=11
y = feature_all['WORD_TOTAL_READING_TIME'].values

for i in range(1,fea_num+1):
    x = feature_all[feature_all.columns[-i]].values
    corr = pearsonr(x, y)
    print('corr between '+feature_all.columns[-i]+' and reading time is '+str(corr))
