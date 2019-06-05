import matplotlib.pyplot as plt
import pickle
import pandas as pd

with open('../data/EGCO/all_feature.pkl', 'rb') as f:
    feature_all = pd.DataFrame(pickle.load(f))
fea_num=11
feature_all = feature_all.replace('.', 0)
feature_all = feature_all.dropna()
y = feature_all['WORD_TOTAL_READING_TIME'].values
n, bins, patches = plt.hist(y, 500, density=True, facecolor='g', alpha=0.75)


plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('Fixation time')
plt.xlim(0,1000)
plt.grid(True)
plt.savefig('visual/time.png')
for i in range(1,fea_num+1):
    print(feature_all.columns[-i])
    x = feature_all[feature_all.columns[-i]].values
    plt.figure()
    plt.hist(x, 100, normed=True, facecolor='g')
    plt.xlabel('value')
    plt.ylabel('Probability')
    plt.title(feature_all.columns[-i])
    plt.savefig('visual/'+feature_all.columns[-i]+'.png')

    plt.figure()
    plt.scatter(x,y)
    plt.xlabel('value')
    plt.ylabel('Reading time')
    plt.title(feature_all.columns[-i]+'-y')
    plt.savefig('visual/' + feature_all.columns[-i] + '-y.png')
