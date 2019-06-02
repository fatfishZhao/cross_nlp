import matplotlib.pyplot as plt
import pickle
import pandas as pd

with open('../all_feature.pkl', 'rb') as f:
    feature_all = pd.DataFrame(pickle.load(f))
fea_num=11
y = feature_all['WORD_TOTAL_READING_TIME'].replace('.', 0).values
n, bins, patches = plt.hist(y, 50, density=True, facecolor='g', alpha=0.75)


plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.grid(True)
plt.show()