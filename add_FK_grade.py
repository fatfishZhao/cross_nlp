import textstat
import pandas as pd
import pickle
import tqdm


with open('feature.pkl' ,'rb') as f:
    old_df = pd.DataFrame(pickle.load(f))
all_grades = []
for i in tqdm.tqdm(range(old_df.shape[0])):
    word = old_df['WORD'][i]
    all_grades.append(textstat.flesch_kincaid_grade(str(word)))
old_df['fea_complexity'] = pd.Series(all_grades)
print(old_df.info())
with open('feature_FK.pkl', 'wb') as f:
    pickle.dump(old_df, f)