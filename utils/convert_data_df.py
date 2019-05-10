import pandas as pd
import pickle


english_material_all = pd.read_excel('/data3/zyx/project/eye_nlp/data/EGCO/EnglishMaterial.xlsx', sheet_name='ALL')
english_material_sentence = pd.read_excel('/data3/zyx/project/eye_nlp/data/EGCO/EnglishMaterial.xlsx', sheet_name='SENTENCE')

with open('../EnglighMaterial_all.pkl', 'wb') as f:
    pickle.dump(english_material_all, f)
with open('../EnglighMaterial_sentence.pkl', 'wb') as f:
    pickle.dump(english_material_sentence, f)