import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import treebank
from nltk.parse.stanford import StanfordParser, StanfordDependencyParser
import os
from nltk.text import TextCollection
from features import feature_cal
import tqdm
import numpy as np







if __name__ == '__main__':
    # read data as dataframe format from pkl
    with open('MonoligualReadingData.pkl', 'rb') as f:
        eye_df = pd.DataFrame(pickle.load(f))

    with open('EnglighMaterial_all.pkl', 'rb') as f:
        english_material_all_df = pd.DataFrame(pickle.load(f))
    with open('EnglighMaterial_sentence.pkl', 'rb') as f:
        english_material_sentence_df = pd.DataFrame(pickle.load(f))

    # make TextCollection
    mytext = TextCollection(list(english_material_sentence_df['SENTENCE'].values))

    # construct word_id 2 sentence_id dict
    word2s = {}
    for i in range(english_material_all_df.shape[0]):
        word2s.update({english_material_all_df['WORD_ID'][i]: english_material_all_df['SENTENCE_ID'][i]})


    # init feature object
    feature_ob = feature_cal(mytext)

    # calculate features by each sentences
    words_list = []; sentence_id = '1-1'
    for i in tqdm.tqdm(range(eye_df.shape[0])):
        if eye_df['WORD_ID'][i] in word2s.keys() and word2s[eye_df['WORD_ID'][i]]!=sentence_id:
            tmp_features = feature_ob.get_feature(words_list)
            words_list = [str(eye_df['WORD'][i])]
            sentence_id = word2s[eye_df['WORD_ID'][i]]
            # print(np.array(tmp_features))
        else:
            words_list.append(str(eye_df['WORD'][i]))





