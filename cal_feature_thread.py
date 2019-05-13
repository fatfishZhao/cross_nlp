# multi processes for feature calculation

import queue
import threading
from nltk.text import TextCollection
import pandas as pd
import pickle
import tqdm
from features import feature_cal
from nltk.corpus import wordnet as wn
import numpy as np
import nltk

class myThread (threading.Thread):
    def __init__(self, threadID, q, feature_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.q = q
        self.feature_ob = feature_cal(mytext)
        self.feature_list = feature_list
    def run(self):
        print ("开启线程：" + str(self.threadID))
        process_data(self.threadID, self.q, self.feature_ob, self.feature_list)
        print ("退出线程：" + str(self.threadID))
def process_data(threadID, q, feature_ob, feature_list):
    while 1:
        queueLock.acquire()
        if not q.empty():
            data = q.get()
            queueLock.release()
            tmp_feature = feature_ob.get_feature(data[1], wn)
            # tmp_feature = 1
            feature_list.append((data[0], tmp_feature))
            print ("%s processing %s" % (threadID, data))
        else:
            queueLock.release()
if __name__ == '__main__':

    # print(wn.__class__)  # <class 'nltk.corpus.util.LazyCorpusLoader'>
    # wn.ensure_loaded()  # first access to wn transforms it
    # print(wn.__class__)  # <class 'nltk.corpus.reader.wordnet.WordNetCorpusReader'>

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

    thread_num = 20
    queueLock = threading.Lock()
    workQueue = queue.Queue(thread_num*2)

    feature_list = []
    threads = []

    # 创建新线程
    for t_i in range(thread_num):
        feature_list.append([])
        thread = myThread(t_i, workQueue, feature_list[t_i])
        thread.start()
        threads.append(thread)

    # calculate features by each sentences
    words_list = []
    sentence_id = '1-1'
    fea_num_wordnet = []
    tokenizer = nltk.tokenize.RegexpTokenizer('\w+')
    for i in tqdm.tqdm(range(eye_df.shape[0])):
    # for i in tqdm.tqdm(range(1000)):
        fea_num_wordnet.append(len(wn.synsets(tokenizer.tokenize(str(eye_df['WORD'][i]))[0])))
        if eye_df['WORD_ID'][i] in word2s.keys() and word2s[eye_df['WORD_ID'][i]] != sentence_id:
            workQueue.put((i, words_list))

            words_list = [str(eye_df['WORD'][i])]
            sentence_id = word2s[eye_df['WORD_ID'][i]]
        elif i == eye_df.shape[0] - 1:
            words_list.append(str(eye_df['WORD'][i]))
            workQueue.put((i+1, words_list))
        else:
            words_list.append(str(eye_df['WORD'][i]))
    # 等待队列清空
    while not workQueue.empty():
        pass
    # 等待所有线程完成
    for t in threads:
        t.join(0.01)
    features_df = pd.DataFrame(
        columns=['fea_num_letter', 'fea_start_capital', 'fea_capital_only', 'fea_have_num', 'fea_abbre',
                 'fea_domi_nodes', 'fea_max_d',
                 'fea_idf'])
    new_df = pd.concat([eye_df, features_df], axis=1)
    # insert data to the dataframe
    for one_list in feature_list:
        for one_sentence in one_list:
            end_index = one_sentence[0]
            feature_array = np.array(one_sentence[1]).T
            s_length = feature_array.shape[0]
            fea_num = feature_array.shape[1]
            new_df.iloc[end_index-s_length:end_index, eye_df.shape[1]:eye_df.shape[1]+fea_num] = feature_array
    new_df['fea_num_wordnet'] = pd.Series(fea_num_wordnet)
    with open('feature.pkl', 'wb') as f:
        pickle.dump(new_df, f)
    print(new_df.info())






