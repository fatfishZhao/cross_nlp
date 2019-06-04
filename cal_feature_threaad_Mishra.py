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
    debug = False
    with open('./data/Mishra/fixation_data.pkl', 'rb') as f:
        eye_df = pd.DataFrame(pickle.load(f))
        if debug:
            eye_df = eye_df.iloc[0:1000, :]
    material_df = pd.read_csv('../data/Mishra/Eye-tracking_and_SA-II_released_dataset/text_and_annorations.csv')
    # make TextCollection
    mytext = TextCollection(list(material_df['Text'].values))

    thread_num = 10
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
    sentence_id = 1
    fea_num_wordnet = []
    tokenizer = nltk.tokenize.RegexpTokenizer('\w+')
    for i in tqdm.tqdm(range(eye_df.shape[0])):
        fea_num_wordnet.append(len(wn.synsets(tokenizer.tokenize(str(eye_df['word'][i]))[0])))
        if eye_df['text_id'][i] != sentence_id:
            workQueue.put((i, words_list))
            words_list = [str(eye_df['word'][i])]
            sentence_id = eye_df['text_id'][i]
        elif i==eye_df.shape[0]-1:
            words_list.append(str(eye_df['word'][i]))
            workQueue.put((i + 1, words_list))
        else:
            words_list.append(str(eye_df['word'][i]))
    # 等待队列清空
    while not workQueue.empty():
        pass
    import time
    time.sleep(20)
    # 等待所有线程完成
    for t in threads:
        t.join(0.01)
    features_df = pd.DataFrame(
        columns=['fea_num_letter', 'fea_start_capital', 'fea_capital_only', 'fea_have_num', 'fea_abbre', 'fea_critical_entity',
                 'fea_domi_nodes', 'fea_max_d',
                 'fea_idf', 'fea_complexity'])
    new_df = pd.concat([eye_df, features_df], axis=1)
    # insert data to the dataframe
    print('start inserting to dataframe')
    all_feature_array = np.zeros((new_df.shape[0], len(features_df.columns)))
    for one_list in tqdm.tqdm(feature_list):
        for one_sentence in one_list:
            end_index = one_sentence[0]
            feature_array = np.array(one_sentence[1]).T
            s_length = feature_array.shape[0]
            fea_num = feature_array.shape[1]
            all_feature_array[end_index - s_length: end_index, : fea_num] = feature_array
            # new_df.iloc[end_index-s_length:end_index, eye_df.shape[1]:eye_df.shape[1]+fea_num] = feature_array
    new_df.iloc[:, eye_df.shape[1]:eye_df.shape[1]+len(features_df.columns)] = all_feature_array
    new_df['fea_num_wordnet'] = pd.Series(fea_num_wordnet)
    with open('./data/Mishra/all_feature.pkl', 'wb') as f:
        pickle.dump(new_df, f)
    print(new_df.info())
    print('save dataframe over...........')






