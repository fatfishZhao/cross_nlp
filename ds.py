import numpy as np
import torch.utils.data as data
import torch
import tqdm
from sklearn.model_selection import train_test_split

def get_train_val(data_df, fea_start, english_material_all_df, transforms=None):
    word2s = {}
    for i in range(english_material_all_df.shape[0]):
        word2s.update({english_material_all_df['WORD_ID'][i]: english_material_all_df['SENTENCE_ID'][i]})
    sentence_id = english_material_all_df['SENTENCE_ID'][0]

    sen_data = []
    max_length = 0
    words_list = []
    for i in tqdm.tqdm(range(data_df.shape[0])):
        if data_df['WORD_ID'][i] in word2s.keys() and word2s[data_df['WORD_ID'][i]] != sentence_id:
            sen_data.append((i - len(words_list), len(words_list)))
            if len(words_list) > max_length:
                max_length = len(words_list)

            words_list = [str(data_df['WORD'][i])]
            sentence_id = word2s[data_df['WORD_ID'][i]]
        elif i == data_df.shape[0] - 1:
            words_list.append(str(data_df['WORD'][i]))
            sen_data.append((i - len(words_list) + 1, len(words_list)))
        else:
            words_list.append(str(data_df['WORD'][i]))
    train_sen_data, val_sen_data = train_test_split(sen_data, test_size=0.1, shuffle=True)
    train_dataset = dataset(data_df, fea_start, train_sen_data, max_length, transforms)
    val_dataset = dataset(data_df, fea_start, val_sen_data, max_length, transforms)
    return train_dataset, val_dataset

class dataset(data.Dataset):
    def __init__(self, data_df, fea_start, sen_data, max_length, transforms=None):
        '''

        :param data_df: dataframe including words and corresponding features
        :param fea_start: start column of the feature
        :param english_material_all_df:
        :param transforms: transform methods
        '''
        # split words into sentences, save the start index and length for each sentences,
        #   calculate the max words in one sentence
        self.fea_start = fea_start
        self.data_df = data_df
        self.sen_data = sen_data
        self.transforms = transforms
        self.max_length = max_length

    def __len__(self):
        return len(self.sen_data)

    def __getitem__(self, item):
        data = self.data_df.iloc[self.sen_data[item][0]:self.sen_data[item][0]+self.sen_data[item][1],
                                    self.fea_start:].values
        data = np.concatenate((data, np.zeros((self.max_length-data.shape[0], data.shape[1]))), axis=0)
        label = self.data_df.iloc[self.sen_data[item][0]:self.sen_data[item][0]+self.sen_data[item][1],
                                    0].values
        word_length = label.shape[0]
        label = np.concatenate((label, np.zeros(self.max_length-label.shape[0])))
        label = label[:, np.newaxis]


        if self.transforms is not None:
            data = self.transforms(data)

        return torch.from_numpy(data).float(), label, word_length


def collate_fn(batch):
    data = []
    label = []
    word_length = []

    for sample in batch:
        data.append(sample[0])
        label.append(sample[1])
        word_length.append(sample[2])

    return torch.stack(data, 0), \
           label, word_length