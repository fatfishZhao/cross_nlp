import os
import torch
from torch import nn
import pickle
import pandas as pd
from sklearn import preprocessing
from ds import get_train_val, collate_fn
from experiment.train_util import train, trainlog
from torch.optim import lr_scheduler
import time

# define net
class RNN(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        outs = []  # save all predictions
        for time_step in range(r_out.size(1)):  # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1)
def normalize_time(old_df):
    people_list = list(old_df.drop_duplicates(subset='PP_NR', inplace=False)['PP_NR'])
    new_df = old_df.copy()
    for one_people in people_list:
        tmp_df = old_df[old_df['PP_NR']==one_people]
        sum_time = tmp_df['WORD_TOTAL_READING_TIME'].sum()
        normalize_time = tmp_df['WORD_TOTAL_READING_TIME'].values/sum_time
        new_df.loc[list(new_df[new_df['PP_NR']==one_people].index), ['WORD_TOTAL_READING_TIME']]=normalize_time
        # new_df[new_df['PP_NR']==one_people]['WORD_TOTAL_READING_TIME']=normalize_time
    return new_df
def normalize_fea(old_df, fea_num):
    new_fea = preprocessing.scale(old_df.iloc[:, -fea_num:].values, axis=0)
    new_df = old_df.copy()
    new_df.iloc[:, -fea_num:] = new_fea
    return new_df

def load_data(fea_path = '../feature.pkl', fea_num=10):
    with open(fea_path, 'rb') as f:
        fea_df = pd.DataFrame(pickle.load(f))
    fea_df = fea_df.replace('.', 0)
    fea_df = fea_df.dropna()
    # fea_df = normalize_time(old_df=fea_df)
    fea_df = normalize_fea(old_df=fea_df, fea_num=fea_num)

    fea_df = fea_df[['WORD_TOTAL_READING_TIME','WORD', 'WORD_ID']+list(fea_df.columns)[-fea_num:]]
    # reset index
    fea_df.index = range(fea_df.shape[0])


    return fea_df
# init dataset
feature_num = 11
DEBUG = False
data_df = load_data('../all_feature.pkl', feature_num)
if DEBUG:
    data_df = data_df.iloc[0:1000, :]
with open('../EnglighMaterial_all.pkl', 'rb') as f:
    english_matrial = pd.DataFrame(pickle.load(f))

save_dir = '/data3/zyx/project/eye_nlp/trained_model/'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logfile = '%s/trainlog.log'%save_dir
trainlog(logfile)
dataset = {}
dataset['train'], dataset['val'] = get_train_val(data_df, -feature_num, english_matrial)
dataloader={}
dataloader['train']=torch.utils.data.DataLoader(dataset['train'], batch_size=16,
                                               shuffle=True, num_workers=4,collate_fn=collate_fn)
dataloader['val']=torch.utils.data.DataLoader(dataset['val'], batch_size=16,
                                               shuffle=True, num_workers=4,collate_fn=collate_fn)

print(data_df.info())
rnn = RNN(INPUT_SIZE=feature_num)
rnn.cuda()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)  # optimize all cnn parameters
loss_func = nn.MSELoss()
h_state = None  # for initial hidden state
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
train(model=rnn,
      epoch_num=20,
      start_epoch=0,
      optimizer=optimizer,
      criterion=loss_func,
      exp_lr_scheduler=exp_lr_scheduler,
      data_set=dataset,
      data_loader=dataloader,
      save_dir=save_dir,
      print_inter=50,
      val_inter=1000
    )