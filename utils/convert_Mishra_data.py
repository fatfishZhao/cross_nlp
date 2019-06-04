import nltk
import pandas as pd
import tqdm

raw_fix_data = pd.read_csv('/data3/zyx/project/eye_nlp/data/Mishra/Eye-tracking_and_SA-II_released_dataset/Fixation_sequence.csv')
material_data = pd.read_csv('/data3/zyx/project/eye_nlp/data/Mishra/Eye-tracking_and_SA-II_released_dataset/text_and_annorations.csv')
raw_fix_data['Word_ID'] = raw_fix_data['Word_ID']-1
raw_fix_data = raw_fix_data[raw_fix_data['Word_ID']!=0]
peoples = list(set(raw_fix_data['Participant_ID']))
peoples.sort()
participant_id_list = []
text_id_list = []
word_id_list = []
word_list = []
fixation_duration_list = []
tokenizer = nltk.tokenize.RegexpTokenizer('\w+')
for participant in peoples:
    print('now participant is ', participant)
    each_fix_data = raw_fix_data[raw_fix_data['Participant_ID']==participant]
    for text_index, row in tqdm.tqdm(material_data.iterrows()):
        words_in_s = row['Text'].split()
        for word_index, tmp_word in enumerate(words_in_s):
            words = tokenizer.tokenize(str(tmp_word))
            if len(words)==0:
                print('none')
                break
            match_fix_data = each_fix_data[each_fix_data['Word_ID']==word_index+1]
            match_fix_data = match_fix_data[match_fix_data['Text_ID']==text_index+1]
            if match_fix_data.shape[0]==0:
                fixation_duration_list.append(0)
            else:
                fixation_duration_list.append(match_fix_data['Fixation_Duration'].sum())
            participant_id_list.append(participant)
            text_id_list.append(text_index+1)
            word_id_list.append(word_index+1)
            word_list.append(str(tmp_word))
new_data_df = pd.DataFrame({'participant_id':participant_id_list,
                            'text_id': text_id_list,
                            'word_id':word_id_list,
                            'word':word_list,
                            'fixation_duration':fixation_duration_list})
import pickle
with open('../data/Mishra/fixation_data.pkl', 'wb') as f:
    pickle.dump(new_data_df, f)