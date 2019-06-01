import spacy
import time
from nltk.parse.stanford import StanfordParser, StanfordDependencyParser
import pickle
import pandas as pd
import tqdm


words_list = 'Apple is looking at buying U.K. startup for $1 billion.'.split()
sentence = ''
word_pos = []
tmp_index = 0
for word in words_list:
    sentence += word+' '
    word_pos.append([tmp_index, tmp_index+len(word)])
    tmp_index += len(word)+1
fea_entity_critical = []


nlp = spacy.load("en_core_web_sm")
doc = nlp(sentence)
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)
for one_pos in word_pos:
    entity_critical = False
    for ent in doc.ents:
        if not(one_pos[1]<=ent.start_char or one_pos[0]>=ent.end_char):
            entity_critical = True
    fea_entity_critical.append(entity_critical)
print(fea_entity_critical)
