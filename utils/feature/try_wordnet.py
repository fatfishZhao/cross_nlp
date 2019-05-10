import nltk
from nltk.corpus import wordnet as wn

print(len(wn.synsets('bank')))
for ss in wn.synsets('bank'):
    print(ss, ss.definition())