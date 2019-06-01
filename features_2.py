import re
from nltk.parse.stanford import StanfordParser, StanfordDependencyParser
import math
import spacy
# from nltk.corpus import wordnet as wn
import nltk
import textstat



def traverse(deps, addr):

    dep = deps.get_by_address(addr)
    for d in dep['deps']:
        for addr2 in dep['deps'][d]:
            traverse(deps, addr2)
def cal_dominate(dep, index):
    tmp_dominate = 0
    for each_dep in dep.nodes[index]['deps']:
        for each_index in dep.nodes[index]['deps'][each_dep]:
            tmp_dominate += cal_dominate(dep, each_index)
    return tmp_dominate+1
def cal_max_d(deps, word_num):
    features = []
    for i in range(1, word_num+1):
        tmp_max = 0
        if deps.nodes[i]['head'] != 0:
            tmp_max = max(tmp_max, math.fabs(i - deps.nodes[i]['head']))
        for each_dep in deps.nodes[i]['deps']:
            for each_index in deps.nodes[i]['deps'][each_dep]:
                tmp_max = max(tmp_max, math.fabs(i - each_index))
        features.append(tmp_max)
    return features
def cal_idf(text_collector, words):
    idfs = []
    for word in words:
        idfs.append(text_collector.idf(word))
    return idfs
def cal_entity_critical(nlp, words_list):
    sentence = ''
    word_pos = []
    tmp_index = 0
    for word in words_list:
        sentence += word + ' '
        word_pos.append([tmp_index, tmp_index + len(word)])
        tmp_index += len(word) + 1
    fea_entity_critical = []


    doc = nlp(sentence)

    for one_pos in word_pos:
        entity_critical = False
        for ent in doc.ents:
            if not (one_pos[1] <= ent.start_char or one_pos[0] >= ent.end_char):
                entity_critical = True
        fea_entity_critical.append(entity_critical)
    return fea_entity_critical
class feature_cal():

    def __init__(self, text_collector):
        # wn.ensure_loaded()
        self.text_collector = text_collector
        self.dep_parser = StanfordDependencyParser('/data3/zyx/project/eye_nlp/data/model/stanford-parser.jar',
                                              '/data3/zyx/project/eye_nlp/data/model/stanford-parser-3.9.2-models.jar',
                                              model_path='/data3/zyx/project/eye_nlp/data/model/englishPCFG.ser.gz')
        self.tokenizer = nltk.tokenize.RegexpTokenizer('\w+')
        self.nlp = spacy.load("en_core_web_sm")

    def get_feature(self, words_list, wn):

        raw_words_list = [self.tokenizer.tokenize(word)[0] for word in words_list]


        fea_num_letter = [len(word) for word in raw_words_list]
        fea_start_capital = [word.istitle() for word in raw_words_list]
        fea_capital_only = [word.isupper() for word in raw_words_list]
        fea_have_num = [True if re.match(r'[+-]?\d+$', word) else False for word in raw_words_list]
        fea_abbre = [word.isupper() and len(word)>=2 for word in raw_words_list]
        fea_entity_critical = cal_entity_critical(self.nlp, words_list)

        # use nlp method

        doc = self.nlp()
        res = self.dep_parser.parse(words_list)
        deps = res.__next__()
        traverse(deps, 0)  # 0 is always the root node
        fea_domi_nodes = []
        for i in range(1, len(words_list)+1):
            this_dominate = cal_dominate(deps, i)
            fea_domi_nodes.append(this_dominate)

        fea_max_d = cal_max_d(deps, len(words_list))

        fea_idf = cal_idf(self.text_collector, raw_words_list)
        if len(fea_max_d)!=len(fea_have_num):
            print('length error')
        # fea_num_wordnet = [len(wn.synsets(word)) for word in raw_words_list]
        fea_complexity = [textstat.flesch_kincaid_grade(str(word)) for word in words_list]
        return [fea_num_letter, fea_start_capital, fea_capital_only, fea_have_num, fea_abbre, fea_entity_critical, fea_domi_nodes, fea_max_d,
                fea_idf, fea_complexity]
if __name__=='__main__':
    words_list = 'Apple, Google are looking at buying U.K. startup for $1 billion.'.split()
