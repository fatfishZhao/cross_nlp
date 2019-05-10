import nltk
from nltk.corpus import treebank
from nltk.parse.stanford import StanfordParser, StanfordDependencyParser
import os
import math
def traverse(deps, addr):

    dep = deps.get_by_address(addr)
    print(dep)
    for d in dep['deps']:
        for addr2 in dep['deps'][d]:
            traverse(deps, addr2)

# code on book
dep_parser = StanfordDependencyParser('/data3/zyx/project/eye_nlp/data/model/stanford-parser.jar',
                                '/data3/zyx/project/eye_nlp/data/model/stanford-parser-3.9.2-models.jar',model_path='/data3/zyx/project/eye_nlp/data/model/englishPCFG.ser.gz')
# print(list(english_parser.raw_parse_sents(('this is the english parser test'))))
# [list(parse.triples()) for parse in dep_parser.raw_parse("The quick brown fox jumps over the lazy dog.")]

s = 'the big dog chased the little cat'
s = 'Your  mother  keeps  well?  I  asked.'
res = dep_parser.parse(s.split()) # can use a simple .split since my input is already tokenised
deps = res.__next__()
traverse(deps, 0) # 0 is always the root node

features = []
for i in range(1,len(deps.nodes)):
    tmp_max = 0
    if deps.nodes[i]['head']!=0:
        tmp_max = max(tmp_max, math.fabs(i-deps.nodes[i]['head']))
    for each_dep in deps.nodes[i]['deps']:
        for each_index in deps.nodes[i]['deps'][each_dep]:
            tmp_max = max(tmp_max,math.fabs(i-each_index))
    features.append(tmp_max)
print(features)