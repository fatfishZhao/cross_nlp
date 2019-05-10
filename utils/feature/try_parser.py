import os
from nltk.parse.stanford import StanfordParser, StanfordDependencyParser

os.environ['STANFORD_PARSER'] = '/data3/zyx/project/eye_nlp/data/model/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/data3/zyx/project/eye_nlp/data/model/stanford-parser-3.9.2-models.jar'

parser = StanfordParser(model_path='/data3/zyx/project/eye_nlp/data/model/englishPCFG.ser.gz')
parser.raw_parse("the quick brown fox jumps over the lazy dog")
sentences = parser.raw_parse_sents(("Hello, My name is Melroy.", "What is your name?"))


# GUI
for line in sentences:
    for sentence in line:
        sentence.draw()