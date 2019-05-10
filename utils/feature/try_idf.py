from nltk.text import TextCollection

mytext = TextCollection(['The cat hit that dog'.lower(),
                         'This is a fat dog'.lower(),
                         'I like this dog',
                         'I like to go to school'])
print(mytext.idf('this'))