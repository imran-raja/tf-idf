"""
In order to run this file, type the following in terminal: 

python tfidf_gen.py path/to/your/folder

"""

import nltk
import string
import os
import io
import re
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict

#path = "/home/imran/PractisePython/tf-idf/20news-bydate/20news-bydate-train/alt.atheism"
#path = "/home/imran/Semion/SemionTextClassifier/rt-polaritydata/rt-polaritydata"
path = sys.argv[1]
stemmer = SnowballStemmer("english")

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems
    #return tokens

token_dict = defaultdict()
for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        lowers = io.open(file_path, 'r', encoding='latin-1').read().lower()
        no_punctuation = lowers.encode('utf-8').translate(string.maketrans("",""), string.punctuation)
        no_numbers = re.sub(r'\d+', '', no_punctuation)
        token_dict[file] = no_numbers
    print "The total number of files is: %s" % (len(files))  


tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfidf_matrix = tfidf.fit_transform(token_dict.values())
feature_names = tfidf.get_feature_names()


final_list = defaultdict()
doc_id = 0
print "\nThe tf-idf scores of the 10 most relevant words are:"

for doc in tfidf_matrix.todense():
    print "\nDocument %d" %(doc_id+1)
    word_id = 0
    for score in doc.tolist()[0]:
        if score > 0:
            word = feature_names[word_id]
            final_list[word.encode("utf-8")] = score
            word_id +=1
    for key, value in sorted(final_list.iteritems(), key=lambda (k,v): (v,k), reverse = True)[0:10]:
    	print "%s: %s" % (key, value)
    doc_id +=1
