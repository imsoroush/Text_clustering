import numpy as np
import pandas as pd
from math import log
import collections
from collections import Counter
import re
import minisom
from minisom import MiniSom
from sklearn.preprocessing import minmax_scale

# input output laoded data
x = data_txt.loc[:]['text']
y = data_txt.loc[:]['category']
class_labels = y.unique()

#Remove all non-letter characters from the documents
x = [re.sub("[^a-zA-Z ]+", "",txt) for txt in x] 
#Extract all words of the document and remove the short words (length ≤ 2)
x = [re.sub(r'\b\w{1,2}\b','',txt) for txt in x] 
target = np.zeros(len(y), dtype=int)
# label
y_array = y.to_numpy(y)
target[y_array == 'business'] = 0
target[y_array == 'entertainment'] = 1
target[y_array == 'politics'] = 2
target[y_array == 'sport'] = 3
target[y_array == 'tech'] = 4

#Remove all stop words (e.g., ‘a’, ‘and’, ‘what’, …), given in file ‘stopwords.txt
stopw_list = list(stop_words.columns)[0].split() 
docs_list = list()
docs_list_split = list()
for txt in x:
    txt = txt.split()
    txt_n = ""
    for word in txt:
        if(word in stopw_list) == False:
            txt_n =txt_n+word+" "
    docs_list_split.append(txt_n.split())
    docs_list.append(txt_n)

# Compute the feature vector for each document, using TF-IDF weighting scheme.
def tfidf(doc, _corpus): 
    dic = collections.defaultdict(int)
    for x in _corpus:
        for y in x:
            dic[y] += 1.
    return {x: doc[x] * log(len(_corpus) / dic[x]) for x in doc}
doc_sets = [Counter(doc.split()) for doc in docs_list]
docs_list_tfidf = [tfidf(x, doc_sets) for x in doc_sets]

# SOM Winner Takes All
sample_num, words_num = df_tfidf_docs.shape
som_shape = (1,5)
som = MiniSom(som_shape[0], som_shape[1], words_num, sigma= 2 , learning_rate= 0.25, 
              activation_distance='euclidean')
som.random_weights_init(data)
som.train_batch(data, 5000 , verbose=True)

# SOM On Center Off Surround
som_shape = (5,5)
som = MiniSom(som_shape[0], som_shape[1], words_num, sigma= 3.5, learning_rate=0.25,
              neighborhood_function='gaussian',topology='hexagonal', 
              activation_distance='euclidean')
som.random_weights_init(data)
som.train_batch(data, 5000, verbose=True)
