import pandas as pd
import pprint
from math import log

# implement tfidf,ref:  https://ithelp.ithome.com.tw/articles/10214726


# from sklearn.feature_extraction.text import TfidfVectorizer


# pre process
doc_0 = 'Today is a nice day'
doc_1 = 'Today is a bad day'
doc_2 = 'Today I want to play all day'
doc_3 = 'I went to play all day yesterday'
doc_all = [doc_0, doc_1, doc_2, doc_3]

# normalize doc
doc_all = [sentence.lower() for sentence in doc_all]

all_word_lower_str = ''
for sentence in doc_all:
    all_word_lower_str += f' {sentence.lower()}'

'''
# TF
lower all sentences
put all word in a dict
cal each word freq in all sentences 
'''


def cal_tf(all_word_lower_str) -> dict:
    term_count = {}
    for word in all_word_lower_str.split():
        if word not in term_count:
            term_count[word] = 1
        else:
            term_count[word] += 1

    tf = {}
    for word in term_count:
        tf[word] = term_count[word] / len(term_count)

    return tf


tf = cal_tf(all_word_lower_str)


def cal_idf(all_word_lower_str, doc_all):
    def word_in_sentence(word, sentence) -> bool:
        for _w in sentence.split():
            if word == _w:
                return True

        return False

    document_count = {}
    words = set()
    for word in all_word_lower_str.split():
        words.add(word)

    # init dict
    for word in words:
        document_count[word] = 0

    for sentence in doc_all:
        for word in words:
            if word_in_sentence(word, sentence):
                document_count[word] += 1

    df = {}
    idf = {}
    for word, d_count in document_count.items():
        df[word] = d_count / len(doc_all)

    for word, d_count in document_count.items():
        idf[word] = len(doc_all) / d_count

    print(f"DF : {df}")

    print(f"\n IDF before log")
    pprint.pprint(idf)

    for word, _num in idf.items():
        idf[word] = log(_num)

    print(f"\n IDF after log")
    pprint.pprint(idf)

    return idf


idf =  cal_idf(all_word_lower_str, doc_all)

words = set()
for word in all_word_lower_str.split():
    words.add(word)


def cal_tfidf(tf, idf, words):
    tfidf = {}
    for word in words:
        if idf[word]:
            tfidf[word] = tf[word]/idf[word]
        else:
            tfidf[word] = 0
    return tfidf

tfidf = cal_tfidf(tf, idf, words)

print('TF-IDF:')
pprint.pprint(tfidf)
# print(doc_all)


#
# # TF-IDF
# vectorizer = TfidfVectorizer(smooth_idf=True)
# tfidf = vectorizer.fit_transform(doc_all)
# result = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
# print('Scikit-Learn:')
# print(result)
