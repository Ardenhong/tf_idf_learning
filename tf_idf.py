import pandas as pd
import pprint
from math import log
import pprint as pp
# dataset : https://clay-atlas.com/blog/2020/08/01/nlp-%E6%96%87%E5%AD%97%E6%8E%A2%E5%8B%98%E4%B8%AD%E7%9A%84-tf-idf-%E6%8A%80%E8%A1%93/
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


# remove word len<2
for i in range(len(doc_all)):
    doc_all[i] = doc_all[i].split()
    doc_all[i] = [ w for w in doc_all[i] if len(w) >=2 ]
    doc_all[i] = ' '.join(doc_all[i])

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
        
    pp.pprint("tf :")
    pp.pprint(tf)
    print("----------")
    

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

    for word in words:
        for sentence in doc_all:
            if word_in_sentence(word, sentence):
                document_count[word] += 1

    # df = {}
    idf = {}
    # for word, d_count in document_count.items():
    #     df[word] = d_count / len(doc_all)

    for word, d_count in document_count.items():
        idf[word] = len(doc_all) / d_count

    # print(f"DF : {df}")

    # print(f"\n IDF before log")
    # pprint.pprint(idf)

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
            tfidf[word] = tf[word]*idf[word]
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
