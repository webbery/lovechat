'''
本文件提供了构建词向量的方式
'''
from gensim.models import FastText

class FastTextModel():
    def __init__(self,sentences):
        self.model = FastText(sentences,size=4, window=3, min_count=1, iter=10,min_n = 3 , max_n = 6,word_ngrams = 0)

    def __getitem__(self, i):
        return self.model.wv[i]