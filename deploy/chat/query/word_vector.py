'''
本文件提供了构建词向量的方式
'''
from gensim.models import FastText

class FastTextModel():
    def __init__(self,sentences=None):
        if sentences!=None:
            self.model = FastText(sentences,size=4, window=3, min_count=1, iter=10,min_n = 3 , max_n = 6,word_ngrams = 0)

    def __getitem__(self, i):
        return self.model.wv[i]

    def get_vector(self,words):
        if type(words)==str:
            return self.model.wv[words]
        if type(words)==list:
            vectors=[]
            for word in words:
                vectors.append(self.model.wv[word])
            return vectors

    def load_model(self,filename):
        self.model = FastText.load(filename)

model = FastTextModel()
model.load_model('fasttext.model')