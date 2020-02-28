from fse.models import uSIF
from fse import SplitIndexedList
from gensim.models import FastText
import jieba

class SIFSpace():
    def __init__(self, questions):
        self.ft = FastText(questions, min_count=1, size=10)
        self.model = uSIF(self.ft)
        self.model.train(SplitIndexedList(questions))
        # self.candidates = questions
        sample = []
        for idx,item in enumerate(questions):
            sample.append((jieba.lcut(item),idx))
        self.model.infer(sample)

    def find_similarity(self,input,tops=10):
        val = self.model.sv.similar_by_sentence(jieba.lcut(input),model=self.model)
        indexes = list(zip(*val))[0]
        return list(indexes[0:tops])

    # def find_similarity(self,input,candidates,tops = 1):
    #     model = uSIF(self.ft)
    #     model.train(SplitIndexedList(candidates))
    #     sample = []
    #     for idx,item in enumerate(candidates):
    #         sample.append((jieba.lcut(item),idx))
            
    #     # sample.append((jieba.lcut(input),len(candidates)))
    #     model.infer(sample)
    #     result = model.sv.similar_by_sentence(jieba.lcut(input),model=model)
    #     result = list(zip(*result))
    #     # print(result)
    #     rg = list(result[0])
    #     # print(candidates)
    #     # print(len(candidates),len(result[1]),len(rg))
    #     # print(result)

    #     reply = list(zip(*[(candidates[index] for index in rg), (result[1])]))
    #     # reply = sorted(reply,key=lambda x: x[1],reverse=True)
    #     print(reply)
    #     return reply[0:tops]
