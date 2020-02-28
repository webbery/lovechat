from deploy.chat.query.word_vector import model as WordPool
import pandas as pd
import numpy as np
from deploy.chat.query.intention_classify import intention,intention_classify
from pandas import Series
from deploy.chat.similarity.simhash import SimHashSpace
from deploy.chat.similarity.sentence import sentence2vec, get_similarity_with_tfidf
from deploy.chat.similarity.SementicKDSpace import SementicSpace
from deploy.chat.similarity.SIFSpace import SIFSpace
from deploy.chat.similarity.BertSimilarity import BertSimilarity

class Corpus():
    def __init__(self,corpus_file):
        super().__init__()
        # 1. 加载语料、预生成的语义向量及它们的hash
        self.corpus = pd.read_pickle(corpus_file)
        # 2. 构建simhash
        self.simhash = SimHashSpace(self.corpus['question_hash'])
        # 3. 构建语义向量KD树
        self.kdtree = SementicSpace(self.corpus['question_vector'])
        # 4. 构建SIF模型
        self.sif = SIFSpace(self.corpus['question'].tolist())
        print('corpus inited')
        # 5. 载入bert模型
        self.bert = BertSimilarity('bert.pkl')

    def max_similarity(self,tops):
        '''
        在所有候选句子中寻找得分最高的句子
        '''
        if len(tops)==0: return None

    def generate_sentence_by_model(self,input):
        '''
        根据深度学习模型生成句子
        '''
        pass

    def generate_sentence_by_rule(self,input,rule):
        '''
        根据规则生成句子
        '''
        pass

    def find_sentence_by_similarity(self,input,tops=10):
        '''
        检索匹配的句子
        '''
        
        # 1. 利用LHS方法检索匹配的句子
        indexes = self.simhash.find_similarity_indexes(input)
        answer = self.corpus['answer'][indexes]
        reply = answer.tolist()
        if len(reply)>5: reply = reply[0:5]
        #2. 使用sif查询相近句子
        indexes = self.sif.find_similarity(input,tops=5)
        reply += self.corpus['answer'][indexes].tolist()

        # 2. 从预分类的KD树中检索匹配的句子
        rest_cnt = 15-len(reply)
        _, indexes = self.kdtree.find_similarity_indexes(input,rest_cnt)

        reply += self.corpus['answer'][indexes].tolist()
        print(reply)
        
        reply = self.bert.find_most_similarity(input,reply)
        # print(reply)
        # sentence = top20[top[0][0]]
        # print(reply)
        return reply

    def get_sentence(self,index):
        return self.corpus['answer'].iloc[index]

# 预加载语料对象
corpus = Corpus('qa_corpus.pkl')