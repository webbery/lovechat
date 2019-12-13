from sklearn.neighbors import KDTree
import deploy.cut_sentence as cs
from deploy.word_vector import model as WordPool
import pandas as pd
import numpy as np
from deploy.intention_classify import intention,intention_classify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pandas import Series
from deploy.bool_search import bool_search as Search
from deploy.bool_search import BoolSearch

class SementicSpace():
    def __init__(self,classes_data):
        '''
        构建KD树
        '''
        points = len(classes_data)
        leaf = points/2-1
        self.tree = KDTree(classes_data,leaf)
        
    def __vectorize__(self,sentence):
        x_quest = cs.segment(sentence,'arr')
        return [sum([WordPool.get_vector(word) for word in x_quest])/len(x_quest)]
    
    def similarity(self,sentence,topk=10):
        '''
        从KD树中找到最相似的k个数据
        '''
        v = self.__vectorize__(sentence)
        dist, ind = self.tree.query(v,k=topk)
        return (dist[0],ind[0])

class Corpus():
    def __init__(self,corpus_file):
        super().__init__()

        qa_corpus = pd.read_csv(corpus_file)
        tem_corpus = []
        answers = []
        for idx in range(len(qa_corpus)):
            q = qa_corpus['question'][idx]
            if isinstance(q,str):
                question = cs.segment(q,'arr')
                tem_corpus.append(question)
            a = qa_corpus['answer'][idx]
            if isinstance(a,str):
                answer = cs.segment(a,'arr')
                tem_corpus.append(answer)
                answers.append(answer)
        # 回答语料
        answer_corpus = []
        self.vectorize_ans = []
        for sentence in answers:
            if len(sentence)>0:
                vec = sum([WordPool.get_vector(word) for word in sentence])/len(sentence)
                self.vectorize_ans.append(vec)
                answer_corpus.append(sentence)
        self.answer_corpus = pd.Series(answer_corpus)
        # bs = BoolSearch(answer_corpus)
        # bs.save('search_answer.jsn')
        # print('saved')
        # intention_classify(self.vectorize_ans)

        # 所有语料
        valid_corpus=[]
        self.vectorize_corpus = []
        for idx in range(len(tem_corpus)):
            sentence = tem_corpus[idx]
            if len(sentence)>0:
                vec = sum([WordPool.get_vector(word) for word in sentence])/len(sentence)
                self.vectorize_corpus.append(vec)
                valid_corpus.append(sentence)
        self.valid_corpus = pd.Series(valid_corpus)
        # 分类回答语料
        self.classified_data = []
        for label in range(intention.get_classes()):
            data = SementicSpace(np.array(self.vectorize_ans)[intention.get_indexes(label)])
            self.classified_data.append(data)
        

    def __array2sentece__(self,arr):
        sentences = []
        for sentence in arr:
            line = ''
            for word in sentence:
                line += word + ' '
            sentences.append(line)
        return sentences

    def __find_similarity_by_tfidf__(self,input,source,tops=10):
        sentences = self.__array2sentece__(source)
        new_input = self.__array2sentece__(input)
        sentences.append(new_input[0])
        tfidf_vec = TfidfVectorizer()
        tfidf_matrix = tfidf_vec.fit_transform(sentences).todense()
        values = cosine_similarity(tfidf_matrix)
        last_indx = len(source)
        result = sorted(list(enumerate(values[last_indx])),key=lambda x:x[1],reverse=True)
        return result[1:(1+tops)]

    def __binary_search__(self,doc):
        indexes = Search.get_indexes(doc)
        return self.__find_similarity_by_tfidf__([doc],self.answer_corpus[indexes])

    def get_similarity(self,input,tops=10):
        intent = intention.get_intent(input)
        top10 = self.classified_data[intent].similarity(input)
        sentences = self.answer_corpus[intention.get_indexes(intent)]
        top10_sentences = sentences.iloc[top10[1]]

        doc = cs.segment(input,'arr')
        # print(doc)
        top10_2 = self.__binary_search__(doc)
        # print(top10_2)
        top20 = top10_sentences.tolist() + [self.answer_corpus[item[0]] for item in top10_2]
        # print(top20)
        top = self.__find_similarity_by_tfidf__([doc],top20,tops=1)
        # print(top)
        #
        sentence = top20[top[0][0]]
        # print(sentence)
        return sentence,top[0][1]

    def get_sentence(self,index):
        return self.answer_corpus[index]

# 预加载语料对象
corpus = Corpus('qa_corpus.csv')