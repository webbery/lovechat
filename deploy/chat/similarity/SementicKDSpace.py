from sklearn.neighbors import KDTree
from deploy.chat.similarity.sentence import sentence2vec
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from gensim.models import FastText
import jieba

class SementicSpace():
    def __init__(self,classes_data,all_classes=None):
        '''
        构建KD树
        '''
        num_class = len(set(all_classes.tolist()))
        self.trees = []
        for idx in range(num_class):
            data = classes_data[all_classes==idx]
            points = len(classes_data)
            leaf = points/2-1
            self.trees.append(KDTree(data.tolist(),leaf))
        # print(type(classes_data),classes_data)
        self.kmean = joblib.load('km.pkl')
        self.fasttext  = FastText.load('fasttext.model')

        
    def __vectorize__(self,sentence):
        return sentence2vec.decompose(sentence)
    
    def find_similarity_indexes(self,sentence,topk=10):
        '''
        从KD树中找到最相似的k个数据
        '''
        v = self.__vectorize__(sentence)

        sentence = jieba.lcut(sentence)
        vec = sum([self.fasttext.wv[word] for word in sentence])/len(sentence)
        clazz = self.kmean.predict([vec.tolist()])[0]
        dist, ind = self.trees[clazz].query(v,k=topk)
        return (1/(1+abs(dist[0])),ind[0])
