from sklearn.neighbors import KDTree
from deploy.chat.similarity.sentence import sentence2vec

class SementicSpace():
    def __init__(self,classes_data):
        '''
        构建KD树
        '''
        classes_data = classes_data.tolist()
        points = len(classes_data)
        leaf = points/2-1
        # print(type(classes_data),classes_data)
        self.tree = KDTree(classes_data,leaf)
        
    def __vectorize__(self,sentence):
        return sentence2vec.decompose(sentence)
    
    def find_similarity_indexes(self,sentence,topk=10):
        '''
        从KD树中找到最相似的k个数据
        '''
        v = self.__vectorize__(sentence)
        dist, ind = self.tree.query(v,k=topk)
        return (1/(1+abs(dist[0])),ind[0])
