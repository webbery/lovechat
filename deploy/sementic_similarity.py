from sklearn.neighbors import KDTree
import cut_sentence as cs

class SementicSpace():
    def __init__(self,classes_data,words_model):
        '''
        构建KD树
        '''
        points = len(classes_data)
        leaf = points/2-1
        self.tree = KDTree(classes_data,leaf)
        self.words_model = words_model
        
    def __vectorize__(self,sentence):
        x_quest = cs.segment(sentence,'arr')
        return [sum([self.words_model[word] for word in x_quest])/len(x_quest)]
    
    def similarity(self,sentence,topk=10):
        '''
        从KD树中找到最相似的k个数据
        '''
        v = self.__vectorize__(sentence)
        dist, ind = self.tree.query(v,k=topk)
        return (dist,ind)