from sklearn.cluster import KMeans
from sklearn.externals import joblib
from deploy.word_vector import model as WordPool
import deploy.cut_sentence as cs
import deploy.sentence_vector as sv
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from collections import defaultdict
import numpy as np

def intention_classify(X,cluster=5):
    kmean = KMeans(n_clusters=cluster)
    kmean.fit(X)
    joblib.dump(kmean , 'km2.pkl')
    print('save classify')

def predict_label(x,dbscan):
    points = len(dbscan.components_)
    leaf = points/2-1
    tree = KDTree(dbscan.components_,leaf)
    labels = dbscan.labels_
    cnt = 2*len(set(labels))
    dist, indx = tree.query([x],k=cnt)
    dict_of_label=defaultdict(int)
    for idx in indx[0]:
        dict_of_label[labels[idx]]+=1
    result = sorted(dict_of_label.items(), key=lambda s: s[1],reverse=True) 
    return result[0]

class Intention():
    def __init__(self,classifier_file):
        self.classifier = joblib.load(classifier_file)

    def get_intent(self,text):
        words = cs.segment(text,'arr')
        # print(words)
        vectorize = WordPool.get_vector(words)
        # print(vectorize)
        vec = sv.make_sentence_vector(vectorize)
        # print(vec)
        # return self.classifier.predict([vec])[0]
        return predict_label(vec,self.classifier)[0]

    def get_classes(self):
        return len(set(self.classifier.labels_))

    def get_indexes(self,label):
        return self.classifier.labels_==(label-1)

intention = Intention('dbscan.pkl')