from sklearn.cluster import KMeans
from sklearn.externals import joblib
from deploy.word_vector import model as WordPool
import deploy.cut_sentence as cs
import deploy.sentence_vector as sv

def intention_classify(X,cluster=5):
    kmean = KMeans(n_clusters=cluster)
    kmean.fit(X)
    joblib.dump(kmean , 'km2.pkl')
    print('save classify')

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
        return self.classifier.predict([vec])[0]

    def get_classes(self):
        return len(self.classifier.cluster_centers_)

    def get_indexes(self,label):
        return self.classifier.labels_==label

intention = Intention('km2.pkl')