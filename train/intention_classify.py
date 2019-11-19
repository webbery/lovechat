from sklearn.cluster import KMeans
from sklearn.externals import joblib

def intention_classify(X,cluster=5):
    kmean = KMeans(n_clusters=cluster)
    kmean.fit(X)
    joblib.dump(kmean , 'km.pkl')