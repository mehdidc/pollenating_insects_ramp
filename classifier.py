import os
os.environ["THEANO_FLAGS"] = "device=gpu"

from sklearn.pipeline import make_pipeline
from caffezoo.vgg import VGG
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

class Classifier(BaseEstimator):
 
    def __init__(self):
        self.clf = make_pipeline(
            VGG(layer_names=["pool3"]),
            RandomForestClassifier(n_estimators=100, max_depth=25)
        )
        
    def fit(self, X, y):
        self.clf.fit(X, y)
        return self
 
    def predict(self, X):
        return self.clf.predict(X)
        
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
