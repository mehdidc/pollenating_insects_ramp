import os
os.environ["THEANO_FLAGS"] = "device=gpu"

from sklearn.pipeline import make_pipeline
from caffezoo.vgg import VGG
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier


"""
More details on VGG : http://arxiv.org/pdf/1409.1556.pdf

layer_names can accept a list of strings, each string
is a layer name. The complete list of layer names
can be seen in this graph :

    https://drive.google.com/open?id=0B1CFqLHwhNoaODJOOEV5M1ZNbWc

Each node is either a convolution, a pooling layer or a nonlinearity layer.
The nodes representing convolution and pooling layers start by
the layer name (which you can put in layer_names). For convolutional layer
if you just use the name of the layer, like "conv_1" it will only take
the activations after applying the convolution. if you want to obtain
the activations after applying the activation function (ReLU), use
layername/relu, for instance conv_1/relu.

You can also provide an aggregation function, which takes
a set of layers features and returns a numpy array. The default
aggregation function used concatenate all the layers.

VGG(aggregate_function=your_function, layer_names=[...])

the default aggregation function looks like this:
    def concat(layers):
        l = np.concatenate(layers, axis=1)
        return l.reshape((l.shape[0], -1))

"""


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
