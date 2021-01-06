from oswegonlp.constants import OFFSET
import numpy as np
import operator
from collections import defaultdict

# use this to find the highest-scoring label
def argmax(scores):
    items = list(scores.items())
    items.sort()
    return items[np.argmax([i[1] for i in items])][0]

def make_feature_vector(x,y):
    """take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param x: counter of base features
    :param y: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    """
    dictionary = {(y, OFFSET): 1}
    i = 1
    for w in x:
        dictionary[(y,w)]=i
        i = i + 1
        
    return dictionary
    

def predict(x,weights,labels):
    """prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    """
    maxCount = -1000 #arbitrary low number
    labelScores = defaultdict(float)
    topLabel = 'NOUN'
    
    
    for label in labels:
        if (label,OFFSET) in weights:
            labelScores[label] = weights[(label,OFFSET)]
        else:
            labelScores[label] = 0
    
    
    
    for label in labels:
            weight = 0
            weight = weights[(label,x)]
            labelScores[label] += weight
            
    
    for label in labels:
        count = labelScores[label]
        if count > maxCount:
            maxCount = count
            topLabel = label
            
    
    return topLabel, labelScores