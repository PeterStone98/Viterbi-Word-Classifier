from oswegonlp.constants import OFFSET
from oswegonlp import classifier_base, evaluation, preprocessing

import numpy as np
import itertools
import functools
from collections import defaultdict
from collections import Counter

def get_nb_weights(trainfile, smoothing):
    """
    estimate_nb function assumes that the labels are one for each document, where as in POS tagging: we have labels for 
    each particular token. So, in order to calculate the emission score weights: P(w|y) for a particular word and a 
    token, we slightly modify the input such that we consider each token and its tag to be a document and a label. 
    The following helper code converts the dataset to token level bag-of-words feature vector and labels. 
    The weights obtained from here will be used later as emission scores for the viterbi tagger.
    
    inputs: train_file: input file to obtain the nb_weights from
    smoothing: value of smoothing for the naive_bayes weights
    
    :returns: nb_weights: naive bayes weights
    """
    token_level_docs=[]
    token_level_tags=[]
    for words,tags in preprocessing.conll_seq_generator(trainfile):
        token_level_docs += [{word:1} for word in words]
        token_level_tags +=tags
    nb_weights = estimate_nb(token_level_docs, token_level_tags, smoothing)
    
    return nb_weights


def corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    counts = Counter()
    for x_i,y_i in zip(x,y):
        if y_i == label:
            counts += x_i
    return counts

    


def estimate_pxy(x,y,label,alpha,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param alpha: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    '''
    probs = defaultdict(float)
    for word in vocab:
        probs[word] = 0
    #probs = {word:0 for word in vocab}
    counts = corpus_counts(x,y,label)
    total_count = functools.reduce(lambda a,b : a+b,counts.values())

    for v in vocab:
        num = alpha + counts[v]
        denom = len(vocab) * alpha + total_count
        probs[v] = np.log(num/denom)

    return probs

    


def estimate_nb(x,y,alpha):
    """
    estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param alpha: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    labels = set(y)
    labelCounts = Counter(y)
    documentCount = len(y)
    vocab = set(itertools.chain.from_iterable([x_i.keys() for x_i in x]))
    weights = defaultdict(float)

    for l in labels:
        pxy = estimate_pxy(x,y,l,alpha,vocab)
        for word, prob in pxy.items():
            weights[(l,word)] = prob
        weights[(l,OFFSET)] = np.log(labelCounts[l]/documentCount)

    return weights

    


def find_best_smoother(x_tr,y_tr,x_dv,y_dv,alphas):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param alphas: list of smoothing values
    :returns: best smoothing value
    :rtype: float

    '''
    best_smoother = alphas[0]
    best_score = 0
    scores = {}

    for alpha in alphas:
        theta_nb = estimate_nb(x_tr, y_tr, alpha)
        results_nb = classifier_base.predict_all(x_dv, theta_nb, set(y_dv))
        score = evaluation.acc(results_nb, y_dv)
        if score > best_score:
            best_score = score
            best_smoother = alpha
        scores[alpha] = score

    return best_smoother, scores
    







