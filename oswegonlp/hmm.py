from oswegonlp.preprocessing import conll_seq_generator
from oswegonlp.constants import START_TAG, END_TAG, OFFSET, UNK
from oswegonlp import naive_bayes, most_common 
import numpy as np
from collections import defaultdict
import torch
import torch.nn
from torch.autograd import Variable


def compute_transition_weights(trans_counts, smoothing):
    """
    Compute the HMM transition weights, given the counts.
    Don't forget to assign smoothed probabilities to transitions which
    do not appear in the counts.
    
    This will also affect your computation of the denominator.

    :param trans_counts: counts, generated from most_common.get_tag_trans_counts
    :param smoothing: additive smoothing
    :returns: dict of features [(curr_tag,prev_tag)] and weights

    """
    weights = defaultdict(float)
    
    

    all_tags = list(trans_counts.keys()) + [END_TAG]
    V = len(all_tags) -1

    for tag1 in trans_counts:
        
        counter = trans_counts[tag1]
        total_count = sum(counter.values())    
        for tag in all_tags:
            count = counter[tag]
            weights[(tag,tag1)] = np.log((smoothing + count)/(smoothing*V + total_count))



    #dictionary[word] = np.log((alpha + lcount)/(alpha*V + totalTokens))

    return weights


def compute_weights_variables(nb_weights, hmm_trans_weights, vocab, word_to_ix, tag_to_ix):
    """
    Computes autograd Variables of two weights: emission_probabilities and the tag_transition_probabilties
    parameters:
    nb_weights: -- a dictionary of emission weights
    hmm_trans_weights: -- dictionary of tag transition weights
    vocab: -- list of all the words
    word_to_ix: -- a dictionary that maps each word in the vocab to a unique index
    tag_to_ix: -- a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    
    :returns:
    emission_probs_vr: torch Variable of a matrix of size Vocab x Tagset_size
    tag_transition_probs_vr: torch Variable of a matrix of size Tagset_size x Tagset_size
    :rtype: autograd Variables of the the weights
    """
    # Assume that tag_to_ix includes both START_TAG and END_TAG

    tag_transition_probs = np.full((len(tag_to_ix), len(tag_to_ix)), -np.inf)
    emission_probs = np.full((len(vocab), len(tag_to_ix)), 0.0)

    '''i=0
    used = []
    for word in vocab:
        tensor = emission_probs[i]
        i+=1
        index=0
        tensor[0] = -np.inf
        for (tag,w) in nb_weights:
            if(tag,word) in nb_weights: 
                if (tag,word) not in used:
                    used.append((tag,word))
                    weight = nb_weights[(tag,word)]
                    index+=1
            else:
                weight = -np.inf
            tensor[index] = weight
        tensor[len(tensor)-1] = -np.inf
  
    #dont mess with below
    i=0
    for tag in tag_to_ix:
        tensor = tag_transition_probs[i]
        i+=1
        index=0
        for tag2 in tag_to_ix:
            if (tag,tag2) in hmm_trans_weights:
                weight = hmm_trans_weights[(tag,tag2)]
            else:
                weight = -np.inf
            tensor[index] = weight
            index += 1'''
            
            
    i = 0
    for word in vocab:
        em_tensor = emission_probs[i]
        i+=1
        index=1
        em_tensor[0] = -np.inf
        tran_i=0
        for tag2 in tag_to_ix:
            weight = -np.inf
            if(tag2,word) in nb_weights:
                weight = nb_weights[(tag2,word)]      
                em_tensor[index] = weight
                index+=1
            #transition stuff
            if i == 1:
                tran_tensor = tag_transition_probs[tran_i]
                tran_i+=1
                tran_index=0
                for tag3 in tag_to_ix:
                    if (tag2,tag3) in hmm_trans_weights:
                        weight = hmm_trans_weights[(tag2,tag3)]
                    else:
                        weight = -np.inf
                    tran_tensor[tran_index] = weight
                    tran_index += 1
            
        em_tensor[len(em_tensor)-1] = -np.inf
            
    emission_probs_vr = Variable(torch.from_numpy(emission_probs.astype(np.float32)))
    tag_transition_probs_vr = Variable(torch.from_numpy(tag_transition_probs.astype(np.float32)))
    
    

    return emission_probs_vr, tag_transition_probs_vr
    
