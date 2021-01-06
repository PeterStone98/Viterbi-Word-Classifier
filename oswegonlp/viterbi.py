import operator
from collections import defaultdict, Counter
from oswegonlp.constants import START_TAG,END_TAG, UNK
import numpy as np
import torch
import torch.nn
from torch import autograd
from torch.autograd import Variable

def get_torch_variable(arr):
    # returns a pytorch variable of the array
    torch_var = torch.autograd.Variable(torch.from_numpy(np.array(arr).astype(np.float32)))
    return torch_var.view(1,-1)

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def viterbi_step(all_tags, tag_to_ix, cur_tag_scores, transition_scores, prev_scores):
    """
    Calculates the best path score and corresponding back pointer for each tag for a word in the sentence in pytorch, which you will call from the main viterbi routine.
    
    parameters:
    - all_tags: list of all tags: includes both the START_TAG and END_TAG
    - tag_to_ix: a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    - cur_tag_scores: pytorch Variable that contains the local emission score for each tag for the current token in the sentence
                       it's size is : [ len(all_tags) ] 
    - transition_scores: pytorch Variable that contains the tag_transition_scores
                        it's size is : [ len(all_tags) x len(all_tags) ] 
    - prev_scores: pytorch Variable that contains the scores for each tag for the previous token in the sentence: 
                    it's size is : [ 1 x len(all_tags) ] 
    
    :returns:
    - viterbivars: a list of pytorch Variables such that each element contains the score for each tag in all_tags for the current token in the sentence
    - bptrs: a list of idx that contains the best_previous_tag for each tag in all_tags for the current token in the sentence
    """
    bptrs = []
    viterbivars = []
    
    for next_tag in list(all_tags):
        bestScore = -np.inf
        emission = cur_tag_scores[tag_to_ix[next_tag]] 
        index = 0
        for prevScore in prev_scores[0]:
            transition = transition_scores[tag_to_ix[next_tag]][index]
            curScore = emission + transition + prev_scores[0][index]
            if curScore > bestScore and emission != -np.inf:
                bestScore = curScore
                backpointer = index
            elif emission == -np.inf:
                backpointer = 0
            index += 1
     
        viterbivars.append(bestScore)
        bptrs.append(backpointer)
        
     
            
 
    return viterbivars, bptrs

def build_trellis(all_tags, tag_to_ix, cur_tag_scores, transition_scores):
    """
    This function should compute the best_path and the path_score. 
    Use viterbi_step to implement build_trellis in viterbi.py in Pytorch.
    
    parameters:
    - all_tags: a list of all tags: includes START_TAG and END_TAG
    - tag_to_ix: a dictionary that maps each tag to a unique id.
    - cur_tag_scores: a list of pytorch Variables where each contains the local emission score for each tag for that particular token in the sentence, len(cur_tag_scores) will be equal to len(words)
                        it's size is : [ len(words in sequence) x len(all_tags) ] 
    - transition_scores: pytorch Variable (a matrix) that contains the tag_transition_scores
                        it's size is : [ len(all_tags) x len(all_tags) ] 
    
    :returns:
    - path_score: the score for the best_path
    - best_path: the actual best_path, which is the list of tags for each token: exclude the START_TAG and END_TAG here.
    """
    
    ix_to_tag = {v: k for k, v in tag_to_ix.items()}

    # setting all the initial score to START_TAG
    # make sure END_TAG is in all_tags
    initial_vec = np.full((1, len(all_tags)), -np.inf)
    initial_vec[0][tag_to_ix[START_TAG]] = 0
    prev_scores = torch.autograd.Variable(torch.from_numpy(initial_vec.astype(np.float32))).view(1, -1)
    whole_bptrs = []
    best_path=[]
    for m in range(len(cur_tag_scores)): 
        scores,bptrs = viterbi_step(all_tags, tag_to_ix, cur_tag_scores[m], transition_scores, prev_scores)
        prev_scores = torch.tensor([scores])
        whole_bptrs.append(bptrs)

    # after you finish calculating the tags for all the words: don't forget to calculate the scores for the END_TAG
    final_vec=np.full((1,len(all_tags)),-np.inf)
    final_vec[0][tag_to_ix[END_TAG]] =0
    final_vec_var = torch.autograd.Variable(torch.from_numpy(final_vec.astype(np.float32))).view(1,-1)
    
   # scores,bptrs = viterbi_step(all_tags, tag_to_ix, final_vec_var[0], transition_scores, prev_scores)
      
   # prev_scores = torch.tensor([scores])
    #whole_bptrs.append(bptrs)
    final_scores,bptrs = viterbi_step(all_tags, tag_to_ix,final_vec[0],transition_scores, prev_scores)
        

    # Calculate the best_score and also the best_path using backpointers and don't forget to reverse the path
    path_score = -np.inf
    bptr = END_TAG
    for i in range(len(final_scores)):
        if final_scores[i] > path_score:
            path_score = final_scores[i]
            bptr = bptrs[i]
            
    best_path = []
    whole_bptrs.reverse()
    for bptrs in whole_bptrs:
        best_path.append(ix_to_tag[bptr])
        bptr = bptrs[bptr]
        
    best_path.reverse()
    return path_score,best_path


    
