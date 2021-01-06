import torch
import torch.nn as nn
import numpy as np
import pickle

class CBOW(torch.nn.Module):
    #6.2
    def __init__(self, vocab_size, embedding_dim):
        """
        Initializes the components of the CBOW model.
        
        :param vocab_size: Size of the vocab.
        :param embedding_dim: size of the embeddings.
        """
        super(CBOW, self).__init__()

        #self.embeddings = 

        #self.linear1 = 
        #self.activation_function1 = 

        #self.linear2 = 
        #self.activation_function2 = 
        
        raise NotImplementedError
    #6.3
    def forward(self, inputs):
        """
        The function obtain the probabilities for each word given context.
        
        :param inputs: a context tensor (first part of the tuple from make_context_tensors)
        :returns: probabilities for each token in the vocabulary (output of log softmax)
        """
        raise NotImplementedError

    def get_word_embedding(self, word, word_to_ix):
        word = torch.LongTensor([word_to_ix[word]])
        return self.embeddings(word).view(1, -1)

#6.1
def build_context(documents, context_size=2):
    """
    build context vectors of context_size for each document in documents.

    :param documents: list of tokenized documents (i.e., a list of lists of tokens)
    :param context_size: number of context tokens on each side of a token.
    :returns: context_vectors
    :rtype: list of (list, string) pairs, where the list contains the context_size 
            tokens before and after the string (token).
    """
    
    context_vectors = []
    
    raise NotImplementedError
    
def make_context_tensors(context_vectors, word_to_ix, device="cpu"):
    """
    convert context vectors into tensors of word indices. 
    
    :param context_vectors: output from build_context. 
    :param word_to_ix: word to index mappings
    :param device: device to compute on. 
    :returns tensor version of context_vectors using word indices instead of text.
    :rtype list of (list, tensor) tuples.
    """
    context_tensors = []
    for context, target in context_vectors:
        idxs = [word_to_ix[w] for w in context]
        context_tensors.append((torch.tensor(idxs, dtype=torch.long).to(device), 
                                torch.tensor([word_to_ix[target]], dtype=torch.long).to(device)))
    return context_tensors

def train_model(model, data, word_to_ix, iters=50, lr=0.001):
    '''
    data: list of (context, target) pairs.
    iters: number of training iterations
    lr: learning rate
    '''
    print("Data size", len(data))
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(iters):
        total_loss = 0
        counter = 0
        for context_tensor, target_tensor in data:
            model.zero_grad()
            log_probs = model(context_tensor)
            loss = loss_function(log_probs, target_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.data
            counter += 1
        if epoch % 2 == 0:
            print("Epoch: ", epoch, " Loss: ", total_loss)

def write_polyglot_format(model, word_to_ix, outfile):

    vocab = [k for [k,v] in word_to_ix.items()]
    
    word_vecs=[] # Index corresponds to vocab.
    for word in vocab:
        embeds = model.get_word_embedding(word, word_to_ix)[0].detach().tolist()
        word_vecs.append(embeds)
        
    vecs = [vocab, word_vecs]
    
    pickle.dump(vecs, open(outfile, 'wb'))
    
        
    
    


