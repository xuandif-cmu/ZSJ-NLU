import re
import random
import numpy as np
import torch

def removeBIO(query):
    """
    remove B-, I-, and O- prefix for BIO format labels
    # input: query([str]), a list of labels to be remove BIO prefix
    # output: ([str]) BIO a list of prefixed removed labels
    """
    return [re.sub("(B|I|O)-","",w) for w in query]

def getBIO(query):
    # query [str], a list of BIO format label, return a list of BIO
    return [s[0] for s in query]

def addSpaceCap(query):
    # query [str], a list of str need to split by capital letter
    arr=re.findall('[A-Z][^A-Z]*', query)
    arr=" ".join(arr)
    return arr

def sdReplace(patternA, patternB, query):
    return [re.sub(patternA, patternB, w) for w in query]

def sdTokenizerW2V(query,w2v):
    """
    input a text query and output its token index
    :param query: [str] list of text tokens
    :param w2v: a gensim word2vec object
    :return: [int] tokenized list
    """
    tokenizedQuery=[]
    for w in query:
        if w in w2v.vocab:
            tokenizedQuery.append(w2v.vocab[w].index)
        else:
            tokenizedQuery.append(0)

    return tokenizedQuery

def findInds(query,orderedLabel):
    return [orderedLabel.index(label) for label in query]

def getHot(listOfLabel, orderedLabel, SEQ=False):

    sampleLen = len(listOfLabel)
    labelLen = len(orderedLabel)

    if SEQ:
        lens = [len(l) for l in listOfLabel]
        maxLen = max(lens)
        outputHot = torch.zeros([sampleLen,maxLen,labelLen],dtype=torch.int64)
        for i in range(sampleLen):
            for j in range(lens[i]):
                outputHot[i,j,orderedLabel.index(listOfLabel[i][j])] = 1
    else:
        outputHot = torch.zeros([sampleLen,labelLen], dtype=torch.int64)
        for i in range(sampleLen):
            outputHot[i,orderedLabel.index(listOfLabel[i])] = 1

    return outputHot

def id2ymat(index,width):
    outp = torch.zeros([len(index),width], dtype=torch.float)
    for i in range(len(index)):
        outp[i,index[i]]=1
    return outp

def padding(listOfList):
    lens=[len(l) for l in listOfList]
    maxLen=max(lens)
    paddedList=[]
    for l in listOfList:
        n=maxLen-len(l)
        paddedList.append(l+n*[0])
    return torch.tensor(paddedList),torch.tensor(lens)

# https://github.com/nhhoang96/ZeroShotCapsule-PyTorch-
def normMatrix(matrix):
    """Nomralize matrix by column
            input: numpy array, dtype = float32
            output: normalized numpy array, dtype = float32
    """
    # check dtype of the input matrix
    np.testing.assert_equal(type(matrix).__name__, 'ndarray')
    np.testing.assert_equal(matrix.dtype, np.float32)
    # axis = 0  across rows (return size is  column length)
    row_sums = matrix.sum(axis = 1) # across columns (return size = row length)

    # Replace zero denominator
    row_sums[row_sums == 0] = 1
    #start:stop:step (:: === :)
    #[:,np.newaxis]: expand dimensions of resulting selection by one unit-length dimension
    # Added dimension is position of the newaxis object in the selection tuple
    norm_matrix = matrix / row_sums[:, np.newaxis]

    return norm_matrix

def generateBatch(n, batch_size):
    batch_index = random.Random().sample(range(n), batch_size)
    return batch_index

def sortBatch(batch_x, batch_len,
              batch_y=None, batch_seq_y=None, batch_BIO_y=None):

    batch_len_new = batch_len
    batch_len_new, perm_idx = batch_len_new.sort(0, descending=True)
    batch_x_new = batch_x[perm_idx]

    if batch_y is not None:
        batch_y = batch_y[perm_idx]
    if batch_seq_y is not None:
        batch_seq_y = batch_seq_y[perm_idx,:,:]
    if batch_BIO_y is not None:
        batch_BIO_y = batch_BIO_y[perm_idx,:,:]

    return batch_x_new, batch_len_new, batch_y, batch_seq_y, batch_BIO_y

def replaceValue(tensor,valueFrom,valueTo):
    tensor[tensor == valueFrom] = valueTo

def seqUnsqueeze(inp,lens):
    """
    extract the parts beside padding tokens, unsqueeze from 3 dim to 2 dim
    :param inp: [BSZ, T, 2u]
    :param lens: [BSZ]
    :return: [sum(lens), 2u]
    """
    noBatch = len(lens)
    outp = inp[0,range(lens[0]),:]
    for i in range(1,noBatch):
        outp = torch.cat((outp,inp[i,range(lens[i]),:]),dim=0)
    return outp

