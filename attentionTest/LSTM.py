import torch
import torch.nn as nn
from Utils import Utils
from Object import Object
from collections import OrderedDict

class LSTMAdvancedAttentionContextOnly(nn.Module, Object):
    '''
    Implementation of LSTM model, in a word-by-word way. Such implementation allows more
    flexible operations.
    Attention mechanism is introduced. Please refer to paper 'Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling'.
    The difference between LSTMAdvancedAttentionContextOnly and LSTMAdvancedAttention is that the former input
    is only context, but the latter's input is concatenated context and word.
    This model:
            1. is not bidirectional
            2. has only 1 layer
    This model is used to encode sequence. It output:
            1. Encoded result for each sequence element, or each word.
            2. Last hidden state of all layers
            3. attention weight
    '''
    def __init__(self, embDim, hiddenDim, device, name="LSTMAdvancedAttention"):
        '''
        Init function.
        :param embDim:      Embedding dimension of input data. Integer.
        :param hiddenDim:   Hidden dimension of output data, including hidden state and output. Integer.
        :param device:      Data device
        '''
        nn.Module.__init__(self)
        Object.__init__(self, name=name)

        # populate model parameters
        self.__embDim = embDim
        self.__hiddenDim = hiddenDim
        self.__layerNum = 1
        self.__device = device
        self.__directionNum = 1
        self.__bidirectional = False

        # build up model
        self.__lstm = nn.LSTM(input_size=self.__embDim,
                              hidden_size=self.__hiddenDim,
                              num_layers=self.__layerNum,
                              bidirectional=self.__bidirectional)
        # fully connected layer: (hiddenState, wordEmbedding) --> a float value
        # Following fully_connected_layer's output describes the correlation between current hiddenState
        # and word in the sentence, given its wordEmbedding. Such value is used to calculate weight used
        # in attention mechanism
        self.__fcEnergy = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(self.__hiddenDim + self.__embDim, self.__hiddenDim + self.__embDim)),
            ('activate2', nn.Tanh()),
            ('linear2', nn.Linear(self.__hiddenDim + self.__embDim, 1)),
            ('activate2', nn.Tanh())
        ])
        )

    def forward(self, input, hiddenStateInit=torch.Tensor()):
        '''
        Forward function.
        :param self: ...
        :param input:
                        It's supposed to be a list of tensor, the result of word embedding.
                        Sequences are allowed to be of different lengths.
                        Type: list of Tensor
                        Shape: BATCH * SEQ * WORD_EMBEDDING_DIM
                        Example:
                            word1 = [1.1, 1.2, 1.3, 1.4]    # word embedding
                            word2 = [2.1, 2.2, 2.3, 2.4]    # word embedding
                            word3 = [3.1, 3.2, 3.3, 3.4]    # word embedding
                            word4 = [4.1, 4.2, 4.3, 4.4]    # word embedding
                            word5 = [5.1, 5.2, 5.3, 5.4]    # word embedding
                            sentence1 = [word1, word2, word3]
                            sentence2 = [word4, word5]
                            sentence3 = [word1]
                            batch = [sentence1, sentence2, sentence3]
                            batchTensorList = [ torch.Tensor(sentence1),
                                                torch.Tensor(sentence2),
                                                torch.Tensor(sentence3)]
        :param hiddenStateInit:
                        It's supposed to be a tensor.
                        Type: Tensor
                        Shape: [LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM]
        :return:
                (output, sqlLengths):
                    Encoded result for each element in each sequence in given batch
                    output:
                            LSTM output. Encoded result for each sequence in a batch.
                            Type:   Tensor
                            Shape:  BATCH * SEQ * DIRECTION_NUM * HIDDEN_DIM
                    sqlLengths:
                            A tensor storing sequence lengths of input batch
                            Type:   tensor
                            Shape:  BATCH
                hiddenState:
                        hidden state of all layers
                        Type: Tensor
                        Shape: [LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM]
                attentionMatrixList:
                        A list of matrix, containing weight distribution of every word over other words
        '''

        # get batch size
        batchSize = len(input)

        # get sequence lengths
        seqLengthList = []
        for sentence in input:
            seqLengthList.append(sentence.shape[0])

        # pad input
        # list[sentenceTensor]   -->  Tensor[BATCH, SENTENCE, EMBEDDING_DIM]
        paddedInput = torch.nn.utils.rnn.pad_sequence(input, batch_first=True)

        # Prepare first hidden and cell state, for each sequence in batch
        # prepare first hidden state
        if hiddenStateInit.nelement() == 0:    # init hidden state into 0, if it's not provided
            hiddenStateInit = torch.zeros(self.__layerNum*self.__directionNum, batchSize, self.__hiddenDim)
        # prepare first cell state, it's all 0
        cellStateInit = torch.zeros(self.__layerNum*self.__directionNum, batchSize, self.__hiddenDim)
        # move hidden state and cell state to device
        hiddenStateInit = Utils.move2Device(hiddenStateInit, self.__device)
        cellStateInit   = Utils.move2Device(cellStateInit, self.__device)

        # hiddenOutput initialize. It will be updated later
        hiddenOutput = torch.zeros(self.__layerNum*self.__directionNum, batchSize, self.__hiddenDim)  # [LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM]

        # iterate all sequences, in time, or, word dimension: for all batch data, first word, second, word, third word ...
        paddedInput = paddedInput.transpose(0, 1)  # [BATCH, SENTENCE, EMBEDDING]  -> [SENTENCE, BATCH, EMBEDDING]
        # initialize first hidden and cell state
        hidden = hiddenStateInit
        cell = cellStateInit
        # lstmOutputList: a list to store lstm output
        lstmOutputList = []    # a list of element tensor[BATCH, EMBEDDING]
        attentionList = []     # a list of weight: wordPos -> sentence  -> weightDistribution of word at wordPos of sentence over all other words in that sentence
        for wordPos, wordBatch in enumerate(paddedInput):   # wordBatch: [BATCH, EMBEDDING]
            # calculate context: for each sentence, it's determined by current hidden and all words in this sentence
            # contextBatch = tensor[BATCH * EMBEDDING]
            # attentionListBatch = list[BATCH], element is tensor[SENTENCE], means weight distribution of word at wordPos over all words in the sentence
            contextBatch, attentionListBatch = self.__calculateContextBatch(paddedInput, seqLengthList, hidden, wordPos)

            contextBatch = contextBatch.unsqueeze(0)
            # lstmOutput:  (seq_len, batch, num_directions * hidden_size)
            # hidden:      (num_layers * num_directions, batch, hidden_size)
            # cell:        (num_layers * num_directions, batch, hidden_size)
            lstmOutput, (hidden, cell) = self.__lstm(contextBatch, (hidden, cell))

            # update hidden state to be output, since some sentence is ended
            hiddenOutput = self.__updateHiddenOutput(hiddenOutput, hidden, wordPos, seqLengthList)

            # store lstm output
            lstmOutputList.append(lstmOutput[0])   # lstmOutput[0]: [BATCH, EMBEDDING]

            # store attention, for review later
            attentionList.append(attentionListBatch)

        # mask lstm output, since some output is just padding
        lstmOutputTensor = self.__maskLSTMOutput(lstmOutputList, seqLengthList)

        # mask attention output, since some word is just padding
        attentionMatrixList = self.__maskLSTMOutputAttention(attentionList, seqLengthList)

        # return
        seqLengthTensor = torch.tensor(seqLengthList, dtype=torch.long)
        return (lstmOutputTensor, seqLengthTensor), hiddenOutput, attentionMatrixList

    def __maskLSTMOutputAttention(self, attentionList, seqLengthList):
        '''
        Mask LSTM attention output, since some words are padding
        @param attentionList: a list of weight: wordPos -> sentence  -> weightDistribution of word at wordPos of
                              sentence over all other words in that sentence
        @param seqLengthList: a list of sequence lengths
        @return: list[BATCH], element is a matrix M(N * N). M(i,j) means when predicting label of word i, what is the
                 weight on word j.
        '''
        attentionListOutput = []

        batchSize = len(seqLengthList)
        for sentencePos in range(batchSize):   # for each sentence in the batch
            matrix = []
            for wordPos in range(seqLengthList[sentencePos]):    # for each word in the sentence
                matrix.append(attentionList[wordPos][sentencePos])
            attentionListOutput.append(matrix)

        return attentionListOutput

    def __calculateContextBatch(self, paddedInput, seqLengthList, hidden, currentWordPos):
        '''
        Calculate context for a batch
        @param paddedInput: tensor[SENTENCE, BATCH, EMBEDDING]
        @param seqLengthList: list[BATCH]
        @param hidden: tensor[LAYERS*DIRECTION_NUM, BATCH, HIDDEN_DIM], LAYERS = 1, DIRECTION_NUM = 1
        @param currentWordPos: current word position, starting from 0. It provides information to avoid unnecessary
                               calculation.
        @return:
            tensor[BATCH * EMBEDDING]
        '''
        # learn input shape
        batchSize = paddedInput.shape[1]

        # sanity check
        if not batchSize == len(seqLengthList):
            print("[ERROR] LSTMAdvancedAttention. Fail to calculate context batch due to inconsistency between input data size and sequence length list size.")

        # Compose energy neural network input:
        # for each sentence in given batch, concatenate current hidden state with all words, respectively
        hiddenStateWordEmbeddingList = []   # hiddenStateWordEmbeddingList stores concatenated hidden state and word embeddings
        caredSentenceList = []    # a list[M], storing index of sentences we cared: such sentence is within boundary of current word position
        for sentencePos in range(batchSize):
            # if current sentence is longer than processing word position, ignore it
            if currentWordPos >= seqLengthList[sentencePos]:
                continue
            else:
                caredSentenceList.append(sentencePos)
                for wordPos in range(seqLengthList[sentencePos]):
                    # hiddenStateWordEmbedding is concatenated tensor
                    hiddenStateWordEmbedding = torch.cat([paddedInput[wordPos][sentencePos], hidden[0][sentencePos]], dim = 0)
                    hiddenStateWordEmbeddingList.append(hiddenStateWordEmbedding)

        # transform a list of tensor into a tensor
        # energyNNInput: tensor[M * (EMBEDDING_DIM + HIDDEN_DIM)], where M is the number of all words in current batch
        energyNNInput = torch.stack(hiddenStateWordEmbeddingList)

        # calculate energy
        energySerialized = self.__fcEnergy(energyNNInput)

        # [M, 1] --> [M]
        energySerialized = energySerialized.squeeze()

        # calculate length of sentence that is cared
        caredSentenceLengthList = []
        for sentenceID in caredSentenceList:
            caredSentenceLengthList.append(seqLengthList[sentenceID])

        # calculate context
        contextList = []
        weightList  = []
        for sentencePos in range(batchSize):
            # if it's cared sentence
            if sentencePos in caredSentenceList:
                # calculate sentence start and end position, in energySerialized
                positionInCaredSentenceList = caredSentenceList.index(sentencePos)
                wordPosStart = sum(caredSentenceLengthList[0:positionInCaredSentenceList])
                wordPosEnd   = wordPosStart + seqLengthList[sentencePos]

                # calculate weight by applying softmax on energy
                weightTensor = torch.nn.functional.softmax(energySerialized[wordPosStart:wordPosEnd], dim = 0)

                # context vector is weighted average of all words in a sentence
                context = torch.matmul(weightTensor, paddedInput[0:seqLengthList[sentencePos], sentencePos])

                # store current context
                contextList.append(context)

                # store weight, for review later
                weightList.append(weightTensor.tolist())

            else:  # if it's not cared sentence
                # fake context
                contextList.append(torch.zeros(self.__embDim, device=self.__device))

                # fake weight
                weightList.append(torch.zeros(seqLengthList[sentencePos], device=self.__device))

        # list -> tensor
        contextTensor = torch.stack(contextList)

        return contextTensor, weightList

    def __updateHiddenOutput(self, hiddenOutput, hidden, wordPos, seqLengthList):
        '''
        Update hidden state to be output, according to current hidden state for word position wordPos, and sequence lengths
        @param hiddenOutput:  tensor[LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM]
        @param hidden:        tensor[LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM]
        @param wordPos:       word index, starting from 0
        @param seqLengthList: a list of sequence lengths
        @return: updated hidden states output
            @type: tensor[LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM]
        '''
        # for each sequence in batch
        for sentencePos, length in enumerate(seqLengthList):
            if wordPos == (length - 1):   # wordPos is the last word of current sequence
                hiddenOutput[0, sentencePos] = hidden[0, sentencePos]

        # return hidden states
        return hiddenOutput

    def __maskLSTMOutput(self, lstmOutputList, seqLengthList):
        '''
        Mask lstm output, according to sequence lengths.
        @param lstmOutputList: a list of lstm output: a list of element tensor[BATCH, EMBEDDING]
        @param seqLengthList: a list of sequence lengths
        @return: masked lstm output
            Shape:  Tensor[BATCH * SEQ * DIRECTION_NUM * HIDDEN_DIM], DIRECTION_NUM = 1
        '''

        # apply mask
        lstmOutputTensor = torch.stack(lstmOutputList)   # SEQ * BATCH * HIDDEN_DIM
        for wordPos in range(lstmOutputTensor.shape[0]):
            for sentencePos in range(lstmOutputTensor.shape[1]):
                if wordPos >= seqLengthList[sentencePos]:    # it's out of the range of current sentence
                    lstmOutputTensor[wordPos][sentencePos] = torch.zeros(self.__hiddenDim)

        # return
        lstmOutputTensor = lstmOutputTensor.transpose(0, 1) # SEQ * BATCH * HIDDEN_DIM  --> BATCH * SEQ * HIDDEN_DIM
        lstmOutputTensor = lstmOutputTensor.unsqueeze(2)
        return lstmOutputTensor

class LSTMAdvancedAttention(nn.Module, Object):
    '''
    Implementation of LSTM model, in a word-by-word way. Such implementation allows more
    flexible operations.
    Attention mechanism is introduced. Please refer to paper 'Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling'.
    This model:
            1. is not bidirectional
            2. has only 1 layer
    This model is used to encode sequence. It output:
            1. Encoded result for each sequence element, or each word.
            2. Last hidden state of all layers
            3. attention weight
    '''
    def __init__(self, embDim, hiddenDim, device, name="LSTMAdvancedAttention"):
        '''
        Init function.
        :param embDim:      Embedding dimension of input data. Integer.
        :param hiddenDim:   Hidden dimension of output data, including hidden state and output. Integer.
        :param device:      Data device
        '''
        nn.Module.__init__(self)
        Object.__init__(self, name=name)

        # populate model parameters
        self.__embDim = embDim
        self.__hiddenDim = hiddenDim
        self.__layerNum = 1
        self.__device = device
        self.__directionNum = 1
        self.__bidirectional = False

        # build up model
        self.__lstm = nn.LSTM(input_size=self.__embDim * 2,
                              hidden_size=self.__hiddenDim,
                              num_layers=self.__layerNum,
                              bidirectional=self.__bidirectional)
        # fully connected layer: (hiddenState, wordEmbedding) --> a float value
        # Following fully_connected_layer's output describes the correlation between current hiddenState
        # and word in the sentence, given its wordEmbedding. Such value is used to calculate weight used
        # in attention mechanism
        self.__fcEnergy = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(self.__hiddenDim + self.__embDim, self.__hiddenDim + self.__embDim)),
            ('activate2', nn.Tanh()),
            ('linear2', nn.Linear(self.__hiddenDim + self.__embDim, 1)),
            ('activate2', nn.Tanh())
        ])
        )

    def forward(self, input, hiddenStateInit=torch.Tensor()):
        '''
        Forward function.
        :param self: ...
        :param input:
                        It's supposed to be a list of tensor, the result of word embedding.
                        Sequences are allowed to be of different lengths.
                        Type: list of Tensor
                        Shape: BATCH * SEQ * WORD_EMBEDDING_DIM
                        Example:
                            word1 = [1.1, 1.2, 1.3, 1.4]    # word embedding
                            word2 = [2.1, 2.2, 2.3, 2.4]    # word embedding
                            word3 = [3.1, 3.2, 3.3, 3.4]    # word embedding
                            word4 = [4.1, 4.2, 4.3, 4.4]    # word embedding
                            word5 = [5.1, 5.2, 5.3, 5.4]    # word embedding
                            sentence1 = [word1, word2, word3]
                            sentence2 = [word4, word5]
                            sentence3 = [word1]
                            batch = [sentence1, sentence2, sentence3]
                            batchTensorList = [ torch.Tensor(sentence1),
                                                torch.Tensor(sentence2),
                                                torch.Tensor(sentence3)]
        :param hiddenStateInit:
                        It's supposed to be a tensor.
                        Type: Tensor
                        Shape: [LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM]
        :return:
                (output, sqlLengths):
                    Encoded result for each element in each sequence in given batch
                    output:
                            LSTM output. Encoded result for each sequence in a batch.
                            Type:   Tensor
                            Shape:  BATCH * SEQ * DIRECTION_NUM * HIDDEN_DIM
                    sqlLengths:
                            A tensor storing sequence lengths of input batch
                            Type:   tensor
                            Shape:  BATCH
                hiddenState:
                        hidden state of all layers
                        Type: Tensor
                        Shape: [LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM]
                attentionMatrixList:
                        A list of matrix, containing weight distribution of every word over other words
        '''

        # get batch size
        batchSize = len(input)

        # get sequence lengths
        seqLengthList = []
        for sentence in input:
            seqLengthList.append(sentence.shape[0])

        # pad input
        # list[sentenceTensor]   -->  Tensor[BATCH, SENTENCE, EMBEDDING_DIM]
        paddedInput = torch.nn.utils.rnn.pad_sequence(input, batch_first=True)

        # Prepare first hidden and cell state, for each sequence in batch
        # prepare first hidden state
        if hiddenStateInit.nelement() == 0:    # init hidden state into 0, if it's not provided
            hiddenStateInit = torch.zeros(self.__layerNum*self.__directionNum, batchSize, self.__hiddenDim)
        # prepare first cell state, it's all 0
        cellStateInit = torch.zeros(self.__layerNum*self.__directionNum, batchSize, self.__hiddenDim)
        # move hidden state and cell state to device
        hiddenStateInit = Utils.move2Device(hiddenStateInit, self.__device)
        cellStateInit   = Utils.move2Device(cellStateInit, self.__device)

        # hiddenOutput initialize. It will be updated later
        hiddenOutput = torch.zeros(self.__layerNum*self.__directionNum, batchSize, self.__hiddenDim)  # [LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM]

        # iterate all sequences, in time, or, word dimension: for all batch data, first word, second, word, third word ...
        paddedInput = paddedInput.transpose(0, 1)  # [BATCH, SENTENCE, EMBEDDING]  -> [SENTENCE, BATCH, EMBEDDING]
        # initialize first hidden and cell state
        hidden = hiddenStateInit
        cell = cellStateInit
        # lstmOutputList: a list to store lstm output
        lstmOutputList = []    # a list of element tensor[BATCH, EMBEDDING]
        attentionList = []     # a list of weight: wordPos -> sentence  -> weightDistribution of word at wordPos of sentence over all other words in that sentence
        for wordPos, wordBatch in enumerate(paddedInput):   # wordBatch: [BATCH, EMBEDDING]
            # calculate context: for each sentence, it's determined by current hidden and all words in this sentence
            # contextBatch = tensor[BATCH * EMBEDDING]
            # attentionListBatch = list[BATCH], element is tensor[SENTENCE], means weight distribution of word at wordPos over all words in the sentence
            contextBatch, attentionListBatch = self.__calculateContextBatch(paddedInput, seqLengthList, hidden, wordPos)

            # concatenate: current word embedding + current context, for each sentence in a bach
            wordEmbeddingContext = torch.cat([wordBatch, contextBatch], dim = 1)

            # lstmOutput:  (seq_len, batch, num_directions * hidden_size)
            # hidden:      (num_layers * num_directions, batch, hidden_size)
            # cell:        (num_layers * num_directions, batch, hidden_size)
            wordEmbeddingContext = wordEmbeddingContext.unsqueeze(0)
            lstmOutput, (hidden, cell) = self.__lstm(wordEmbeddingContext, (hidden, cell))

            # update hidden state to be output, since some sentence is ended
            hiddenOutput = self.__updateHiddenOutput(hiddenOutput, hidden, wordPos, seqLengthList)

            # store lstm output
            lstmOutputList.append(lstmOutput[0])   # lstmOutput[0]: [BATCH, EMBEDDING]

            # store attention, for review later
            attentionList.append(attentionListBatch)

        # mask lstm output, since some output is just padding
        lstmOutputTensor = self.__maskLSTMOutput(lstmOutputList, seqLengthList)

        # mask attention output, since some word is just padding
        attentionMatrixList = self.__maskLSTMOutputAttention(attentionList, seqLengthList)

        # return
        seqLengthTensor = torch.tensor(seqLengthList, dtype=torch.long)
        return (lstmOutputTensor, seqLengthTensor), hiddenOutput, attentionMatrixList

    def __maskLSTMOutputAttention(self, attentionList, seqLengthList):
        '''
        Mask LSTM attention output, since some words are padding
        @param attentionList: a list of weight: wordPos -> sentence  -> weightDistribution of word at wordPos of
                              sentence over all other words in that sentence
        @param seqLengthList: a list of sequence lengths
        @return: list[BATCH], element is a matrix M(N * N). M(i,j) means when predicting label of word i, what is the
                 weight on word j.
        '''
        attentionListOutput = []

        batchSize = len(seqLengthList)
        for sentencePos in range(batchSize):   # for each sentence in the batch
            matrix = []
            for wordPos in range(seqLengthList[sentencePos]):    # for each word in the sentence
                matrix.append(attentionList[wordPos][sentencePos])
            attentionListOutput.append(matrix)

        return attentionListOutput

    def __calculateContextBatch(self, paddedInput, seqLengthList, hidden, currentWordPos):
        '''
        Calculate context for a batch
        @param paddedInput: tensor[SENTENCE, BATCH, EMBEDDING]
        @param seqLengthList: list[BATCH]
        @param hidden: tensor[LAYERS*DIRECTION_NUM, BATCH, HIDDEN_DIM], LAYERS = 1, DIRECTION_NUM = 1
        @param currentWordPos: current word position, starting from 0. It provides information to avoid unnecessary
                               calculation.
        @return:
            tensor[BATCH * EMBEDDING]
        '''
        # learn input shape
        batchSize = paddedInput.shape[1]

        # sanity check
        if not batchSize == len(seqLengthList):
            print("[ERROR] LSTMAdvancedAttention. Fail to calculate context batch due to inconsistency between input data size and sequence length list size.")

        # Compose energy neural network input:
        # for each sentence in given batch, concatenate current hidden state with all words, respectively
        hiddenStateWordEmbeddingList = []   # hiddenStateWordEmbeddingList stores concatenated hidden state and word embeddings
        caredSentenceList = []    # a list[M], storing index of sentences we cared: such sentence is within boundary of current word position
        for sentencePos in range(batchSize):
            # if current sentence is longer than processing word position, ignore it
            if currentWordPos >= seqLengthList[sentencePos]:
                continue
            else:
                caredSentenceList.append(sentencePos)
                for wordPos in range(seqLengthList[sentencePos]):
                    # hiddenStateWordEmbedding is concatenated tensor
                    hiddenStateWordEmbedding = torch.cat([paddedInput[wordPos][sentencePos], hidden[0][sentencePos]], dim = 0)
                    hiddenStateWordEmbeddingList.append(hiddenStateWordEmbedding)

        # transform a list of tensor into a tensor
        # energyNNInput: tensor[M * (EMBEDDING_DIM + HIDDEN_DIM)], where M is the number of all words in current batch
        energyNNInput = torch.stack(hiddenStateWordEmbeddingList)

        # calculate energy
        energySerialized = self.__fcEnergy(energyNNInput)

        # [M, 1] --> [M]
        energySerialized = energySerialized.squeeze()

        # calculate length of sentence that is cared
        caredSentenceLengthList = []
        for sentenceID in caredSentenceList:
            caredSentenceLengthList.append(seqLengthList[sentenceID])

        # calculate context
        contextList = []
        weightList  = []
        for sentencePos in range(batchSize):
            # if it's cared sentence
            if sentencePos in caredSentenceList:
                # calculate sentence start and end position, in energySerialized
                positionInCaredSentenceList = caredSentenceList.index(sentencePos)
                wordPosStart = sum(caredSentenceLengthList[0:positionInCaredSentenceList])
                wordPosEnd   = wordPosStart + seqLengthList[sentencePos]

                # calculate weight by applying softmax on energy
                weightTensor = torch.nn.functional.softmax(energySerialized[wordPosStart:wordPosEnd], dim = 0)

                # context vector is weighted average of all words in a sentence
                context = torch.matmul(weightTensor, paddedInput[0:seqLengthList[sentencePos], sentencePos])

                # store current context
                contextList.append(context)

                # store weight, for review later
                weightList.append(weightTensor.tolist())

            else:  # if it's not cared sentence
                # fake context
                contextList.append(torch.zeros(self.__embDim, device=self.__device))

                # fake weight
                weightList.append(torch.zeros(seqLengthList[sentencePos], device=self.__device))

        # list -> tensor
        contextTensor = torch.stack(contextList)

        return contextTensor, weightList

    def __updateHiddenOutput(self, hiddenOutput, hidden, wordPos, seqLengthList):
        '''
        Update hidden state to be output, according to current hidden state for word position wordPos, and sequence lengths
        @param hiddenOutput:  tensor[LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM]
        @param hidden:        tensor[LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM]
        @param wordPos:       word index, starting from 0
        @param seqLengthList: a list of sequence lengths
        @return: updated hidden states output
            @type: tensor[LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM]
        '''
        # for each sequence in batch
        for sentencePos, length in enumerate(seqLengthList):
            if wordPos == (length - 1):   # wordPos is the last word of current sequence
                hiddenOutput[0, sentencePos] = hidden[0, sentencePos]

        # return hidden states
        return hiddenOutput

    def __maskLSTMOutput(self, lstmOutputList, seqLengthList):
        '''
        Mask lstm output, according to sequence lengths.
        @param lstmOutputList: a list of lstm output: a list of element tensor[BATCH, EMBEDDING]
        @param seqLengthList: a list of sequence lengths
        @return: masked lstm output
            Shape:  Tensor[BATCH * SEQ * DIRECTION_NUM * HIDDEN_DIM], DIRECTION_NUM = 1
        '''

        # apply mask
        lstmOutputTensor = torch.stack(lstmOutputList)   # SEQ * BATCH * HIDDEN_DIM
        for wordPos in range(lstmOutputTensor.shape[0]):
            for sentencePos in range(lstmOutputTensor.shape[1]):
                if wordPos >= seqLengthList[sentencePos]:    # it's out of the range of current sentence
                    lstmOutputTensor[wordPos][sentencePos] = torch.zeros(self.__hiddenDim)

        # return
        lstmOutputTensor = lstmOutputTensor.transpose(0, 1) # SEQ * BATCH * HIDDEN_DIM  --> BATCH * SEQ * HIDDEN_DIM
        lstmOutputTensor = lstmOutputTensor.unsqueeze(2)
        return lstmOutputTensor

class LSTMAdvanced(nn.Module):
    '''
    Implementation of typical LSTM model, in a word-by-word way. Such implementation allows more
    flexible operations.
    This model:
            1. is not bidirectional
            2. has only 1 layer
    This model is used to encode sequence. It output:
            1. Encoded result for each sequence element, or each word.
            2. Last hidden state of all layers
    '''
    def __init__(self, embDim, hiddenDim, device):
        '''
        Init function.
        :param embDim:      Embedding dimension of input data. Integer.
        :param hiddenDim:   Hidden dimension of output data, including hidden state and output. Integer.
        :param device:      Data device
        '''
        super().__init__()

        # populate model parameters
        self.__embDim = embDim
        self.__hiddenDim = hiddenDim
        self.__layerNum = 1
        self.__device = device
        self.__directionNum = 1
        self.__bidirectional = False

        # build up model
        self.__lstm = nn.LSTM(input_size=self.__embDim,
                              hidden_size=self.__hiddenDim,
                              num_layers=self.__layerNum,
                              bidirectional=self.__bidirectional)

    def forward(self, input, hiddenStateInit=torch.Tensor()):
        '''
        Forward function.
        :param self: ...
        :param input:
                        It's supposed to be a list of tensor, the result of word embedding.
                        Sequences are allowed to be of different lengths.
                        Type: list of Tensor
                        Shape: BATCH * SEQ * WORD_EMBEDDING_DIM
                        Example:
                            word1 = [1.1, 1.2, 1.3, 1.4]    # word embedding
                            word2 = [2.1, 2.2, 2.3, 2.4]    # word embedding
                            word3 = [3.1, 3.2, 3.3, 3.4]    # word embedding
                            word4 = [4.1, 4.2, 4.3, 4.4]    # word embedding
                            word5 = [5.1, 5.2, 5.3, 5.4]    # word embedding
                            sentence1 = [word1, word2, word3]
                            sentence2 = [word4, word5]
                            sentence3 = [word1]
                            batch = [sentence1, sentence2, sentence3]
                            batchTensorList = [ torch.Tensor(sentence1),
                                                torch.Tensor(sentence2),
                                                torch.Tensor(sentence3)]
        :param hiddenStateInit:
                        It's supposed to be a tensor.
                        Type: Tensor
                        Shape: [LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM], LAYERS = 1, DIRECTION_NUM = 1
        :return:
                (output, sqlLengths):
                    Encoded result for each element in each sequence in given batch
                    output:
                            LSTM output. Encoded result for each sequence in a batch.
                            Type:   Tensor
                            Shape:  BATCH * SEQ * DIRECTION_NUM * HIDDEN_DIM, DIRECTION_NUM = 1
                    sqlLengths:
                            A tensor storing sequence lengths of input batch
                            Type:   tensor
                            Shape:  BATCH
                hiddenState:
                        hidden state of all layers
                        Type: Tensor
                        Shape: [LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM], LAYER = 1, DIRECTION_NUM = 1
        '''
        # get batch size
        batchSize = len(input)

        # get sequence lengths
        seqLengthList = []
        for sentence in input:
            seqLengthList.append(sentence.shape[0])

        # pad input
        # list[sentenceTensor]   -->  Tensor[BATCH, SENTENCE, EMBEDDING_DIM]
        paddedInput = torch.nn.utils.rnn.pad_sequence(input, batch_first=True)

        # Prepare first hidden and cell state, for each sequence in batch
        # prepare first hidden state
        if hiddenStateInit.nelement() == 0:    # init hidden state into 0, if it's not provided
            hiddenStateInit = torch.zeros(self.__layerNum*self.__directionNum, batchSize, self.__hiddenDim)
        # prepare first cell state, it's all 0
        cellStateInit = torch.zeros(self.__layerNum*self.__directionNum, batchSize, self.__hiddenDim)
        # move hidden state and cell state to device
        hiddenStateInit = Utils.move2Device(hiddenStateInit, self.__device)
        cellStateInit   = Utils.move2Device(cellStateInit, self.__device)

        # hiddenOutput initialize. It will be updated later
        hiddenOutput = torch.Tensor(self.__layerNum*self.__directionNum, batchSize, self.__hiddenDim)  # [LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM]

        # iterate all sequences, in time, or, word dimension: for all batch data, first word, second, word, third word ...
        paddedInput = paddedInput.transpose(0, 1)  # [BATCH, SENTENCE, EMBEDDING]  -> [SENTENCE, BATCH, EMBEDDING]
        # initialize first hidden and cell state
        hidden = hiddenStateInit
        cell = cellStateInit
        # lstmOutputList: a list to store lstm output
        lstmOutputList = []    # a list of element tensor[BATCH, EMBEDDING]
        for wordPos, wordBatch in enumerate(paddedInput):   # wordBatch: [1, BATCH, EMBEDDING]
            # lstmOutput:  (seq_len, batch, num_directions * hidden_size)
            # hidden:      (num_layers * num_directions, batch, hidden_size)
            # cell:        (num_layers * num_directions, batch, hidden_size)
            wordBatch = wordBatch.unsqueeze(0)
            lstmOutput, (hidden, cell) = self.__lstm(wordBatch, (hidden, cell))

            # update hidden state to be output, since some sentence is ended
            hiddenOutput = self.__updateHiddenOutput(hiddenOutput, hidden, wordPos, seqLengthList)

            # store lstm output
            lstmOutputList.append(lstmOutput[0])   # lstmOutput[0]: [BATCH, EMBEDDING]

        # update lstm output, since some output is just padding
        lstmOutputTensor = self.__maskLSTMOutput(lstmOutputList, seqLengthList)

        # return
        seqLengthTensor = torch.tensor(seqLengthList, dtype=torch.long)
        return (lstmOutputTensor, seqLengthTensor), hiddenOutput

    def __updateHiddenOutput(self, hiddenOutput, hidden, wordPos, seqLengthList):
        '''
        Update hidden state to be output, according to current hidden state for word position wordPos, and sequence lengths
        @param hiddenOutput:  tensor[LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM]
        @param hidden:        tensor[LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM]
        @param wordPos:       word index, starting from 0
        @param seqLengthList: a list of sequence lengths
        @return: updated hidden states output
            @type: tensor[LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM]
        '''
        # for each sequence in batch
        for sentencePos, length in enumerate(seqLengthList):
            if wordPos == (length - 1):   # wordPos is the last word of current sequence
                hiddenOutput[0, sentencePos] = hidden[0, sentencePos]

        # return hidden states
        return hiddenOutput

    def __maskLSTMOutput(self, lstmOutputList, seqLengthList):
        '''
        Mask lstm output, according to sequence lengths.
        @param lstmOutputList: a list of lstm output: a list of element tensor[BATCH, EMBEDDING]
        @param seqLengthList: a list of sequence lengths
        @return: masked lstm output
            Shape:  Tensor[BATCH * SEQ * DIRECTION_NUM * HIDDEN_DIM], DIRECTION_NUM = 1
        '''

        # apply mask
        lstmOutputTensor = torch.stack(lstmOutputList)   # SEQ * BATCH * HIDDEN_DIM
        for wordPos in range(lstmOutputTensor.shape[0]):
            for sentencePos in range(lstmOutputTensor.shape[1]):
                if wordPos >= seqLengthList[sentencePos]:    # it's out of the range of current sentence
                    lstmOutputTensor[wordPos][sentencePos] = torch.zeros(self.__hiddenDim)

        # return
        lstmOutputTensor = lstmOutputTensor.transpose(0, 1) # SEQ * BATCH * HIDDEN_DIM  --> BATCH * SEQ * HIDDEN_DIM
        lstmOutputTensor = lstmOutputTensor.unsqueeze(2)
        return lstmOutputTensor

class LSTM(nn.Module):
    '''
        Implementation of typical LSTM model.
        This model is used to encode sequence. It output two result:
            1. Encoded result for each sequence element, or, each word.
            2. Last hidden state, containing info of the whole sequence.
    '''
    def __init__(self, embDim, hiddenDim, device, layerNum = 3, bidirectional = True):
        '''
        Init function.
        :param embDim:      Embedding dimension of input data. Integer.
        :param hiddenDim:   Hidden dimension of output data, including hidden state and output. Integer.
        :param layerNum:    Layer number. Integer.
        :param bidirectional:     True if it's bidirectional, False if it's not
        '''
        super().__init__()

        # populate model parameters
        self.__embDim = embDim
        self.__hiddenDim = hiddenDim
        self.__layerNum = layerNum
        self.__bidirectional = bidirectional
        self.__device = device
        if self.__bidirectional:
            self.__directionNum = 2
        else:
            self.__directionNum = 1

        # build up model
        self.__lstm = nn.LSTM(input_size=self.__embDim,
                              hidden_size=self.__hiddenDim,
                              num_layers=self.__layerNum,
                              bidirectional=self.__bidirectional)

    def showInfo(self):
        '''
        Print model information
        :return: no
        '''
        print ("BiLSTM model info:\n",
               "\t",
               "Embedding dim=%d,"%(self.__embDim),
               "Hidden dim=%d,"%(self.__hiddenDim),
               "layer dim=%d,"%(self.__layerNum),
               "directional num=%d."%(self.__directionNum))

    def forward(self, input, hiddenStateInit = torch.Tensor()):
        '''
        Forward function.
        :param self: ...
        :param input:
                        It's supposed to be a list of tensor, the result of word embedding.
                        Sequences are allowed to be of different lengths.
                        Type: list of Tensor
                        Shape: BATCH * SEQ * WORD_EMBEDDING_DIM
                        Example:
                            word1 = [1.1, 1.2, 1.3, 1.4]    # word embedding
                            word2 = [2.1, 2.2, 2.3, 2.4]    # word embedding
                            word3 = [3.1, 3.2, 3.3, 3.4]    # word embedding
                            word4 = [4.1, 4.2, 4.3, 4.4]    # word embedding
                            word5 = [5.1, 5.2, 5.3, 5.4]    # word embedding
                            sentence1 = [word1, word2, word3]
                            sentence2 = [word4, word5]
                            sentence3 = [word1]
                            batch = [sentence1, sentence2, sentence3]
                            batchTensorList = [ torch.Tensor(sentence1),
                                                torch.Tensor(sentence2),
                                                torch.Tensor(sentence3)]
        :param hiddenState:
                        It's supposed to be a tensor.
                        Type: Tensor
                        Shape: [LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM]
        :return:
                (output, sqlLengths):
                    Encoded result for each element in each sequence in given batch
                    output:
                            LSTM output. Encoded result for each sequence in a batch.
                            Type:   Tensor
                            Shape:  BATCH * SEQ * DIRECTION_NUM * HIDDEN_DIM
                    sqlLengths:
                            A tensor storing sequence lengths of input batch
                            Type:   tensor
                            Shape:  BATCH
                hiddenState:
                        hidden state of all layers
                        Type: Tensor
                        Shape: [LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM]
        '''
        # learn about input data
        seqLengthMax = -1          # max length of sequence
        for sentence in input:
            if sentence.size()[0] > seqLengthMax:
                seqLengthMax = sentence.size()[0]
        batchSize = len(input)        # batch size
        directionNum = self.__directionNum

        # collect sequence lengths, for packing process later
        seqLengths = []
        for sentence in input:
            seqLengths.append(sentence.size()[0])

        batchTensorPadded = nn.utils.rnn.pad_sequence(input, batch_first=True)

        # pack
        embPacked = nn.utils.rnn.pack_padded_sequence(batchTensorPadded, seqLengths, batch_first=True, enforce_sorted=False)

        # LSTM
        if hiddenStateInit.nelement() == 0:    # init hidden state
            hiddenStateInit = torch.zeros(self.__layerNum*self.__directionNum, batchSize, self.__hiddenDim)
        # initialize cell state
        cellStateInit = torch.zeros(self.__layerNum*self.__directionNum, batchSize, self.__hiddenDim)   # init cell state
        # move hidden state and cell state to device
        hiddenStateInit = Utils.move2Device(hiddenStateInit, self.__device)
        cellStateInit   = Utils.move2Device(cellStateInit, self.__device)
        output, (hidden, cell) = self.__lstm(embPacked, (hiddenStateInit, cellStateInit))

        # reshape hidden state, according to pytorch document
        #hiddenReshape = hidden.view(self.__layerNum, directionNum, batchSize, self.__hiddenDim)
        #lastLayerIndex = self.__layerNum - 1    # last layer index
        #hiddenStateLastLayer = hiddenReshape[lastLayerIndex]
        # DIRECTION * BATCH * HIDDEN_DIM  -->   BATCH * DIRECTION * HIDDEN_DIM
        #hiddenStateLastLayer = torch.transpose(hiddenStateLastLayer, 0, 1)
        # LAYERS * DIRECTION * BATCH * HIDDEN_DIM   -->   BATCH * LAYERS * DIRECTION * HIDDEN_DIM
        #hiddenReshape = torch.transpose(hiddenReshape, 1, 2)
        #hiddenReshape = torch.transpose(hiddenReshape, 0, 1)

        # process output
        # pad packed output
        output, seqLengthsOutput = nn.utils.rnn.pad_packed_sequence(output)
        # reshape output, according to pytorch document:
        #               maxSequenceLength * batchSize * directionNum * hiddenSize
        output = output.view(seqLengthMax, batchSize, directionNum, self.__hiddenDim)

        # move batch to the first place:
        # SEQ * BATCH * DIRECTION_NUM * HIDDEN_DIM  --> BATCH * SEQ * DIRECTION_NUM * HIDDEN_DIM
        output = torch.transpose(output, 0, 1)

        return (output, seqLengthsOutput), hidden
