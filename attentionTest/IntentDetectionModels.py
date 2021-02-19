import torch.nn as nn
from collections import OrderedDict
import torch
from LSTM import LSTM
from Utils import Utils
import math

from Object import Object


class IntentDetectionOnlyAttentionModel(torch.nn.Module, Object):
    '''
    Intent detection model. This model is of three components:
        1> LSTM to encode given utterance
        2> Fully connected layers to generate attention
        3> Fully connected layers to make prediction

    Graph:
        utterance  -->    LSTM   --> attention --> Fully connected  -->  predicted probability distribution
    '''
    def __init__(self, inputDim, hiddenDim, outputDim, device, layerNum=3,  name="IntentDetectionModel"):
        '''
        @param inputDim:   model input dimension, it should be word embedding dimension
        @param hiddenDim:  LSTM hidden state dimensions
        @param outputDim:  model output dimension, it should be equal to number of possible labels
        @param device:     device
        @param layerNum:   LSTM layer number
        @param name:       object name
        '''
        # parent class init
        nn.Module.__init__(self)
        Object.__init__(self, name=name)

        # member init
        self.__inputDim = inputDim
        self.__hiddenDim = hiddenDim
        self.__layerNum = layerNum
        self.__outputDim = outputDim
        self.__device = device
        self.__LSTMDirections = 2

        # model components
        self.__lstm = LSTM(embDim=self.__inputDim, hiddenDim=self.__hiddenDim, layerNum=self.__layerNum, device=device)

        # fully connected layer to generate attention
        self.__fcEnergy = nn.Sequential(OrderedDict([
            ('batchNorm', nn.BatchNorm1d(self.__hiddenDim*2 + self.__hiddenDim * 2)),
            ('linear1', nn.Linear(self.__hiddenDim*2 + self.__hiddenDim*2, self.__hiddenDim*2 + self.__hiddenDim*2)),
            ('activate2', nn.Tanh()),
            ('linear2', nn.Linear(self.__hiddenDim*2 + self.__hiddenDim *2, 1)),
            ('activate2', nn.Tanh()),
        ])
        )

        # fully connected layer to predict
        self.__fcPredict = nn.Sequential(OrderedDict([
            ('batchNorm', nn.BatchNorm1d(self.__hiddenDim * 2)),
            ('linear1', nn.Linear(self.__hiddenDim * 2, self.__hiddenDim *2)),
            ('activate2', nn.Tanh()),
            ('linear2', nn.Linear(self.__hiddenDim * 2, self.__outputDim)),
            ('activate2', nn.Tanh()),
            ('softmax', nn.Softmax())
        ])
        )

    def forward(self, batchData):
        '''
        @param batchData: Please refer to LSTM input comments
        @return: output: tensor[BATCH * OUTPUT_DIM]
                 weightList: list[BATCH]
        '''
        # pass data to LSTM
        (lstmOutput, seqLengthList), hiddenState = self.__lstm(batchData)

        # reshape hidden state into:   LAYER * DIRECTIONS * BATCH * HIDDEN_DIM
        batchSize = len(batchData)
        hiddenState = hiddenState.view(self.__layerNum, self.__LSTMDirections, batchSize, self.__hiddenDim)

        # concatenate two directions of last layer, as utterance encoding result
        sentenceEmbedding = self.__generateSentenceEmbeddingFromHiddenState(hiddenState)

        # concatenate two directions of LSTM output
        # [BATCH, SEQ, DIRECTION_NUM, HIDDEN_DIM]  --> [BATCH, SEQ, 2 * HIDDEN_DIM]
        concatLstmOutput = torch.cat([lstmOutput[:,:,0], lstmOutput[:,:,1]], dim = 2)

        # calculate context
        contextBatch, weightList = self.__calculateContextBatch(concatLstmOutput, seqLengthList, sentenceEmbedding)

        # pass sentence embedding to fully connected layer
        output = self.__fcPredict(contextBatch)

        return output, weightList

    def __calculateContextBatch(self, inputData, seqLenList, sentenceEmbedding):
        '''
        Calculate context: a weighted average of all words (LSTM encoding result).
        @param inputData: input data. It should be LSTM output data. Two directions are supposed to be already connected.
                @type: tensor[BATCH, SEQ, 2 * HIDDEN_DIM]
        @param seqLenList: a tensor containing sentence lengths
                @type: tensor[BATCH]
        @param sentenceEmbedding: sentence embedding, output of BiLSTM
                @type: tensor[BATCH * (2 * HIDDEN_DIM)]
        @return: contextTensor: context for current batch
                @type: tensor[BATCH * (2 * HIDDEN_DIM)]
                weightList: A list of weigth , containing weight distribution of every word over intent
                @type: list[BATCH], element is list[N], N is sentence length
        '''
        # learn input shape
        batchSize = inputData.shape[0]

        # sanity check
        if not batchSize == len(seqLenList):
            print("[ERROR] IntentDetectionAttentionModel. Fail to calculate context batch due to inconsistency between input data size and sequence length list size.")

        # Compose energy neural network input:
        # for each sentence in given batch, concatenate sentence embedding with all words, respectively
        sentenceEmbeddingWordEmbeddingList = []   # sentenceEmbeddingWordEmbeddingList stores concatenated sentence embedding and word embeddings
        for sentencePos in range(batchSize):
            for wordPos in range(seqLenList[sentencePos]):
                # hiddenStateWordEmbedding is concatenated tensor
                wordEmbedding = inputData[sentencePos][wordPos]
                sentenceEmbeddingWordEmbedding = torch.cat((wordEmbedding, sentenceEmbedding[sentencePos]), dim=0)
                sentenceEmbeddingWordEmbeddingList.append(sentenceEmbeddingWordEmbedding)

        # transform a list of tensor into a tensor
        # energyNNInput: tensor[M * (EMBEDDING_DIM + HIDDEN_DIM)], where M is the number of all words in current batch
        energyNNInput = torch.stack(sentenceEmbeddingWordEmbeddingList)

        # calculate energy
        energySerialized = self.__fcEnergy(energyNNInput)

        # [M, 1] --> [M]
        energySerialized = energySerialized.squeeze()

        # calculate context
        contextList = []
        weightList  = []
        for sentencePos in range(batchSize):
            # calculate sentence start and end position, in energySerialized
            wordPosStart = sum(seqLenList[0:sentencePos])
            wordPosEnd   = wordPosStart + seqLenList[sentencePos]

            # calculate weight by applying softmax on energy
            weightTensor = torch.nn.functional.softmax(energySerialized[wordPosStart:wordPosEnd], dim = 0)

            # context vector is weighted average of all words in a sentence
            context = torch.matmul(weightTensor, inputData[sentencePos, 0:seqLenList[sentencePos]])

            # store current context
            contextList.append(context)

            # store weight, for review later
            weightList.append(weightTensor.tolist())

        # list -> tensor
        contextTensor = torch.stack(contextList)

        return contextTensor, weightList


    def __generateSentenceEmbeddingFromHiddenState(self, hiddenState):
        '''
        Generate sentence embedding from hidden states, by concatenating the left and right hidden states,
        of the last layer of LSTM.
        @param hiddenState: tensor[LAYER * DIRECTION * BATCH * HIDDEN_DIM]
        @return: tensor[BATCH * (2 * HIDDEN_DIM)]
        '''
        # get last layer index
        shape = hiddenState.shape
        lastLayerIndex = shape[0] - 1

        # last layer hidden state
        lastLayerHiddenState = hiddenState[lastLayerIndex]    # DIRECTIONS * BATCH * HIDDEN_DIM

        # transpose
        # DIRECTIONS * BATCH * HIDDEN_DIM   -->   BATCH * DIRECTIONS * HIDDEN_DIM
        lastLayerHiddenState = torch.transpose(lastLayerHiddenState, 0, 1)

        # concatenate two directions
        # Concatenation: BATCH * DIRECTIONS * HIDDEN_DIM  -->  BATCH  * (2 * HIDDEN_DIM)
        sentenceEmbedding = torch.cat((lastLayerHiddenState[:,0], lastLayerHiddenState[:,1]), 1)

        return sentenceEmbedding

class IntentDetectionAttentionModel(torch.nn.Module, Object):
    '''
    Intent detection model. This model is of three components:
        1> LSTM to encode given utterance
        2> Fully conneected layers to generate attention
        3> Fully connected layers to make prediction

    Graph:
        utterance  -->    LSTM   --> Fully connected  -->  predicted probability distribution
                            |            ^
                            |           |
                        attention  ------
    '''
    def __init__(self, inputDim, hiddenDim, outputDim, device, layerNum=3,  name="IntentDetectionModel"):
        '''
        @param inputDim:   model input dimension, it should be word embedding dimension
        @param hiddenDim:  LSTM hidden state dimensions
        @param outputDim:  model output dimension, it should be equal to number of possible labels
        @param device:     device
        @param layerNum:   LSTM layer number
        @param name:       object name
        '''
        # parent class init
        nn.Module.__init__(self)
        Object.__init__(self, name=name)

        # member init
        self.__inputDim = inputDim
        self.__hiddenDim = hiddenDim
        self.__layerNum = layerNum
        self.__outputDim = outputDim
        self.__device = device
        self.__LSTMDirections = 2

        # model components
        self.__lstm = LSTM(embDim=self.__inputDim, hiddenDim=self.__hiddenDim, layerNum=self.__layerNum, device=device)

        # fully connected layer to generate attention
        self.__fcEnergy = nn.Sequential(OrderedDict([
            ('batchNorm', nn.BatchNorm1d(self.__hiddenDim*2 + self.__hiddenDim * 2)),
            ('linear1', nn.Linear(self.__hiddenDim*2 + self.__hiddenDim*2, self.__hiddenDim*2 + self.__hiddenDim*2)),
            ('activate2', nn.Tanh()),
            ('linear2', nn.Linear(self.__hiddenDim*2 + self.__hiddenDim *2, 1)),
            ('activate2', nn.Tanh()),
        ])
        )

        # fully connected layer to predict
        self.__fcPredict = nn.Sequential(OrderedDict([
            ('batchNorm', nn.BatchNorm1d(self.__hiddenDim * 2 + self.__hiddenDim * 2)),
            ('linear1', nn.Linear(self.__hiddenDim * 4, self.__hiddenDim *4)),
            ('activate2', nn.Tanh()),
            ('linear2', nn.Linear(self.__hiddenDim * 4, self.__outputDim)),
            ('activate2', nn.Tanh()),
            ('softmax', nn.Softmax())
        ])
        )

    def forward(self, batchData):
        '''
        @param batchData: Please refer to LSTM input comments
        @return: output: tensor[BATCH * OUTPUT_DIM]
                 weightList: list[BATCH]
        '''
        # pass data to LSTM
        (lstmOutput, seqLengthList), hiddenState = self.__lstm(batchData)

        # reshape hidden state into:   LAYER * DIRECTIONS * BATCH * HIDDEN_DIM
        batchSize = len(batchData)
        hiddenState = hiddenState.view(self.__layerNum, self.__LSTMDirections, batchSize, self.__hiddenDim)

        # concatenate two directions of last layer, as utterance encoding result
        sentenceEmbedding = self.__generateSentenceEmbeddingFromHiddenState(hiddenState)

        # concatenate two directions of LSTM output
        # [BATCH, SEQ, DIRECTION_NUM, HIDDEN_DIM]  --> [BATCH, SEQ, 2 * HIDDEN_DIM]
        concatLstmOutput = torch.cat([lstmOutput[:,:,0], lstmOutput[:,:,1]], dim = 2)

        # calculate context
        contextBatch, weightList = self.__calculateContextBatch(concatLstmOutput, seqLengthList, sentenceEmbedding)

        # concatenate context and sentence embedding
        predictInput = torch.cat((sentenceEmbedding, contextBatch), dim = 1)

        # pass sentence embedding to fully connected layer
        output = self.__fcPredict(predictInput)

        return output, weightList

    def __calculateContextBatch(self, inputData, seqLenList, sentenceEmbedding):
        '''
        Calculate context: a weighted average of all words (LSTM encoding result).
        @param inputData: input data. It should be LSTM output data. Two directions are supposed to be already connected.
                @type: tensor[BATCH, SEQ, 2 * HIDDEN_DIM]
        @param seqLenList: a tensor containing sentence lengths
                @type: tensor[BATCH]
        @param sentenceEmbedding: sentence embedding, output of BiLSTM
                @type: tensor[BATCH * (2 * HIDDEN_DIM)]
        @return: contextTensor: context for current batch
                @type: tensor[BATCH * (2 * HIDDEN_DIM)]
                weightList: A list of weigth , containing weight distribution of every word over intent
                @type: list[BATCH], element is list[N], N is sentence length
        '''
        # learn input shape
        batchSize = inputData.shape[0]

        # sanity check
        if not batchSize == len(seqLenList):
            print("[ERROR] IntentDetectionAttentionModel. Fail to calculate context batch due to inconsistency between input data size and sequence length list size.")

        # Compose energy neural network input:
        # for each sentence in given batch, concatenate sentence embedding with all words, respectively
        sentenceEmbeddingWordEmbeddingList = []   # sentenceEmbeddingWordEmbeddingList stores concatenated sentence embedding and word embeddings
        for sentencePos in range(batchSize):
            for wordPos in range(seqLenList[sentencePos]):
                # hiddenStateWordEmbedding is concatenated tensor
                wordEmbedding = inputData[sentencePos][wordPos]
                sentenceEmbeddingWordEmbedding = torch.cat((wordEmbedding, sentenceEmbedding[sentencePos]), dim=0)
                sentenceEmbeddingWordEmbeddingList.append(sentenceEmbeddingWordEmbedding)

        # transform a list of tensor into a tensor
        # energyNNInput: tensor[M * (EMBEDDING_DIM + HIDDEN_DIM)], where M is the number of all words in current batch
        energyNNInput = torch.stack(sentenceEmbeddingWordEmbeddingList)

        # calculate energy
        energySerialized = self.__fcEnergy(energyNNInput)

        # [M, 1] --> [M]
        energySerialized = energySerialized.squeeze()

        # calculate context
        contextList = []
        weightList  = []
        for sentencePos in range(batchSize):
            # calculate sentence start and end position, in energySerialized
            wordPosStart = sum(seqLenList[0:sentencePos])
            wordPosEnd   = wordPosStart + seqLenList[sentencePos]

            # calculate weight by applying softmax on energy
            weightTensor = torch.nn.functional.softmax(energySerialized[wordPosStart:wordPosEnd], dim = 0)

            # context vector is weighted average of all words in a sentence
            context = torch.matmul(weightTensor, inputData[sentencePos, 0:seqLenList[sentencePos]])

            # store current context
            contextList.append(context)

            # store weight, for review later
            weightList.append(weightTensor.tolist())

        # list -> tensor
        contextTensor = torch.stack(contextList)

        return contextTensor, weightList


    def __generateSentenceEmbeddingFromHiddenState(self, hiddenState):
        '''
        Generate sentence embedding from hidden states, by concatenating the left and right hidden states,
        of the last layer of LSTM.
        @param hiddenState: tensor[LAYER * DIRECTION * BATCH * HIDDEN_DIM]
        @return: tensor[BATCH * (2 * HIDDEN_DIM)]
        '''
        # get last layer index
        shape = hiddenState.shape
        lastLayerIndex = shape[0] - 1

        # last layer hidden state
        lastLayerHiddenState = hiddenState[lastLayerIndex]    # DIRECTIONS * BATCH * HIDDEN_DIM

        # transpose
        # DIRECTIONS * BATCH * HIDDEN_DIM   -->   BATCH * DIRECTIONS * HIDDEN_DIM
        lastLayerHiddenState = torch.transpose(lastLayerHiddenState, 0, 1)

        # concatenate two directions
        # Concatenation: BATCH * DIRECTIONS * HIDDEN_DIM  -->  BATCH  * (2 * HIDDEN_DIM)
        sentenceEmbedding = torch.cat((lastLayerHiddenState[:,0], lastLayerHiddenState[:,1]), 1)

        return sentenceEmbedding

class IntentDetectionModel(torch.nn.Module, Object):
    '''
    Intent detection model. This model is of two components:
        1> LSTM to encode given utterance
        2> Fully connected layers to make prediction

    Graph:
        utterance  -->    LSTM   --> Fully connected  -->  predicted probability distribution
    '''
    def __init__(self, inputDim, hiddenDim, outputDim, device, layerNum=3,  name="IntentDetectionModel"):
        '''
        @param inputDim:   model input dimension, it should be word embedding dimension
        @param hiddenDim:  LSTM hidden state dimensions
        @param outputDim:  model output dimension, it should be equal to number of possible labels
        @param device:     device
        @param layerNum:   LSTM layer number
        @param name:       object name
        '''
        # parent class init
        nn.Module.__init__(self)
        Object.__init__(self, name=name)

        # member init
        self.__inputDim = inputDim
        self.__hiddenDim = hiddenDim
        self.__layerNum = layerNum
        self.__outputDim = outputDim
        self.__device = device
        self.__LSTMDirections = 2

        # model components
        self.__lstm = LSTM(embDim=self.__inputDim, hiddenDim=self.__hiddenDim, layerNum=self.__layerNum, device=device)
        # fully connected layer
        self.fc1 = nn.Sequential(OrderedDict([
                  ('batchNorm', nn.BatchNorm1d(self.__hiddenDim*2)),
                  ('linear1', nn.Linear(self.__hiddenDim * 2, self.__hiddenDim *2)),
                  ('activate2', nn.Tanh()),
                  ('linear2', nn.Linear(self.__hiddenDim * 2, self.__outputDim)),
                  ('activate2', nn.Tanh()),
                  ('softmax', nn.Softmax())
                ])
        )

    def forward(self, batchData):
        '''
        @param batchData: Please refer to LSTM input comments
        @return: tensor[BATCH * OUTPUT_DIM]
        '''
        # pass data to LSTM
        (_, _), hiddenState = self.__lstm(batchData)

        # reshape hidden state into:   LAYER * DIRECTIONS * BATCH * HIDDEN_DIM
        batchSize = len(batchData)
        hiddenState = hiddenState.view(self.__layerNum, self.__LSTMDirections, batchSize, self.__hiddenDim)

        # concatenate two directions of last layer, as utterance encoding result
        sentenceEmbedding = self.__generateSentenceEmbeddingFromHiddenState(hiddenState)

        # pass sentence embedding to fully connected layer
        output = self.fc1(sentenceEmbedding)

        return output

    def __generateSentenceEmbeddingFromHiddenState(self, hiddenState):
        '''
        Generate sentence embedding from hidden states, by concatenating the left and right hidden states,
        of the last layer of LSTM.
        @param hiddenState: tensor[LAYER * DIRECTION * BATCH * HIDDEN_DIM]
        @return: tensor[BATCH * (2 * HIDDEN_DIM)]
        '''
        # get last layer index
        shape = hiddenState.shape
        lastLayerIndex = shape[0] - 1

        # last layer hidden state
        lastLayerHiddenState = hiddenState[lastLayerIndex]    # DIRECTIONS * BATCH * HIDDEN_DIM

        # transpose
        # DIRECTIONS * BATCH * HIDDEN_DIM   -->   BATCH * DIRECTIONS * HIDDEN_DIM
        lastLayerHiddenState = torch.transpose(lastLayerHiddenState, 0, 1)

        # concatenate two directions
        # Concatenation: BATCH * DIRECTIONS * HIDDEN_DIM  -->  BATCH  * (2 * HIDDEN_DIM)
        sentenceEmbedding = torch.cat((lastLayerHiddenState[:,0], lastLayerHiddenState[:,1]), 1)

        return sentenceEmbedding