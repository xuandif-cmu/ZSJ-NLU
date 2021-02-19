from commonVar import STATUS_FAIL
from commonVar import STATUS_OK
from commonVar import STATUS_WARNING
from LSTM import LSTM
from LSTM import LSTMAdvanced
from LSTM import LSTMAdvancedAttention
from LSTM import LSTMAdvancedAttentionContextOnly
import torch.nn as nn
import torch

from Object import Object


class Encoder(nn.Module, Object):
    '''
    Class of encoder. It's encoder part of encoder-decoder framework.
    '''
    def __init__(self, inputDim, hiddenDim, device, rnnLayer=3, name="Encoder"):
        '''
        @param inputDim: input dimension of RNN unit.
        @param hiddenDim: hidden dimension of RNN unit. For LSTM, it's also output dimension
        @param rnnLayer: RNN layer number
        @param name: instance name
        '''
        # parent init
        nn.Module.__init__(self)
        Object.__init__(self, name=name)

        # member init
        self.__device = device
        self.__rnnLayerNum = rnnLayer
        self.__inputDim = inputDim
        self.__hiddenDim = hiddenDim
        self.__device = device
        self.__directionNum = 2   # it's hard coded
        self.__RNN = LSTM(embDim=self.__inputDim, hiddenDim=self.__hiddenDim, layerNum=self.__rnnLayerNum, device=device)  # RNN is set to LSTM

    def forward(self, batchData, resultPadding = True):
        '''
        @param batchData: input data for current batch. For details please refer to which RNN is being used.
        @param resultPadding: True, encoding result will be padded. Lengths information will also be returned for usage.
                              False: result will be unpadded before returned.
        @return: output, seqLenList, lastHiddenState
                if resultPadding == True:
                    return Tensor[BATCH * SEQ * (2 * HIDDEN_DIM)], seqLengthsOutput, Tensor[BATCH * (2 * HIDDEN_DIM)], hiddenState
                else:
                    return list[Tensor[SEQ * (2 * HIDDEN_DIM)]], None, Tensor[BATCH * (2 * HIDDEN_DIM)], hiddenState
        '''

        # encode it with RNN
        (output, seqLengthsOutput), hiddenState = self.__RNN(batchData)

        # concatenate two directions
        output = self.__concatOutputTwoDirectionLSTM(output)

        # remove padding, or not
        if resultPadding:
            return output, seqLengthsOutput, hiddenState
        else:
            output = self.__unpadEncodingResult(output, seqLengthsOutput)
            return output, None, hiddenState

    def __concatOutputTwoDirectionLSTM(self, input):
        '''
        @param input: input data
               Type:   Tensor
               Shape:  BATCH * SEQ * DIRECTION_NUM * HIDDEN_DIM
        @return:
                Type: Tensor
                Shape: BATCH * SEQ * (2 * HIDDEN_DIM)
        '''
        return torch.cat((input[:,:,0], input[:,:,1]), 2)

    def __unpadEncodingResult(self, input, seqLenList):
        '''
        @param input: For LSTM it's:
               Tensor[BATCH * SEQ * (2 * HIDDEN_DIM)]
        @param seqLenList: sequence length list
                Type: List
                Shape: BATCH
        @return:  a Tensor list, containing unpadded encoding result
                Type: list[Tensor[SEQ * (2 * HIDDEN_DIM)]]
        '''
        codeList = []
        for index, sentence in enumerate(input):
            codeList.append(sentence[0:seqLenList[index]])
        return codeList

    def getLSTMLayerNum(self):
        return self.__rnnLayerNum

    def getLSTMInputDim(self):
        return self.__inputDim

    def getLSTMHiddenDim(self):
        return self.__hiddenDim

    def getLSTMDirectionNum(self):
        return self.__directionNum


class Decoder(nn.Module, Object):
    '''
    Class of decoder. It's decoder part of encoder-decoder framework.
    '''
    def __init__(self, inputDim, hiddenDim, device, rnnLayer = 3, name="Decoder"):
        '''
        @param inputDim: input dimension of RNN unit.
        @param hiddenDim: hidden dimension of RNN unit. For LSTM, it's also output dimension
        @param rnnLayer: RNN layer number
        @param name: instance name
        '''
        # parent init
        nn.Module.__init__(self)
        Object.__init__(self, name=name)

        # member init
        self.__device = device
        self.__RNN = LSTM(embDim=inputDim, hiddenDim=hiddenDim, layerNum=rnnLayer, bidirectional = True, device=self.__device)  # RNN is set to LSTM

    def forward(self, batchData, hiddenStateInit = torch.Tensor()):
        '''
        @param batchData: input data for current batch. For details please refer to which RNN is being used.
        @param hiddenStateInit: initiate hidden state to feed into RNN
        @return: please refer to RNN class
        '''
        (output, seqLengthsOutput), _ = self.__RNN(batchData, hiddenStateInit)
        return output, seqLengthsOutput

class DecoderLSTMAdvanced(nn.Module, Object):
    '''
    Class of decoder using LSTMAdvanced. It's decoder part of encoder-decoder framework.
    DecoderLSTMAdvanced is built up by LSTM, but in a word-by-word way. Such implementation allows more
    flexible operation.
    '''
    def __init__(self, inputDim, hiddenDim, device, name="DecoderLSTMAdvanced"):
        '''
        @param inputDim: input dimension of RNN unit.
        @param hiddenDim: hidden dimension of RNN unit. For LSTM, it's also output dimension
        @param device: device
        @param name: instance name
        '''
        # parent init
        nn.Module.__init__(self)
        Object.__init__(self, name=name)

        # member init
        self.__device = device
        self.__layerNum = 1
        self.__directionNum = 1
        self.__RNN = LSTMAdvanced(embDim=inputDim, hiddenDim=hiddenDim, device=self.__device)  # RNN is set to LSTM

    def getLayerNum(self):
        return self.__layerNum

    def getDirectionNum(self):
        return self.__directionNum

    def forward(self, batchData, hiddenStateInit = torch.Tensor()):
        '''
        @param batchData: input data for current batch. For details please refer to which RNN is being used.
        @param hiddenStateInit: initiate hidden state to feed into RNN
        @return: please refer to RNN class
        '''
        (output, seqLengthsOutput), _ = self.__RNN(batchData, hiddenStateInit)
        return output, seqLengthsOutput


class DecoderLSTMAdvancedAttention(nn.Module, Object):
    '''
    Class of decoder using LSTMAdvanced, with attention mechanism. It's decoder part of encoder-decoder framework.
    DecoderLSTMAdvancedAttention is built up by LSTM, in a word-by-word way. Such implementation allows more
    flexible operation.
    Attention mechanism is introduced:
                word_context_i = sum_j(alpha(i,j) * h_j)
                where: sum_j means sum for every word j
                       alpha(i,j) means weight of j on word_context_i, and alpha() is a learnable function
                       h_j is encoding result of word j, by BiLSTM

    For attention mechanism, please refer to paper 'Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling'
    '''
    def __init__(self, inputDim, hiddenDim, device, name="DecoderLSTMAdvancedAttention"):
        '''
        @param inputDim: input dimension of RNN unit.
        @param hiddenDim: hidden dimension of RNN unit. For LSTM, it's also output dimension
        @param device: device
        @param name: instance name
        '''
        # parent init
        nn.Module.__init__(self)
        Object.__init__(self, name=name)

        # member init
        self.__device = device
        self.__layerNum = 1
        self.__directionNum = 1
        self.__RNN = LSTMAdvancedAttention(embDim=inputDim, hiddenDim=hiddenDim, device=self.__device)  # RNN is set to LSTM

    def getLayerNum(self):
        return self.__layerNum

    def getDirectionNum(self):
        return self.__directionNum

    def forward(self, batchData, hiddenStateInit = torch.Tensor()):
        '''
        @param batchData: input data for current batch. For details please refer to which RNN is being used.
        @param hiddenStateInit: initiate hidden state to feed into RNN
        @return: please refer to RNN class
        '''
        (output, seqLengthsOutput), _, attentionWeight = self.__RNN(batchData, hiddenStateInit)
        return output, seqLengthsOutput, attentionWeight


class DecoderLSTMAdvancedAttentionContextOnly(nn.Module, Object):
    '''
    Class of decoder using LSTMAdvanced, with attention mechanism. It's decoder part of encoder-decoder framework.
    DecoderLSTMAdvancedAttention is built up by LSTM, in a word-by-word way. Such implementation allows more
    flexible operation.
    Attention mechanism is introduced:
                word_context_i = sum_j(alpha(i,j) * h_j)
                where: sum_j means sum for every word j
                       alpha(i,j) means weight of j on word_context_i, and alpha() is a learnable function
                       h_j is encoding result of word j, by BiLSTM

    For attention mechanism, please refer to paper 'Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling'

    It exploits LSTM. Its input is only context, not concatenated context and word.
    '''
    def __init__(self, inputDim, hiddenDim, device, name="DecoderLSTMAdvancedAttention"):
        '''
        @param inputDim: input dimension of RNN unit.
        @param hiddenDim: hidden dimension of RNN unit. For LSTM, it's also output dimension
        @param device: device
        @param name: instance name
        '''
        # parent init
        nn.Module.__init__(self)
        Object.__init__(self, name=name)

        # member init
        self.__device = device
        self.__layerNum = 1
        self.__directionNum = 1
        self.__RNN = LSTMAdvancedAttentionContextOnly(embDim=inputDim, hiddenDim=hiddenDim, device=self.__device)  # RNN is set to LSTM

    def getLayerNum(self):
        return self.__layerNum

    def getDirectionNum(self):
        return self.__directionNum

    def forward(self, batchData, hiddenStateInit = torch.Tensor()):
        '''
        @param batchData: input data for current batch. For details please refer to which RNN is being used.
        @param hiddenStateInit: initiate hidden state to feed into RNN
        @return: please refer to RNN class
        '''
        (output, seqLengthsOutput), _, attentionWeight = self.__RNN(batchData, hiddenStateInit)
        return output, seqLengthsOutput, attentionWeight
