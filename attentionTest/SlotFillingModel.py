import EncoderDecoder
import torch.nn as nn
from collections import OrderedDict
import torch

from Object import Object


class SlotFillingModelLSTMAdvancedAttentionContextOnly(nn.Module, Object):
    '''
    It's a model to do slot filling.
    The model is of encoder-decoder framework. Both coders are LSTM. Decoder output is passed to
    2 fully connected layers. The final output for each sequence element is a probability
    distribution over all possible labels. Attention mechanism of context for each word:
                word_context_i = sum_j(alpha(i,j) * h_j)
                where: sum_j means sum for every word j
                       alpha(i,j) means weight of j on word_context_i, and alpha() is a learnable function
                       h_j is encoding result of word j, by BiLSTM

    For attention mechanism, please refer to paper 'Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling'

    The decoder is LSTM. Its input is only context, not concatenated context and word.

    Graph:
          sequence  -->  Encoder (LSTM, 3 layer, 2 directions) --> Decoder(LSTM, 1 layer , 1 direction, attention mechanism)  --> fully connected layer --> output
          note: Last hidden state of the last layer of LSTM1 is fed into LSTM2
    '''
    def __init__(self, encoderInputDim, encoderHiddenDim, decoderOutputDim, outputDim, device, name="slotFillingModelB"):
        # parent class init
        nn.Module.__init__(self)
        Object.__init__(self, name=name)

        # member init
        self.__encoderInputDim = encoderInputDim
        self.__encoderHiddenDim = encoderHiddenDim
        self.__decoderOutputDim = decoderOutputDim
        self.__outputDim = outputDim
        self.__device = device

        # model component
        # encoder-decoder framework
        self.__encoder = EncoderDecoder.Encoder(inputDim=self.__encoderInputDim, hiddenDim=self.__encoderHiddenDim, device = device)     # encoder
        self.__decoder = EncoderDecoder.DecoderLSTMAdvancedAttentionContextOnly(inputDim=2*self.__encoderHiddenDim, hiddenDim=self.__decoderOutputDim, device = device)  # decoder
        # fully connected layer
        self.fc1 = nn.Sequential(OrderedDict([
            ('batchNorm', nn.BatchNorm1d(self.__decoderOutputDim)),
            ('linear1', nn.Linear(self.__decoderOutputDim, self.__decoderOutputDim)),
            ('activate2', nn.Tanh()),
            ('linear2', nn.Linear(self.__decoderOutputDim, self.__outputDim)),
            ('activate2', nn.Tanh()),
            ('softmax', nn.Softmax())
        ])
        )

    def forward(self, batchData):
        '''
        Forward function.
        @param batchData: batch data
        @return: final output
        '''
        # encode
        codingResult, _, hiddenState = self.__encoder(batchData, resultPadding=False)

        # get hidden state of the last layer, left-to-right direction, to feed into decoder
        # [LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM] --> [LAYERS, DIRECTION_NUM, BATCH, HIDDEN_DIM]
        layerNum = self.__encoder.getLSTMLayerNum()
        directionNum = self.__encoder.getLSTMDirectionNum()
        batchSize = len(batchData)
        hiddenState = hiddenState.view(layerNum, directionNum, batchSize, self.__encoderHiddenDim)
        # fetch hidden state of the last layer, left to right direction: [BATCH, HIDDEN_DIM]
        hiddenLastLayerLeftToRight = hiddenState[layerNum-1][0]
        # [BATCH, HIDDEN_DIM]  -->  [LAYER * DIRECTION_NUM, BATCH, HIDDEN_DIM]
        hiddenLastLayerLeftToRight = hiddenLastLayerLeftToRight.unsqueeze(0)

        # decode
        decodingResult, lengths, attention = self.__decoder(codingResult, hiddenLastLayerLeftToRight)

        # fully connected
        # [BATCH, SEQ, DIRECTION_NUM, HIDDEN_DIM]  --> [BATCH, SEQ, HIDDEN_DIM], since DIRECTION_NUM = 1
        decodingResult = decodingResult.squeeze()
        shape = decodingResult.shape    # BATCH * SEQ * HIDDEN_DIM
        decodingResultView = decodingResult.reshape(shape[0] * shape[1], shape[2])
        fcResult = self.fc1(decodingResultView)
        output = fcResult.view(shape[0], shape[1], 1, self.__outputDim)

        #return decodingResult, lengths
        return output, lengths, attention


class SlotFillingModelLSTMAdvancedAttention(nn.Module, Object):
    '''
    It's a mdodel to do slot filling.
    The model is of encoder-decoder framework. Both coders are LSTM. Decoder output is passed to
    2 fully connected layers. The final output for each sequence element is a probability
    distribution over all possible labels. Attention mechanism to context for each word:
                word_context_i = sum_j(alpha(i,j) * h_j)
                where: sum_j means sum for every word j
                       alpha(i,j) means weight of j on word_context_i, and alpha() is a learnable function
                       h_j is encoding result of word j, by BiLSTM

    For attention mechanism, please refer to paper 'Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling'

    Graph:
          sequence  -->  Encoder (LSTM, 3 layer, 2 directions) --> Decoder(LSTM, 1 layer , 1 direction, attention mechanism)  --> fully connected layer --> output
          note: Last hidden state of the last layer of LSTM1 is fed into LSTM2
    '''
    def __init__(self, encoderInputDim, encoderHiddenDim, decoderOutputDim, outputDim, device, name="slotFillingModelB"):
        # parent class init
        nn.Module.__init__(self)
        Object.__init__(self, name=name)

        # member init
        self.__encoderInputDim = encoderInputDim
        self.__encoderHiddenDim = encoderHiddenDim
        self.__decoderOutputDim = decoderOutputDim
        self.__outputDim = outputDim
        self.__device = device

        # model component
        # encoder-decoder framework
        self.__encoder = EncoderDecoder.Encoder(inputDim=self.__encoderInputDim, hiddenDim=self.__encoderHiddenDim, device = device)     # encoder
        self.__decoder = EncoderDecoder.DecoderLSTMAdvancedAttention(inputDim=2*self.__encoderHiddenDim, hiddenDim=self.__decoderOutputDim, device = device)  # decoder
        # fully connected layer
        self.fc1 = nn.Sequential(OrderedDict([
            ('batchNorm', nn.BatchNorm1d(self.__decoderOutputDim)),
            ('linear1', nn.Linear(self.__decoderOutputDim, self.__decoderOutputDim)),
            ('activate2', nn.Tanh()),
            ('linear2', nn.Linear(self.__decoderOutputDim, self.__outputDim)),
            ('activate2', nn.Tanh()),
            ('softmax', nn.Softmax())
        ])
        )

    def forward(self, batchData):
        '''
        Forward function.
        @param batchData: batch data
        @return: final output
        '''
        # encode
        codingResult, _, hiddenState = self.__encoder(batchData, resultPadding=False)

        # get hidden state of the last layer, left-to-right direction, to feed into decoder
        # [LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM] --> [LAYERS, DIRECTION_NUM, BATCH, HIDDEN_DIM]
        layerNum = self.__encoder.getLSTMLayerNum()
        directionNum = self.__encoder.getLSTMDirectionNum()
        batchSize = len(batchData)
        hiddenState = hiddenState.view(layerNum, directionNum, batchSize, self.__encoderHiddenDim)
        # fetch hidden state of the last layer, left to right direction: [BATCH, HIDDEN_DIM]
        hiddenLastLayerLeftToRight = hiddenState[layerNum-1][0]
        # [BATCH, HIDDEN_DIM]  -->  [LAYER * DIRECTION_NUM, BATCH, HIDDEN_DIM]
        hiddenLastLayerLeftToRight = hiddenLastLayerLeftToRight.unsqueeze(0)

        # decode
        decodingResult, lengths, attention = self.__decoder(codingResult, hiddenLastLayerLeftToRight)

        # fully connected
        # [BATCH, SEQ, DIRECTION_NUM, HIDDEN_DIM]  --> [BATCH, SEQ, HIDDEN_DIM], since DIRECTION_NUM = 1
        decodingResult = decodingResult.squeeze()
        shape = decodingResult.shape    # BATCH * SEQ * HIDDEN_DIM
        decodingResultView = decodingResult.reshape(shape[0] * shape[1], shape[2])
        fcResult = self.fc1(decodingResultView)
        output = fcResult.view(shape[0], shape[1], 1, self.__outputDim)

        #return decodingResult, lengths
        return output, lengths, attention


class SlotFillingModelLSTMAdvanced(nn.Module, Object):
    '''
    It's a mdodel to do slot filling.
    The model is of encoder-decoder framework. Both coders are LSTM. Decoder output is passed to
    2 fully connected layers. The final output for each sequence element is a probability
    distribution over all possible labels.
    Graph:
          sequence  -->  Encoder (LSTM, 3 layer, 2 directions) --> Decoder(LSTM, 1 layer , 1 direction)  --> fully connected layer --> output
          note: Last hidden state of the last layer of LSTM1 is fed into LSTM2
    '''
    def __init__(self, encoderInputDim, encoderHiddenDim, decoderOutputDim, outputDim, device, name="slotFillingModelB"):
        # parent class init
        nn.Module.__init__(self)
        Object.__init__(self, name=name)

        # member init
        self.__encoderInputDim = encoderInputDim
        self.__encoderHiddenDim = encoderHiddenDim
        self.__decoderOutputDim = decoderOutputDim
        self.__outputDim = outputDim
        self.__device = device

        # model component
        # encoder-decoder framework
        self.__encoder = EncoderDecoder.Encoder(inputDim=self.__encoderInputDim, hiddenDim=self.__encoderHiddenDim, device = device)     # encoder
        self.__decoder = EncoderDecoder.DecoderLSTMAdvanced(inputDim=2*self.__encoderHiddenDim, hiddenDim=self.__decoderOutputDim, device = device)  # decoder
        # fully connected layer
        self.fc1 = nn.Sequential(OrderedDict([
            ('batchNorm', nn.BatchNorm1d(self.__decoderOutputDim)),
            ('linear1', nn.Linear(self.__decoderOutputDim, self.__decoderOutputDim)),
            ('activate2', nn.Tanh()),
            ('linear2', nn.Linear(self.__decoderOutputDim, self.__outputDim)),
            ('activate2', nn.Tanh()),
            ('softmax', nn.Softmax())
        ])
        )

    def forward(self, batchData):
        '''
        Forward function.
        @param batchData: batch data
        @return: final output
        '''
        # encode
        codingResult, _, hiddenState = self.__encoder(batchData, resultPadding=False)

        # get hidden state of the last layer, left-to-right direction, to feed into decoder
        # [LAYERS * DIRECTION_NUM, BATCH, HIDDEN_DIM] --> [LAYERS, DIRECTION_NUM, BATCH, HIDDEN_DIM]
        layerNum = self.__encoder.getLSTMLayerNum()
        directionNum = self.__encoder.getLSTMDirectionNum()
        batchSize = len(batchData)
        hiddenState = hiddenState.view(layerNum, directionNum, batchSize, self.__encoderHiddenDim)
        # fetch hidden state of the last layer, left to right direction: [BATCH, HIDDEN_DIM]
        hiddenLastLayerLeftToRight = hiddenState[layerNum-1][0]
        # [BATCH, HIDDEN_DIM]  -->  [LAYER * DIRECTION_NUM, BATCH, HIDDEN_DIM]
        hiddenLastLayerLeftToRight = hiddenLastLayerLeftToRight.unsqueeze(0)

        # decode
        decodingResult, lengths = self.__decoder(codingResult, hiddenLastLayerLeftToRight)

        # fully connected
        # [BATCH, SEQ, DIRECTION_NUM, HIDDEN_DIM]  --> [BATCH, SEQ, HIDDEN_DIM], since DIRECTION_NUM = 1
        decodingResult = decodingResult.squeeze()
        shape = decodingResult.shape    # BATCH * SEQ * HIDDEN_DIM
        decodingResultView = decodingResult.reshape(shape[0] * shape[1], shape[2])
        fcResult = self.fc1(decodingResultView)
        output = fcResult.view(shape[0], shape[1], 1, self.__outputDim)

        #return decodingResult, lengths
        return output, lengths

class SlotFillingModel(nn.Module, Object):
    '''
    It's a model to do slot filling.
    The model is of encoder-decoder framework. Both coders are LSTM. Decoder output is passed to
    2 fully connected layers. The final output for each sequence element is a probability
    distribution over all possible labels.
    Graph:
          sequence  -->  Encoder (LSTM1, 3 layer, 2 directions) --> Decoder(LSTM2, 3 layer , 2 direction)  --> fully connected layer --> output
          note: Last hidden states of all layers of LSTM1 is fed into LSTM2
    '''
    def __init__(self, encoderInputDim, encoderHiddenDim, decoderOutputDim, outputDim, device, name="slotFillingModel"):
        # parent class init
        nn.Module.__init__(self)
        Object.__init__(self, name=name)

        # member init
        self.__encoderInputDim = encoderInputDim
        self.__encoderHiddenDim = encoderHiddenDim
        self.__decoderOutputDim = decoderOutputDim
        self.__outputDim = outputDim
        self.__device = device

        # model component
        # encoder-decoder framework
        self.__encoder = EncoderDecoder.Encoder(inputDim=self.__encoderInputDim, hiddenDim=self.__encoderHiddenDim, device = device)     # encoder
        self.__decoder = EncoderDecoder.Decoder(inputDim=2*self.__encoderHiddenDim, hiddenDim=self.__decoderOutputDim, device = device)  # decoder
        # fully connected layer
        self.fc1 = nn.Sequential(OrderedDict([
                  ('batchNorm', nn.BatchNorm1d(self.__decoderOutputDim*2)),
                  ('linear1', nn.Linear(self.__decoderOutputDim * 2, self.__decoderOutputDim *2)),
                  ('activate2', nn.Tanh()),
                  ('linear2', nn.Linear(self.__decoderOutputDim * 2, self.__outputDim)),
                  ('activate2', nn.Tanh()),
                  ('softmax', nn.Softmax())
                ])
        )

    def forward(self, batchData):
        '''
        Forward function.
        @param batchData: batch data
        @return: final output
        '''
        # encode
        codingResult, _, hiddenState = self.__encoder(batchData, resultPadding = False)

        # decode
        decodingResult, lengths = self.__decoder(codingResult, hiddenState)

        # fully connected
        decodingResultConcatTwoDirections = self.__concatOutputTwoDirectionLSTM(decodingResult)
        shape = decodingResultConcatTwoDirections.shape    # BATCH * SENTENCE * (2 * DECODE_DIM)
        decodingResultView = decodingResult.reshape(shape[0] * shape[1], shape[2])
        fcResult = self.fc1(decodingResultView)
        output = fcResult.view(shape[0], shape[1], 1, self.__outputDim)

        #return decodingResult, lengths
        return output, lengths

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
