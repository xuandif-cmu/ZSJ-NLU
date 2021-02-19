from commonVar import STATUS_FAIL
from commonVar import STATUS_WARNING
import torch.nn as nn
from Utils import Utils
import torch

class IntentDetectionLossCrossEntropy():
    '''
        cross entropy loss
    '''
    def __init__(self, datasetLoader, oneHotCoder, device):
        self.__weight = None    # weight on different labels
        self.__device = device  # device
        self.__datasetLoader = datasetLoader    # data set loader
        self.__oneHotCoder = oneHotCoder        # one hot coder
        self.__setWeight(datasetLoader=self.__datasetLoader, oneHotCoder=self.__oneHotCoder) # fill up self.__weight
        self.__criterion = nn.CrossEntropyLoss(weight=self.__weight)   # criterion

    def __setWeight(self, datasetLoader, oneHotCoder):
        '''
        set intent label weight
        @param datasetLoader: dataset loader, to load data information
        @param oneHotCoder: one hot coder, to tokenize label
        @return: exit status
        @attention:
            self.__weight is affected
        '''
        # get label count
        label2CountDict = datasetLoader.getClassLabelsCountDict(train=True)

        # sanity check
        if not len(label2CountDict) == oneHotCoder.getclsLabNum():
            print ("[ERROR] Intent detection loss. Fill to set label weight, due to inconsistent label number between dataset and oneHotCoder.")
            return STATUS_FAIL

        # fill up weight vector
        # get max count
        maxCount = -1
        for lab in label2CountDict:
            if label2CountDict[lab] > maxCount:
                maxCount = label2CountDict[lab]
        # calculate weight:  weight of label A is maxCount/countOfLabelA
        labNum = len(label2CountDict)
        self.__weight = torch.zeros(labNum)    # initialize weight to 0
        self.__weight.requires_grad = False    # turn gradient off
        for lab in label2CountDict:
            self.__weight[oneHotCoder.clsLab2Token(lab)] = maxCount/label2CountDict[lab]

        # move data to device
        self.__weight = Utils.move2Device(self.__weight, self.__device)


    def calculateLoss(self, pred, target, useWeight = True):
        '''
        @param pred:    Predicted result. It's supposed to be output of intent detection model.
              @type:    tensor[BATCH * LABEL]
        @param target:  Target, it's supposed to be intent labels 1-hot code.
                @type:  list[BATCH * LABEL]
        @return: exit_status, tensor
          @type: tensor[1]
        '''

        # sanity check
        # is target size consistent with that of prediction?
        if not pred.shape[0] == len(target):
            print("[ERROR] Intent detection loss calculation. Inconsistent tensor size is found:")
            print("-------- Predicted result size: ")
            print(pred.shape[0])
            print("-------- Target size: ")
            print(len(target))
            return STATUS_FAIL, None

        # is pred empty?
        if pred.nelement() == 0:
            print("[WARNING] Intent detection loss calculation. Predicted tensor is empty. Will return zero loss. This loss value is supposed not to affect model parameters in BP process.")
            tempInput = torch.tensor([[1.]])
            tempTarget = torch.tesnor([0], dtype=torch.long)
            loss = self.__criterion(tempInput, tempTarget)
            return STATUS_WARNING, loss

        # transform target
        # list[oneHotcode]   -->  list[token]
        targetTokenList = self.__oneHotCoder.oneHot2Token(target)
        targetTokenListTensor = torch.tensor(targetTokenList, dtype=torch.long)

        # move target sequence to current device
        targetTokenListTensor = Utils.move2Device(targetTokenListTensor, self.__device)

        # calculate loss
        if useWeight:
            return self.__criterion(pred, targetTokenListTensor)
        else:
            return self.__criterion(pred, targetTokenListTensor)

