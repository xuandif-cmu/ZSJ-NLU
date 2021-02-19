from commonVar import STATUS_FAIL
from commonVar import STATUS_WARNING
import torch.nn as nn
from Utils import Utils
import torch

class SlotFillingLossCrossEntropy():
    '''
        cross entropy loss
    '''
    def __init__(self, datasetLoader, oneHotCoder, device):
        self.__weight = None            # loss weight on labels
        self.__device = device          # device
        self.setWeight(datasetLoader=datasetLoader, oneHotCoder=oneHotCoder)    # fill up self.__weight
        self.__criterion = nn.CrossEntropyLoss(weight=self.__weight)  # criterion

    def setWeight(self, datasetLoader, oneHotCoder):
        '''
        Set slot filling label weight
        @param datasetLoader: dataset loader, to load data information
        @param oneHotCoder: one hot coder, to tokenize label
        @return: exit status
        @attention:
            self.__weight is affected
        '''
        # get label count
        label2CountDict = datasetLoader.getSeqLabelsCountDict(train=True)

        # sanity check
        if not len(label2CountDict) == oneHotCoder.getSequenceLabelNum():
            print ("[ERROR] slot filling loss. Fill to set label weight, due to inconsistent label number between dataset and oneHotCoder.")
            return STATUS_FAIL, None

        # fill up weight vector
        # get max count
        maxCount = -1
        for lab in label2CountDict:
            if label2CountDict[lab] > maxCount:
                maxCount = label2CountDict[lab]
        # calculate weight:  weight of label A is maxCount/countOfLabelA
        labNum = len(label2CountDict)
        self.__weight = torch.zeros(labNum)
        self.__weight.requires_grad = False    # turn gradient off
        for lab in label2CountDict:
            self.__weight[oneHotCoder.seqLabel2Token(lab)] = maxCount/label2CountDict[lab]

        # move data to device
        self.__weight = Utils.move2Device(self.__weight, self.__device)


    def calculateLoss(self, pred, seqLenList, target, useWeight = True):
        '''
        @param pred:    Predicted result. It's supposed to be output of slot filling model.
              @type:    tensor[BATCH * SEQ * 1 * LABEL]
        @param seqLenList: sequence length list, storing sequence length. It's supposed to be
                            output of slot filling model, together with prediction result. It's
                            needed since in prediction result, short sequence is padded with
                            meaningless number, in places aligned with longest sequence, in current
                            batch.
                    @type: list[BATCH]
        @param target:  Target, it's supposed to be slot filling label 1-hot code.
                @type:  list[BATCH * SEQ * LABEL]
        @return: exit_status, tensor
          @type: tensor[1]
        '''

        # sanity check
        # is target length consistent with prediction length?
        targetLengthList = []
        for sentence in target:
            targetLengthList.append(len(sentence))
        targetLengthListTensor = torch.tensor(targetLengthList, dtype=torch.long)
        if not seqLenList.equal(targetLengthListTensor):
            print("[ERROR] slotFillingLossFunc. Inconsistent sequence length is found:")
            print("-------- Input sequence length list: ")
            print(seqLenList)
            print("-------- target list lengths: ")
            print(targetLengthList)
            return STATUS_FAIL, None

        # is pred empty?
        if len(seqLenList) == 0:
            print("[WARNING] slotFillingLossFunc. Empty input. Will return zero loss. This loss value is supposed not to affect model parameters in BP process.")
            tempInput = torch.tensor([[1.]])
            tempTarget = torch.tesnor([0], dtype=torch.long)
            loss = self.__criterion(tempInput, tempTarget)
            return STATUS_WARNING, loss

        # transform pred data shape:
        # tensor[BATCH * SEQ * 1 * LABEL]   -->  tensor[SEQ * LABEL]
        # sequences in the batch are concatenated sequentially
        predSeq = Utils.serializeBatchData(pred, seqLenList)

        # transform target
        # list[BATCH]   -->  list[SEQ]
        targetSeq = Utils.serializeTarget(target)

        # move target sequence to current device
        targetSeq = Utils.move2Device(targetSeq, self.__device)

        # calculate loss
        if useWeight:
            return self.__criterion(predSeq, targetSeq)
        else:
            return self.__criterion(predSeq, targetSeq)

