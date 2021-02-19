import torch
from dataset import DataLoaderSnipsJson
from OneHotCoder import OneHotCoderSNIPS
from WordEmbedder import WordEmbedder
from SlotFillingModel import SlotFillingModel
import torch.nn as nn

import configparser

import SlotFillingLoss

loss = SlotFillingLoss.SlotFillingLossCrossEntropy()

if __name__ == "__main__":
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

    # predict data
    label0 = [0.5, 0.1,  0.1, 0.1]
    label1 = [0.1, 0.5,  0.1, 0.2]
    label2 = [0.1, 0.15, 0.9, 0.2]
    label3 = [0.1, 0.1,  0.1, 1.4]
    labelP = [0.9, 0.91, 0.92,0.93]   # Padding label

    sequence1 = [label0, label2, label3, label0, label1]
    sequence2 = [label3, label2, labelP, labelP, labelP]
    sequence3 = [label2, label2, label1, label0, labelP]

    lengthList = [5, 2, 4]
    lengthListTensor = torch.tensor(lengthList, dtype = torch.long)

    batch = torch.tensor([sequence1, sequence2, sequence3])
    batch = torch.unsqueeze(batch, 2)

    # target
    label0Hot = [1, 0, 0, 0]
    label1Hot = [0, 1, 0, 0]
    label2Hot = [0, 0, 1, 0]
    label3Hot = [0, 0, 0, 1]

    sequence1_target = [label0Hot, label2Hot, label3Hot, label0Hot, label1Hot]
    sequence2_target = [label3Hot, label2Hot]
    sequence3_target = [label2Hot, label2Hot, label1Hot, label0Hot]

    target = [sequence1_target, sequence2_target, sequence3_target]

    l = loss.calculateLoss(batch, lengthListTensor, target)

    print(l)
