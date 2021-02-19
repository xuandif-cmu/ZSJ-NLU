import math
import torch

from commonVar import FLOAT_ERROR
from commonVar import STATUS_OK
from commonVar import STATUS_FAIL

class Utils:
    '''
    Utils class, hosting a wide range of tools.
    '''
    @staticmethod
    def isEqualFloat(float1, float2):
        return (abs(float1 - float2)) <= FLOAT_ERROR


    @staticmethod
    def splitIntByRatio(integer, ratioList):
        '''
        Split integer by given ratio. For example:
            integer = 10, ratio = [0.3,0.1,0.2, 0.4]
            output = [3,1,2,4]
        If integer cannot be perfectly divided, rounding will be performed so that:
                sum(output) == integer
        @param integer: input integer
        @param ratioList: a list of ratio, for example [0.3, 0.1, 0.1, 0.5]
        @return:
                (exit status, result):
                result is a list of integers
        '''
        # sanity check
        if not Utils.isEqualFloat(sum(ratioList), 1):
            print("[ERROR] splitIntByRatio, ratio sum is not 1. Ratio list: !")
            print(ratioList)
            return STATUS_FAIL, _

        # split integer
        # it should be done in a recursive way: Cut part of the whole, then do the same thing to
        # remaining part. However, it has limit from call stack depth.
        # Therefore, let's make it in a loop
        result = []
        partNum = len(ratioList)
        left = integer
        leftPortion = 1.0
        for i in range(partNum):
            if i == (partNum-1):    # if it's last portion, say, 0.5 in [0.3, 0.1, 0.1, 0.5]
                                    # just append left amount
                result.append(left)
                break
            else:                   # if it's not last portion, cut down and do the same thing
                                    # to remaining part
                cutoff = math.floor(left * (ratioList[i]/leftPortion))
                result.append(cutoff)

                left = left - cutoff
                leftPortion = leftPortion - ratioList[i]

        # sanity check
        if not (sum(result) == integer):
            print("[ERROR] splitIntByRatio, fail to split integer.")
            print("Split result sum is not equal to original integer.")
            print("Original integer is %d, but split result is:"%(integer))
            print (result)
            return STATUS_FAIL, None

        # return result
        return STATUS_OK, result

    @staticmethod
    def printModelParams(model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    @staticmethod
    def serializeBatchData(inputData, seqLenList):
        '''
        Sequentialize batch data, in other words, all data in given batch are concatenated one by one.
        @param inputData: input data
                   @type: tensor[BATCH * SEQ * 1 * LABEL]
        @param seqLenList: sequence length list
                    @type: list[BATCH]
        @return:
            serializedData: serialized result
                         @type: tensor[SEQ * LABEL]

        '''

        # learn about input data shape
        shape = inputData.shape
        labelNum = shape[3]

        # init result
        for index, sentence in enumerate(inputData):
            if index == 0:  # for the first sentence, use it to initialize result
                result = inputData[0][0:seqLenList[0]].reshape(seqLenList[0], labelNum)
            else:  # for second and later sentences, cut it by corresponding length, reshape
                # and concatenate it onto result
                result = torch.cat((result, sentence[0:seqLenList[index]].reshape(seqLenList[index], labelNum)), dim=0)

        return result

    @staticmethod
    def serializeTarget(target):
        '''
        transform target from list[BATCH] to list[SEQ]
        @param target: a list of target
                @type: list[BATCH * SEQ * LABEL]
        @return:
                serialized target
                @type: tensor[SEQ]
        '''
        targetSerializedList = []
        for sentence in target:
            for word in sentence:
                targetSerializedList.append(word.index(1))

        return torch.tensor(targetSerializedList, dtype=torch.long)

    @staticmethod
    def move2Device(data, device):
        '''
        @param data:     Tensor, or model
        @param device:   cpu device or gpu device
        @return:         Data, after moved to device
        '''
        return data.to(device)

    @staticmethod
    def probabilityDistributionTensor2Index():
        '''
        Translate a tensor of probability distribution to a list a index, indexing the largest index.
        @param oneHotList: tensor[SEQ * LABEL]
        @return: list[SEQ]
        '''


    @staticmethod
    def compModel(param1, param2):
        for key_item1, key_item2 in zip(param1.items(), param2.items()):
            if torch.equal(key_item1[1], key_item2[1]):
                pass
            else:
                print ("Diff model: %s"%(key_item1[0]))