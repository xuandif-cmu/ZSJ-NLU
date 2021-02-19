import torch
from Dataset import DataLoaderSnipsJson
from OneHotCoder import OneHotCoderSNIPS
from WordEmbedder import WordEmbedder
from SlotFillingModel import SlotFillingModel
from SlotFillingModel import SlotFillingModelLSTMAdvanced
from SlotFillingModel import SlotFillingModelLSTMAdvancedAttention
from SlotFillingModel import SlotFillingModelLSTMAdvancedAttentionContextOnly
from Utils import Utils
import torch.nn as nn

import configparser

def loadConfig(configFilePath):
    """
    load configuraiton file
    :param
        configFilePath: config file path
    :return:
        result:
            config file content. Content can be retrieved by:
                result[sectionName][keyName]
                result.get("sectionName","keyName")
            Type: configparser.ConfigParser()
    """
    config = configparser.ConfigParser()
    config.read(configFilePath)
    return config

if __name__ == "__main__":
    # get device
    gpuFlag = False
    if gpuFlag:
        print("[INFO] GPU flag is turned ON!")
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            print("[WARNING] Cuda device is not available, will use cpu in following procedures.")
            device = torch.device('cpu')
    else:
        print("[INFO] GPU flag is turned OFF!")
        device = torch.device('cpu')

    # load config
    configFilePath = "./config.ini"
    config = loadConfig(configFilePath)

    # load data and show data
    dataPath = config.get("DATA", "dataPath")
    wordEmbedding = config.get("DATA", "word2vecFilePath")


    dataLoader = DataLoaderSnipsJson(jsonFilePath=dataPath, batchSize=3, random=True)
    oneHotEncoder = OneHotCoderSNIPS(dataLoader=dataLoader)
    wordEmbedder = WordEmbedder(embeddingFilePath=wordEmbedding)
    # model = SlotFillingModel(encoderInputDim=300, encoderHiddenDim=32, decoderOutputDim=32, outputDim=10, device=device)
    # model = SlotFillingModelLSTMAdvanced(encoderInputDim=300, encoderHiddenDim=32, decoderOutputDim=32, outputDim=10, device=device)
    # model = SlotFillingModelLSTMAdvancedAttention(encoderInputDim=300, encoderHiddenDim=32, decoderOutputDim=32, outputDim=10, device=device)
    model = SlotFillingModelLSTMAdvancedAttentionContextOnly(encoderInputDim=300, encoderHiddenDim=32, decoderOutputDim=32, outputDim=10, device=device)

    print("model parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.data)

    # move model to device
    model = Utils.move2Device(model, device)    # move model to device

    repeatNum = 1
    for repeatNo in range(repeatNum):    # repeat training process for several times
        batchNo = 0
        while True:   # iterate all batches in data loader, until it hits the end
            print ("---------------batch No. %d-------------------"%(batchNo))

            # retrieve next batch data
            queries, clsLabels, seqLabels = dataLoader.nextBatch(train=True)

            lengthList = []
            for sentence in queries:
                lengthList.append(len(sentence))
            print(lengthList)

            # prepare for training
            # query  --> word embedding
            queriesWordEmbedding = wordEmbedder.wordEmbed(queries)

            # class labels      -->  one-hot code
            clsLabOneHot = oneHotEncoder.encodeClassLabel(clsLabels)

            # sequence labels   -->  one-hot code
            seqLabOneHot = oneHotEncoder.encodeSeqLabel(seqLabels)

            # assemble data into a list: [sentence1, sentence2, sentence3, ... ...]
            inputData = []
            for sentence in queriesWordEmbedding:
                sentenceEmbedding = torch.Tensor(sentence)
                sentenceEmbedding = Utils.move2Device(sentenceEmbedding, device)
                inputData.append(sentenceEmbedding)

            predict, seqLengths, attention = model(inputData)

            print ("------------output:")
            print ("predict result shape:")
            print(seqLengths)
            print(predict)

            print ("-------------attention:")
            print (queries)
            for i, sentence in enumerate(attention):
                print("sentence %d"%(i))
                for j, wordAttention in enumerate(sentence):
                    print ("sentence %d, word %d"%(i, j))
                    print(wordAttention)

            batchNo = batchNo + 1
            if batchNo > 0:
                break
