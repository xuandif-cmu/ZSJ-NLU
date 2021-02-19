import torch
from dataset import DataLoaderSnipsJson
from OneHotCoder import OneHotCoderSNIPS
from WordEmbedder import WordEmbedder
from EncoderDecoder import Encoder
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
    # load config
    configFilePath = "./config.ini"
    config = loadConfig(configFilePath)

    # load data and show data
    dataPath = config.get("DATA", "dataPath")
    wordEmbedding = config.get("DATA", "word2vecFilePath")


    dataLoader = DataLoaderSnipsJson(jsonFilePath=dataPath, batchSize=3, random=True)
    oneHotEncoder = OneHotCoderSNIPS(dataLoader=dataLoader)
    wordEmbedder = WordEmbedder(embeddingFilePath=wordEmbedding)

    encoder = Encoder(inputDim=300, hiddenDim=6)

    repeatNum = 1
    for repeatNo in range(repeatNum):    # repeat training process for several times
        batchNo = 0
        while True:   # iterate all batches in data loader, until it hits the end
            print ("---------------batch No. %d-------------------"%(batchNo))

            # retrieve next batch data
            queries, clsLabels, seqLabels = dataLoader.nextBatch(train=True)

            # prepare for training
            # query  --> word embedding
            queriesWordEmbedding = wordEmbedder.wordEmbed(queries)

            # class labels      -->  one-hot code
            clsLabOneHot = oneHotEncoder.encodeClassLabel(clsLabels)

            # sequence labels   -->  one-hot code
            seqLabOneHot = oneHotEncoder.encodeSeqLabel(seqLabels)

            # forward
            queriesWordEmbeddingTensorList = []
            for sentence in queriesWordEmbedding:
                queriesWordEmbeddingTensorList.append(torch.Tensor(sentence))
            output, lenVec, hiddenState = encoder(queriesWordEmbeddingTensorList)

            print("----------------output------------")
            print ("length vector:")
            print(lenVec)
            print("Output for each sentence")
            for i in output:
                print(i)
            print("Hidden State")
            print (hiddenState)

            print("-------------encoding, unpadding ---------")
            output, lenVec, hiddenState = encoder(queriesWordEmbeddingTensorList, resultPadding=False)
            print("-------------unpadded encoding result ---------")
            print (output)
            print("Hidden State")
            print (hiddenState)

            batchNo = batchNo + 1
            if batchNo > 0:
                break
