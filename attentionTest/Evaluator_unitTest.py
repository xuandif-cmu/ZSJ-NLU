import torch
from dataset import DataLoaderSnipsJson
from OneHotCoder import OneHotCoderSNIPS
from WordEmbedder import WordEmbedder
from SlotFillingModel import SlotFillingModel
import torch.nn as nn
from SlotFillingLoss import SlotFillingLossCrossEntropy
from Evaluator import Evaluator
from commonVar import STATUS_OK
from commonVar import STATUS_WARNING
from commonVar import STATUS_FAIL

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

    dataLoader = DataLoaderSnipsJson(jsonFilePath=dataPath, batchSize=400, random=True)
    seqLabSetSize = dataLoader.getSeqLabelSetSize()
    trainDataCount = dataLoader.getTrainDataCount()

    oneHotEncoder = OneHotCoderSNIPS(dataLoader=dataLoader)
    wordEmbedder = WordEmbedder(embeddingFilePath=wordEmbedding)

    device = torch.device('cpu')
    model = SlotFillingModel(encoderInputDim=300, encoderHiddenDim=32,  decoderOutputDim = 32,outputDim=seqLabSetSize, device=device)
    criterion = SlotFillingLossCrossEntropy(datasetLoader=dataLoader, oneHotCoder=oneHotEncoder, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    modelEvaluator = Evaluator(dataLoader=dataLoader, wordEmbedder=wordEmbedder, oneHotCoder=oneHotEncoder, device=device)

    epochNum = 5
    epoch = 0

    rt, F1Score, loss, report = modelEvaluator.evaluateSlotFillingModel(model=model)

    if not rt == STATUS_OK:
        print ("[ERROR] Fail to evaluate slot filling model.")

    print ("Evaluation result, F1 score: %f"%(F1Score))
    print(report)
