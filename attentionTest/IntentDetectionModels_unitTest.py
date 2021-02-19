import torch
from Dataset import DataLoaderSnipsJson
from OneHotCoder import OneHotCoderSNIPS
from WordEmbedder import WordEmbedder
from IntentDetectionModels import IntentDetectionModel
from IntentDetectionModels import IntentDetectionAttentionModel
from IntentDetectionModels import IntentDetectionOnlyAttentionModel
from Utils import Utils
from IntentDetectionLoss import IntentDetectionLossCrossEntropy
import torch.nn as nn

import configparser

if __name__ == "__main__":

    # get device
    gpuFlag = True
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

    # load data and show data
    dataPath = "./data/dataSNIPrestrictYiTon.json"
    wordEmbedding = "./data/wordvec_yitongEmb.txt"


    dataLoader = DataLoaderSnipsJson(jsonFilePath=dataPath, batchSize=3, random=True)
    oneHotEncoder = OneHotCoderSNIPS(dataLoader=dataLoader)
    wordEmbedder = WordEmbedder(embeddingFilePath=wordEmbedding)
    model = IntentDetectionModel(inputDim=300, hiddenDim=32, outputDim=7, device=device)
    modelAttention = IntentDetectionAttentionModel(inputDim=300, hiddenDim=32, outputDim=7, device=device)
    modelAttentionOnly = IntentDetectionOnlyAttentionModel(inputDim=300, hiddenDim=32, outputDim=7, device=device)
    print("model parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.data)

    model = Utils.move2Device(model, device)    # move model to device
    modelAttention = Utils.move2Device(modelAttention, device)
    modelAttentionOnly = Utils.move2Device(modelAttentionOnly, device)

    criterion = IntentDetectionLossCrossEntropy(datasetLoader=dataLoader, oneHotCoder=oneHotEncoder, device=device)

    repeatNum = 1
    for repeatNo in range(repeatNum):    # repeat training process for several times
        batchNo = 0
        while True:   # iterate all batches in data loader, until it hits the end
            print ("---------------batch No. %d-------------------"%(batchNo))

            # retrieve next batch data
            queries, clsLabels, _ = dataLoader.nextBatch(train=True)

            lengthList = []
            for sentence in queries:
                lengthList.append(len(sentence))
            print(lengthList)

            # prepare for training
            # query  --> word embedding
            queriesWordEmbedding = wordEmbedder.wordEmbed(queries)

            # class labels      -->  one-hot code
            _, clsLabOneHot = oneHotEncoder.encodeClassLabel(clsLabels)

            # forward
            inputData = []
            for sentence in queriesWordEmbedding:
                sentenceEmbedding = torch.Tensor(sentence)
                sentenceEmbedding = Utils.move2Device(sentenceEmbedding, device)
                inputData.append(sentenceEmbedding)

            # test model
            print ("===================Testing   model===========================")
            output = model(inputData)
            shape = output.shape

            print ("------------output:")
            print(output)
            print ("predict result shape:")
            print(shape)

            # calcualte loss
            loss = criterion.calculateLoss(output, clsLabOneHot)

            print("loss:")
            print(loss.item())


            print ("===================Testing attention model===========================")
            output, weightList = modelAttention(inputData)
            shape = output.shape

            print ("------------output:")
            print(output)
            print ("predict result shape:")
            print(shape)

            # calcualte loss
            loss = criterion.calculateLoss(output, clsLabOneHot)

            print("loss:")
            print(loss.item())

            print("Weight:")
            for weight in weightList:
                print (weight)

            print ("===================Testing attention only model===========================")
            output, weightList = modelAttentionOnly(inputData)
            shape = output.shape

            print ("------------output:")
            print(output)
            print ("predict result shape:")
            print(shape)

            # calcualte loss
            loss = criterion.calculateLoss(output, clsLabOneHot)

            print("loss:")
            print(loss.item())

            print("Weight:")
            for weight in weightList:
                print (weight)
            batchNo = batchNo + 1
            if batchNo > 0:
                break
