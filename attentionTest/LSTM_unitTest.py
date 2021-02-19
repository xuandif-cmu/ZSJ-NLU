import torch
from LSTM import LSTM
from LSTM import LSTMAdvanced
from LSTM import LSTMAdvancedAttention

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


    biLSTM = LSTM(embDim=4, hiddenDim=5, device=device)

    lstm = LSTM(embDim=4, hiddenDim=5, bidirectional=False, device=device)

    lstmAdvanced = LSTMAdvanced(embDim=4, hiddenDim=5, device=device)

    lstmAdvancedAttention = LSTMAdvancedAttention(embDim=4, hiddenDim=5, device=device)

    repeatNum = 1
    for repeatNo in range(repeatNum):    # repeat training process for several times
        batchNo = 0
        while True:   # iterate all batches in data loader, until it hits the end
            print ("---------------batch No. %d-------------------"%(batchNo))

            # retrieve next batch data
            #queries, clsLabels, seqLabels = dataLoader.nextBatch(train=True)

            # prepare for training
            # query  --> word embedding
            #queriesWordEmbedding = wordEmbedder.wordEmbed(queries)

            # class labels      -->  one-hot code
            #clsLabOneHot = oneHotEncoder.encodeClassLabel(clsLabels)

            # sequence labels   -->  one-hot code
            #seqLabOneHot = oneHotEncoder.encodeSeqLabel(seqLabels)

            # forward
            word1 = [1.1, 1.2, 1.3, 1.4]
            word2 = [2.1, 2.2, 2.3, 2.4]
            word3 = [3.1, 3.2, 3.3, 3.4]
            word4 = [4.1, 4.2, 4.3, 4.4]
            word5 = [5.1, 5.2, 5.3, 5.4]
            sentence1 = [word1, word2, word3, word1, word2]
            sentence2 = [word1]
            sentence3 = [word4, word5]
            batch = [sentence1, sentence2, sentence3]
            batchTensorList = [torch.Tensor(sentence1),
                               torch.Tensor(sentence2),
                               torch.Tensor(sentence3)]

            print ("------Input------")
            print(batchTensorList)
            print ("=========BiLSTM test================")
            print ("=========BiLSTM test=======Init hidden state=========")
            # init hidden state
            hiddenStateInit = torch.rand(3 * 2, 3, 5)
            (output, lenVec), hiddenState = biLSTM(batchTensorList, hiddenStateInit)
            print ("length vector:")
            print(lenVec)
            print("----------------output------------")
            print("Output for each sentence")
            for i in output:
                print (i)
            print("Hidden state for each sentence")
            for i in hiddenState:
                print (i)

            print ("=========BiLSTM test=======Default hidden state=========")
            # init hidden state
            (output, lenVec), hiddenState = biLSTM(batchTensorList)
            print ("length vector:")
            print(lenVec)
            print("----------------output------------")
            print("Output for each sentence")
            for i in output:
                print (i)
            print("Hidden state for each sentence")
            for i in hiddenState:
                print (i)

            print ("=========single layer LSTM test================")
            (output, lenVec), hiddenState = lstm(batchTensorList)
            print ("length vector:")
            print(lenVec)

            print("----------------output------------")
            print("Output for each sentence")
            for i in output:
                print (i)
            print("Hidden state for each sentence")
            for i in hiddenState:
                print(i)


            print ("=========LSTMAdvanced ================")
            (output, seqLengths), hiddenState = lstmAdvanced(batchTensorList)

            print("----------------output------------")
            print ("length vector:")
            print(seqLengths)
            print("Output for each sentence")
            for i in output:
                print (i)
            print("Hidden state for each sentence")
            for i in hiddenState:
                print(i)


            print ("=========LSTMAdvanced Attention================")
            (output, seqLengths), hiddenState, attention = lstmAdvancedAttention(batchTensorList)

            print("----------------output------------")
            print ("length vector:")
            print(seqLengths)
            print("Output for each sentence")
            for i in output:
                print (i)
            print("Hidden state for each sentence")
            for i in hiddenState:
                print(i)
            print("Attention:")
            for i in attention:
                print(i)

            batchNo = batchNo + 1
            if batchNo > 0:
                break
