import torch
from Dataset import DataLoaderSnipsJson
from OneHotCoder import OneHotCoderSNIPS
from WordEmbedder import WordEmbedder
from SlotFillingModel import SlotFillingModel
from SlotFillingLoss import SlotFillingLossCrossEntropy
from SlotFillingModel import SlotFillingModelLSTMAdvancedAttention
from Evaluator import Evaluator
from Utils import Utils
import numpy as np
from TrainingLogger import TrainingLogger


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

    # training parameters
    dataPath = "./data/dataSNIPrestrictYiTon.json"
    wordEmbedding = "./data/wordvec_yitongEmb.txt"

    # data loader: manage data
    dataLoader = DataLoaderSnipsJson(jsonFilePath=dataPath, batchSize=20, random=True)

    # one-hot coder: encode labels
    oneHotEncoder = OneHotCoderSNIPS(dataLoader=dataLoader)

    # word embedder: do word embedding
    wordEmbedder = WordEmbedder(embeddingFilePath=wordEmbedding)

    # construct model
    seqLabSetSize = dataLoader.getSeqLabelSetSize()
    model = SlotFillingModelLSTMAdvancedAttention(encoderInputDim=300, encoderHiddenDim=300, decoderOutputDim=300, outputDim=seqLabSetSize, device=device)

    # loss function
    criterion = SlotFillingLossCrossEntropy(datasetLoader=dataLoader, oneHotCoder=oneHotEncoder, device=device)

    # load model from disk
    model = torch.load("./testResult/result_slotFilling_EncoderDecoder_LSTMAdvancedAttention/SlotFilling_LSTMAdvancedAttention_epoch37.pth", map_location=device)
    model.eval()

    # move model to device
    model = Utils.move2Device(model, device)    # move model to device

    batchNum = 50
    dataLoader.resetBatchIteration()    # reset data loader iteration, for training dataset
    queryListGlobal = []
    intentListGlobal = []
    seqLabListGlobal = []
    weightListGlobal = []
    with torch.no_grad():
        for batch in range(batchNum):
            # retrieve next batch data
            queries, intentLabs, seqLabels = dataLoader.nextBatch(train=True)

            if queries == None:  # data is exhausted: this epoch is done!
                break

            # prepare for training
            # query  --> word embedding
            queriesWordEmbedding = wordEmbedder.wordEmbed(queries)

            # sequence labels   -->  one-hot code
            _, seqLabOneHot = oneHotEncoder.encodeSeqLabel(seqLabels)

            # forward
            # assemble data into a list: [sentence1, sentence2, sentence3, ... ...]
            inputData = []
            for sentence in queriesWordEmbedding:
                sentenceEmbedding = torch.Tensor(sentence)
                sentenceEmbedding = Utils.move2Device(sentenceEmbedding, device)
                inputData.append(sentenceEmbedding)
            # do prediction
            predict, seqLengths, weightList = model(inputData)

            # message
            print("[INFO] Batch %4d, data count %5d."%(batch, len(queries)))

            # show data
            # predict -> predicted tokens
            predictTokenList = []   # list[BATCH * SEQUENCE * LABEL]
            for sentenceID, sentence in enumerate(predict):
                sentenceTokenList = []
                for word in sentence[0:seqLengths[sentenceID]]:
                    _, token = word[0].max(dim = 0)
                    sentenceTokenList.append(token.item())
                predictTokenList.append(sentenceTokenList)
            predictLabList = []
            for sentence in predictTokenList:
                labList = []
                for token in sentence:
                    label = oneHotEncoder.tocken2seqLab(token)
                    labList.append(label)
                predictLabList.append(labList)

            # collect data
            queryListGlobal = queryListGlobal + queries
            intentListGlobal = intentListGlobal + intentLabs
            seqLabListGlobal = seqLabListGlobal + seqLabels
            weightListGlobal = weightListGlobal + weightList

            a = 123

    print ("[INFO] %d data is collected!"%(len(queryListGlobal)))

    # sanity check
    if not (len(queryListGlobal) == len(intentListGlobal) and
            len(intentListGlobal) == len(seqLabListGlobal) and
            len(seqLabListGlobal) == len(weightListGlobal)):
        print ("[ERROR] Sanity check fail. List lengths are inconsistent among global lists.")

    # do analysis:
    # Does attention distribution cover slot?
    intentLabSet = set(intentListGlobal)
    statistics = {}
    for intent in intentLabSet:    # for each intent
        statistics[intent] = {"count":0, "firstHalf":0, "secondHalf":0, "topValid":0, "heavySet":set(), "heavySetCount":{}}    # dataCount, first Half Covered label, second half covered label
        for i in range(len(queryListGlobal)):
            if intentListGlobal[i] == intent:   # for a sentence belonging to current intent
                seqLabs = seqLabListGlobal[i]
                weights = weightListGlobal[i]
                distribution = np.mean(np.array(weights), axis=0)

                length = len(queryListGlobal[i])

                firstHalf = 0
                secondHalf = 0
                for pos, (weight, lab, query) in enumerate(sorted(zip(distribution, seqLabs, queryListGlobal[i]), reverse=True)):
                    if pos < (length/2):   # first half
                        if not lab == 'o':
                            firstHalf = firstHalf + 1
                    else:                  # second half
                        if not lab == 'o':
                            secondHalf = secondHalf + 1

                    if pos == 0:
                        statistics[intent]["heavySet"].add(query)
                        if query in statistics[intent]["heavySetCount"]:
                            statistics[intent]["heavySetCount"][query] = statistics[intent]["heavySetCount"][query] + 1
                        else:
                            statistics[intent]["heavySetCount"][query] = 1

                    if (pos == 0) and (not lab == 'o'):
                        statistics[intent]["topValid"] =statistics[intent]["topValid"] + 1

                statistics[intent]["count"] = statistics[intent]["count"] + 1
                statistics[intent]["firstHalf"] = statistics[intent]["firstHalf"] + firstHalf
                statistics[intent]["secondHalf"] = statistics[intent]["secondHalf"] + secondHalf
                statistics[intent]["ratio1/2"]  = statistics

    # sanity check
    print ("%d intents are analyzed"%(len(statistics)))
    dataCount = 0
    for a in statistics:
        dataCount = dataCount + statistics[a]["count"]
    print ("%d data are analyzed"%(dataCount))
    for a in statistics:
        print ("------intent %s: "%(a))
        print ("data count:                           %d"%(statistics[a]["count"]))
        print ("Accumulated first half hit labels  :  %d"%(statistics[a]["firstHalf"]))
        print ("Accumulated second half hit labels:  %d"%(statistics[a]["secondHalf"]))
        print ("Top word is valid slot:            %d"%(statistics[a]["topValid"]))
        print ("Heavy set:                             ", statistics[a]["heavySet"])
        print ("Heavy set count:",                        statistics[a]["heavySetCount"])
