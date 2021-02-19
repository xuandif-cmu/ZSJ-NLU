import torch
from Dataset import DataLoaderSnipsJson
from OneHotCoder import OneHotCoderSNIPS
from WordEmbedder import WordEmbedder
from SlotFillingModel import SlotFillingModel
from SlotFillingLoss import SlotFillingLossCrossEntropy
from SlotFillingModel import SlotFillingModelLSTMAdvancedAttention
from Evaluator import Evaluator
from Utils import Utils
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

    batchNum = 5
    dataLoader.resetBatchIteration()    # reset data loader iteration, for training dataset
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

            # compute loss
            loss = criterion.calculateLoss(predict, seqLengths, seqLabOneHot)

            # message
            print("[INFO] =======================batch %3d============================"%(batch))
            print("[INFO] Batch %4d, data count %5d, averaged loss: %f "%(batch, len(queries), loss))

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
            # display
            for i in range(len(queries)):
                query = queries[i]
                seqLabs = seqLabels[i]
                predictSeqLabs = predictLabList[i]

                print("------------------------")
                print("[INFO] query:    ", ["%20s"%(word) for word in query])
                print("[INFO] truth:    ", ["%20s"%(lab) for lab in seqLabs])
                print("[INFO] predict:  ", ["%20s"%(lab) for lab in predictSeqLabs])
                # print attention matrix
                for wordPos in range(seqLengths[i]):
                    print("[INFO] attention:", ["%20f"%(num) for num in (weightList[i][wordPos])])
                # intent
                print("[INFO] intent:    ", "%20s"%(intentLabs[i]))

    print ("[INFO] Test is done!")