import torch
from Dataset import DataLoaderSnipsJson
from OneHotCoder import OneHotCoderSNIPS
from WordEmbedder import WordEmbedder
from IntentDetectionModels import IntentDetectionModel
from Evaluator import Evaluator
from Utils import Utils
from TrainingLogger import TrainingLogger
from IntentDetectionLoss import IntentDetectionLossCrossEntropy


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
    clsLabSetSize = dataLoader.getClsLabSetSize()
    # model = IntentDetectionModel(inputDim=300, hiddenDim=300, outputDim=clsLabSetSize, device=device)

    # loss function
    criterion = IntentDetectionLossCrossEntropy(datasetLoader=dataLoader, oneHotCoder=oneHotEncoder, device=device)

    # model evaluator
    modelEvaluator = Evaluator(dataLoader=dataLoader, criterionSlotFilling=None, criterionIntentDetection=criterion, wordEmbedder=wordEmbedder, oneHotCoder=oneHotEncoder, device=device)

    # load model from disk
    #model = torch.load("./testResult/result_intentDetection_LstmFullyconnected/FinalModel_IntentDetection.pth")
    model = torch.load("./testResult/result_intentDetection_attentionOnly/IntentDetection_Attention_epoch18.pth")
    model.eval()

    # move model to device
    model = Utils.move2Device(model, device)

    # model logger
    logger = TrainingLogger(path="./", evaluator=modelEvaluator, testBatchNum=150)

    # perform test on several batches, display result
    batchNum = 5
    dataLoader.resetBatchIteration()  # reset data loader iteration, for training dataset
    count = 0    # sentence that two ends >= 50% weight
    with torch.no_grad():
        for batch in range(batchNum):
            queries, clsLabels, seqLabsList = dataLoader.nextBatch(train=False)

            if queries == None:  # data is exhausted: this epoch is done!
                break

            # prepare for training
            # query  --> word embedding
            queriesWordEmbedding = wordEmbedder.wordEmbed(queries)
            # cls labels --> one-hot code
            _, clsLabOneHot = oneHotEncoder.encodeClassLabel(clsLabels)

            # forward
            # assemble data into a list: [sentence1, sentence2, sentence3, ... ...]
            inputData = []
            for sentence in queriesWordEmbedding:
                sentenceEmbedding = torch.Tensor(sentence)
                sentenceEmbedding = Utils.move2Device(sentenceEmbedding, device)
                inputData.append(sentenceEmbedding)
            # do prediction
            predict, weightList = model(inputData)

            # compute loss
            loss = criterion.calculateLoss(predict, clsLabOneHot)

            # message
            print("[INFO] =======================batch %3d============================"%(batch))
            print("[INFO] Batch %4d, data count %5d, averaged loss: %f "%(batch, len(queries), loss))

            logger.logIntentDetectionModelStatusBatch(model, -1, 1, batchStep=1)

            # show data
            predictTokenList = []
            for sentence in predict:
                _, token = sentence.max(dim=0)
                predictTokenList.append(token.item())

            for i in range(len(queries)):
                query = queries[i]
                intent = clsLabels[i]
                predictedIntent = predictTokenList[i]
                weight = weightList[i]
                seqLabs = seqLabsList[i]

                print("------------------------")
                #print("[INFO] query:   %s"%(query))
                print("[INFO] query:   ", ["%20s"%(word) for word in query])
                print("[INFO] labels:  ", ["%20s"%(lab) for lab in seqLabs])
                print("[INFO] truth:   %s"%(intent))
                print("[INFO] predict: %s"%(oneHotEncoder.token2clsLab(predictedIntent)))
                print("[INFO] weight:    ", ["%20f"%(number) for number in weight])

                length = len(query)
                if (weight[0] + weight[length - 1]) >= 0.5:
                    count = count + 1

        print ("In total, %d data has two ends weight!"%(count))