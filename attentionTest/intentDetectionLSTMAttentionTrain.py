import torch
from Dataset import DataLoaderSnipsJson
from OneHotCoder import OneHotCoderSNIPS
from WordEmbedder import WordEmbedder
from IntentDetectionModels import IntentDetectionAttentionModel
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
    dataLoader = DataLoaderSnipsJson(jsonFilePath=dataPath, batchSize=200, random=True)

    # one-hot coder: encode labels
    oneHotEncoder = OneHotCoderSNIPS(dataLoader=dataLoader)

    # word embedder: do word embedding
    wordEmbedder = WordEmbedder(embeddingFilePath=wordEmbedding)

    # construct model
    clsLabSetSize = dataLoader.getClsLabSetSize()
    model = IntentDetectionAttentionModel(inputDim=300, hiddenDim=300, outputDim=clsLabSetSize, device=device)

    # loss function
    criterion = IntentDetectionLossCrossEntropy(datasetLoader=dataLoader, oneHotCoder=oneHotEncoder, device=device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    # model evaluator
    modelEvaluator = Evaluator(dataLoader=dataLoader, criterionSlotFilling=None, criterionIntentDetection=criterion, wordEmbedder=wordEmbedder, oneHotCoder=oneHotEncoder, device=device)

    # training process logger
    modelSavePath = './trainedModel/'
    logger = TrainingLogger(path=modelSavePath, evaluator=modelEvaluator, testBatchNum=5)

    # start training
    model = Utils.move2Device(model, device)    # move model to device
    epochNum = 20
    epoch = 0
    for epoch in range(epochNum):  # for each epoch

        batch = 0
        dataLoader.resetBatchIteration()    # reset data loader iteration, for training dataset
        while True:  # iterate all batches in data loader, until it hits the end
            # retrieve next batch data
            queries, clsLabels, _ = dataLoader.nextBatch(train=True)

            if queries == None:  # data is exhausted: this epoch is done!
                break

            # prepare for training
            # query  --> word embedding
            queriesWordEmbedding = wordEmbedder.wordEmbed(queries)
            # cls labels --> one-hot code
            _, clsLabOneHot = oneHotEncoder.encodeClassLabel(clsLabels)

            # clear gradients
            optimizer.zero_grad()

            # forward
            # assemble data into a list: [sentence1, sentence2, sentence3, ... ...]
            inputData = []
            for sentence in queriesWordEmbedding:
                sentenceEmbedding = torch.Tensor(sentence)
                sentenceEmbedding = Utils.move2Device(sentenceEmbedding, device)
                inputData.append(sentenceEmbedding)
            # do prediction
            predict, weight = model(inputData)

            # compute loss
            loss = criterion.calculateLoss(predict, clsLabOneHot)

            # backward
            loss.backward()

            # optimize
            optimizer.step()

            # message
            print ("[INFO] Epoch %d, batch %4d, data count %5d, averaged training loss: %f "%(epoch, batch, len(queries), loss))

            # evaluate and log
            #logger.logIntentDetectionModelStatusBatch(model, epoch, batch, batchStep=1)

            # Increase batch
            batch = batch + 1

        # evaluate and log
        logger.logIntentDetectionModelStatusEpoch(model, epoch, epochStep=1, detailedReport=True)

        # save model
        if epoch % 1 == 0:
            logger.saveModel(model, "IntentDetection_Attention_epoch%d"%(epoch))

    print ("[INFO] Training is done! Epoch = %d."%(epochNum))
    logger.saveModel(model, "IntentDetection_Attention_epoch_final_epoch%d"%(epoch))
