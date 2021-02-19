import torch
from Dataset import DataLoaderSnipsJson
from OneHotCoder import OneHotCoderSNIPS
from WordEmbedder import WordEmbedder
from SlotFillingModel import SlotFillingModel
from SlotFillingModel import SlotFillingModelLSTMAdvanced
from SlotFillingLoss import SlotFillingLossCrossEntropy
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
    dataLoader = DataLoaderSnipsJson(jsonFilePath=dataPath, batchSize=200, random=True)

    # one-hot coder: encode labels
    oneHotEncoder = OneHotCoderSNIPS(dataLoader=dataLoader)

    # word embedder: do word embedding
    wordEmbedder = WordEmbedder(embeddingFilePath=wordEmbedding)

    # construct model
    seqLabSetSize = dataLoader.getSeqLabelSetSize()
    model = SlotFillingModelLSTMAdvanced(encoderInputDim=300, encoderHiddenDim=300, decoderOutputDim=300, outputDim=seqLabSetSize, device=device)

    # loss function
    criterion = SlotFillingLossCrossEntropy(datasetLoader=dataLoader, oneHotCoder=oneHotEncoder, device=device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # model evaluator
    modelEvaluator = Evaluator(dataLoader=dataLoader, criterionSlotFilling=criterion, criterionIntentDetection=None, wordEmbedder=wordEmbedder, oneHotCoder=oneHotEncoder, device=device)

    # training process logger
    modelSavePath = './trainedModel/'
    logger = TrainingLogger(path=modelSavePath, evaluator=modelEvaluator, testBatchNum=5)

    # start training
    model = Utils.move2Device(model, device)    # move model to device
    epochNum = 100
    epoch = 0
    for epoch in range(epochNum):  # for each epoch
        batch = 0
        dataLoader.resetBatchIteration()    # reset data loader iteration, for training dataset
        while True:  # iterate all batches in data loader, until it hits the end
            # retrieve next batch data
            queries, _, seqLabels = dataLoader.nextBatch(train=True)

            if queries == None:  # data is exhausted: this epoch is done!
                break

            # prepare for training
            # query  --> word embedding
            queriesWordEmbedding = wordEmbedder.wordEmbed(queries)

            # sequence labels   -->  one-hot code
            _, seqLabOneHot = oneHotEncoder.encodeSeqLabel(seqLabels)

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
            predict, seqLengths = model(inputData)

            # compute loss
            loss = criterion.calculateLoss(predict, seqLengths, seqLabOneHot)

            # backward
            loss.backward()

            # optimize
            optimizer.step()

            # message
            print ("[INFO] Epoch %d, batch %4d, data count %5d, averaged training loss: %f "%(epoch, batch, len(queries), loss))

            # evaluate and log
            #logger.logSlotFillingModelStatusBatch(model, epoch, batch, batchStep=1)

            # Increase batch
            batch = batch + 1

            if batch > 2:
                break

        # evaluate and log
        logger.logSlotFillingModelStatusEpoch(model, epoch, epochStep=1)
        #logger.saveSlotFillingModelEpoch(model, epoch, epochStep=30)

    print ("[INFO] Training is done! Epoch = %d."%(epochNum))
    logger.saveModel(model, "FinalModel_SlotFilling_LSTMAdvanced")
