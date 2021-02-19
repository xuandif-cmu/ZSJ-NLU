from commonVar import STATUS_FAIL
from commonVar import STATUS_OK
import torch
from Utils import Utils
from sklearn import metrics
import copy

from Object import Object

class Evaluator(Object):
    '''
    Class to evaluate model.
    '''
    def __init__(self, dataLoader, criterionSlotFilling, criterionIntentDetection, wordEmbedder, oneHotCoder, device, name="Evaluator"):
        Object.__init__(self, name=name)

        # member initialization
        self.__dataLoader = dataLoader
        self.__wordEmbedder = wordEmbedder
        self.__oneHotCoder = oneHotCoder
        self.__device = device
        if not criterionSlotFilling == None:
            self.__criterionSlotFilling = copy.deepcopy(criterionSlotFilling)
        else:
            self.__criterionSlotFilling = None
        if not criterionIntentDetection == None:
            self.__criterionIntentDetection = copy.deepcopy(criterionIntentDetection)
        else:
            self.__criterionIntentDetection = None

    def evaluateSlotFillingModel(self, model, testBatchNum):
        '''
        Function to evaluate slot filling model.
        @param model:   model to be evaluated
        @param testBatchNum:  test batch number to be selected for test
        @return:
        '''
        # init, preparation
        trueLabList = []
        predLabList = []

        # turn on model evaluation mode
        model.eval()

        # do prediction, collect predict labels
        accumulateLoss = 0
        batch = 0
        self.__dataLoader.resetBatchIteration(train=False)     # reset test dataset
        with torch.no_grad():
            while True:
                queries, _, seqLabels = self.__dataLoader.nextBatch(train=False)

                if queries == None:    # data is exhausted
                    print ("[WARNING] Test data is exhausted during evaluation. This may be caused by a too large parameter testBatchNum. Evaluate loss is probably not reliable.")
                    break

                # prepare for testing
                # query  --> word embedding
                queriesWordEmbedding = self.__wordEmbedder.wordEmbed(queries)
                # sequence labels   -->  one-hot code
                _, seqLabOneHot = self.__oneHotCoder.encodeSeqLabel(seqLabels)

                # forward
                # assemble data into a list: [sentence1, sentence2, sentence3, ... ...]
                inputData = []
                for sentence in queriesWordEmbedding:
                    sentenceEmbedding = torch.Tensor(sentence)
                    sentenceEmbedding = Utils.move2Device(sentenceEmbedding, self.__device)
                    inputData.append(sentenceEmbedding)
                # do prediction
                predict, seqLengths, _ = model(inputData)

                # calculate loss
                loss = self.__criterionSlotFilling.calculateLoss(predict, seqLengths, seqLabOneHot)
                accumulateLoss = accumulateLoss + loss.item()

                # collect predict result
                # serialize predicted labels
                serializedPredict = Utils.serializeBatchData(predict, seqLengths)
                # find max and push it into result
                for word in serializedPredict:
                    _, index = torch.max(word, 0)
                    predLabList.append(index.item())

                # collect true labels
                for sentence in seqLabels:
                    rt, sentenceToken = self.__oneHotCoder.tokenizeSeqLabel(sentence)
                    if not rt == STATUS_OK:   # error check
                        print ("[ERROR] Fail to evaluate slot filling model. Fail to tokenize following sequence labels:")
                        print (sentence)
                        return STATUS_FAIL, None
                    trueLabList = trueLabList + sentenceToken

                # batch increment
                batch = batch + 1

                # exceeds max test batch?
                if batch >= testBatchNum:   # iteration is done
                    break


        # collect label information
        labelList = self.__oneHotCoder.getSequenceLabelList()

        # do evaluation
        F1Score, report = self.__evaluateClassifier(trueLabList, predLabList, labelList)

        # calculate average loss
        averageLoss = accumulateLoss/batch

        # turn on model train mode
        model.train()
        return STATUS_OK, F1Score, averageLoss, report

    def evaluateIntentDetectionModel(self, model, testBatchNum):
        '''
        Function to evaluate intent detection model.
        @param model:         model to be evaluated
        @param testBatchNum:  test batch number to be selected for test
        @return:
        '''
        # init, preparation
        trueLabList = []
        predLabList = []

        # turn on model evaluation mode
        model.eval()

        # do prediction, collect predict labels
        accumulateLoss = 0
        batch = 0
        self.__dataLoader.resetBatchIteration(train=False)     # reset test dataset
        with torch.no_grad():
            while True:
                queries, clsLabels, _ = self.__dataLoader.nextBatch(train=False)

                if queries == None:    # data is exhausted
                    print ("[WARNING] Test data is exhausted during evaluation. This may be caused by a too large parameter testBatchNum. Evaluate loss is probably not reliable.")
                    break

                # prepare for testing
                # query  --> word embedding
                queriesWordEmbedding = self.__wordEmbedder.wordEmbed(queries)
                # sequence labels   -->  one-hot code
                _, clsLabOneHot = self.__oneHotCoder.encodeClassLabel(clsLabels)

                # forward
                # assemble data into a list: [sentence1, sentence2, sentence3, ... ...]
                inputData = []
                for sentence in queriesWordEmbedding:
                    sentenceEmbedding = torch.Tensor(sentence)
                    sentenceEmbedding = Utils.move2Device(sentenceEmbedding, self.__device)
                    inputData.append(sentenceEmbedding)
                # do prediction
                predict, _ = model(inputData)

                # calculate loss
                loss = self.__criterionIntentDetection.calculateLoss(predict, clsLabOneHot)

                # accumulate loss, will be averaged later
                accumulateLoss = accumulateLoss + loss.item()

                # collect predict result
                for sentence in predict:
                    _, index = torch.max(sentence, 0)
                    predLabList.append(index.item())

                # collect true labels
                for label in clsLabels:
                    rt, labelToken = self.__oneHotCoder.tokenizeClsLabel(label)
                    if not rt == STATUS_OK:
                        print("[ERROR] Fail to evaluate intent detection model. Fail to tokenize following class labels:")
                        print(label)
                        return STATUS_FAIL, None
                    trueLabList.append(labelToken)

                # batch increment
                batch = batch + 1

                # exceeds max test batch?
                if batch >= testBatchNum:   # iteration is done
                    break


        # collect label information
        labelList = self.__oneHotCoder.getClsLabList()

        # do evaluation
        F1Score, report = self.__evaluateClassifier(trueLabList, predLabList, labelList)

        # calculate average loss
        averageLoss = accumulateLoss/batch

        # turn on model train mode
        model.train()
        return STATUS_OK, F1Score, averageLoss, report

    def __evaluateClassifier(self, trueList, predList, classLabelList):
        '''
        Function to evaluate classifiers
        @param trueList:         list[N]
        @param predList:         list[N]
        @param classLabelList:   list[M], M is unique label number
        @return: weighted average F1 score
        '''
        # evaluate result
        precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(trueList, predList, beta=1, average="weighted")

        # report    # code to generate detailed report
        # TODO: generate evaluation report
        report = metrics.classification_report(trueList, predList, labels = list(range(len(classLabelList))), target_names=classLabelList)

        return fbeta_score, report
