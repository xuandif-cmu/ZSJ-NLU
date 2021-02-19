import torch
from Object import Object
import datetime
import os

from commonVar import STATUS_OK
from commonVar import STATUS_FAIL
from commonVar import STATUS_WARNING


class TrainingLogger(Object):
    '''
    Training logger is designed to log training process:
        1. Report training status: model accuracy, loss tendency, etc.
        2. Save model into disk when necessary, together with model information
        3. Load model from disk
        4. Track best model information
    '''

    def __init__(self, path, evaluator, testBatchNum, name="TrainingLogger"):
        '''
        @param path:    path to save model
        @param evaluator: evaluator to evaluate model, evaluation result shall be reported
        @param testBatchNum: testBatchNum, say, if it's 5, then 5 batches will be selected to test
        @param name: object name
        '''
        # parent init
        Object.__init__(self, name)

        # member init
        self.__path = path      # model saving directory
        self.__evaluator = evaluator    # evaluator to evaluate model
        self.__testBatchNum = testBatchNum  # number of tested batch

        # check directory
        if not os.path.exists(path):
            os.makedirs(path)

    def logSlotFillingModelStatusBatch(self, model, epoch, batch, batchStep):
        '''
        Log slot filling model status, every batchStep batchs
        @param model: model to log
        @param epoch: current epoch
        @param batch: current batch
        @param batchStep: If it's 5, then log will be reported every 5 batches
        @return: No
        '''
        if batch % batchStep == 0:    # time to do batch level log
            # evaluate model
            rt, F1Score, loss, _ = self.__evaluator.evaluateSlotFillingModel(model=model, testBatchNum=self.__testBatchNum)
            if not rt == STATUS_OK:
                print("[ERROR] TrainingLogger. Log slot filling model status. Fail to evaluate slot filling model.")
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("[INFO] %s: Model evaluation. Epoch=%4d, batch=%4d. F1_score=%6f. Average_loss=%6f."%(time, epoch, batch, F1Score, loss))

    def logIntentDetectionModelStatusBatch(self, model, epoch, batch, batchStep):
        '''
        Log intent detection model status, every batchStep batchs
        @param model: model to log
        @param epoch: current epoch
        @param batch: current batch
        @param batchStep: If it's 5, then log will be reported every 5 batches
        @return: No
        '''
        if batch % batchStep == 0:    # time to do batch level log
            # evaluate model
            rt, F1Score, loss, _ = self.__evaluator.evaluateIntentDetectionModel(model=model, testBatchNum=self.__testBatchNum)
            if not rt == STATUS_OK:
                print("[ERROR] TrainingLogger. Log intent detection model status. Fail to evaluate intent detection model.")
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("[INFO] %s: Model evaluation. Epoch=%4d, batch=%4d. F1_score=%6f. Average_loss=%6f."%(time, epoch, batch, F1Score, loss))

    def logSlotFillingModelStatusEpoch(self, model, epoch, epochStep, detailedReport=False):
        '''
        Log slot filling model status, every epoch
        @param model: model to log
        @param epoch:  current epoch
        @param epochStep:  If it's 3, then log will be reported every 3 epochs
        @param detailedReport: True: detailed log will be reported. False: no detailed log.
        @return:
        '''
        if epoch % epochStep == 0:    # time to do batch level log
            # evaluate model
            rt, F1Score, loss, report = self.__evaluator.evaluateSlotFillingModel(model=model, testBatchNum=self.__testBatchNum)
            if not rt == STATUS_OK:
                print("[ERROR] TrainingLogger. Log intent detection model status. Fail to evaluate slot filling model.")

            # log model metrics
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("[INFO] %s: Model evaluation. Epoch=%4d is done. F1_score=%6f. Average_loss=%6f."%(time, epoch, F1Score, loss))
            if detailedReport:
                print("[INFO] %s: Model evaluation. Detailed report: "%(time))
                print (report)

    def logIntentDetectionModelStatusEpoch(self, model, epoch, epochStep, detailedReport=False):
        '''
        Log intent detection model status, every epoch
        @param model: model to log
        @param epoch:  current epoch
        @param epochStep:  If it's 3, then log will be reported every 3 epochs
        @param detailedReport: True: detailed log will be reported. False: no detailed log.
        @return:
        '''
        if epoch % epochStep == 0:    # time to do batch level log
            # evaluate model
            rt, F1Score, loss, report = self.__evaluator.evaluateIntentDetectionModel(model=model, testBatchNum=self.__testBatchNum)
            if not rt == STATUS_OK:
                print("[ERROR] TrainingLogger. Log intent detection model status. Fail to evaluate intent detection model.")

            # log model metrics
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("[INFO] %s: Model evaluation. Epoch=%4d is done. F1_score=%6f. Average_loss=%6f."%(time, epoch, F1Score, loss))
            if detailedReport:
                print("[INFO] %s: Model evaluation. Detailed report: "%(time))
                print(report)

    def saveSlotFillingModelEpoch(self, model, epoch, epochStep):
        '''
        function to periodically save model into disk
        @param model: model to save
        @param epoch: current epoch
        @param epochStep: If it's 3, then model will be saved every 3 epochs
        @return:
        '''
        if epoch % epochStep == 0:
            # construct model name
            fileName = "model_epoch%d"%(epoch)

            # save model
            torch.save(model, self.__path + '/' + fileName+'.pth')

            # log
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("[INFO] %s: Model after epoch %d is saved."%(time, epoch))

    def saveModel(self, model, modelName):
        '''
        save model to specified path, with specified name
        @param model:  model to save
        @param modelName: model file name
        @return:
        '''
        # construct model name
        fileName = modelName

        # save model
        file = self.__path + '/' + fileName + '.pth'
        torch.save(model, file)

        # log
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("[INFO] %s: Model saved as: %s."%(time, file))


    def updateBestModel(self, model, F1Score):
        '''
        Update best model. If input model is better, use it replace the best one.
        @param model:
        @param F1Score:
        @return:
        '''
        return STATUS_OK

    def saveBestModel(self):
        '''
        Save best model to disk.
        @return:
        '''
        return STATUS_OK
