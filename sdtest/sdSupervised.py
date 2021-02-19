import os
import sys
import time
import configparser
from pickle import dump

sys.path.append("..")
sys.path.append("/sddo")

import torch
import torch.optim as optim
from sklearn.metrics import classification_report, precision_recall_fscore_support

from sddo.sdmodel import modelBasic, modelJoint, sdZSStrategy
import sddo.sdworld.sdUtils as sdUtils
from sddo.sdworld import sdText

def evaluate(sdData, utteranceEncoder,
             batchSize=50, gpuFlag=False):

    print("__ test start __")
    timeTest0 = time.time()

    utteranceEncoder.eval()

    # # load data
    tokenY = sdData.tokenY
    tokenSeqY = sdData.tokenSeqY
    predC = torch.tensor([], dtype=torch.long)
    predSeqC = torch.tensor([], dtype=torch.long)
    truthC = torch.tensor([], dtype=torch.long)
    truthSeqC = torch.tensor([], dtype=torch.long)
    noBatch = int(sdData.dataInfo["noXte"]/batchSize)+1


    with torch.no_grad():
        for i in range(noBatch):
            # generate batch for test
            batchX, batchLen, \
            batchY, batchSeqY, \
            batchBIOY= sdData.generateBatch(batchSize=batchSize, i=i,
                                            isTrain=False, mode="inTurn")

            # encode utterance
            utteranceEncoder.forward(batchX, batchLen, sdData.embedding)

            # make prediction
            batchPred = utteranceEncoder.predY
            batchYid = torch.argmax(batchY, dim=1)
            batchSeqPred = utteranceEncoder.predSeqY
            batchSeqYid = torch.argmax(batchSeqY, dim=1)

            # training evaluate for batch
            # move2CPU
            if gpuFlag:
                batchPred = batchPred.cpu()
                batchYid = batchYid.cpu()
                batchSeqPred = batchSeqPred.cpu()
                batchSeqYid = batchSeqYid.cpu()
            predC = torch.cat((predC, batchPred))
            truthC = torch.cat((truthC, batchYid))
            predSeqC = torch.cat((predSeqC, batchSeqPred))
            truthSeqC = torch.cat((truthSeqC, batchSeqYid))

        print(classification_report(truthC, predC, digits=4,
                                    target_names=sdData.dataInfo["vocabY"]))
        batchAllSeqY = torch.cat((truthSeqC, predSeqC))
        indBatchSeqY = torch.unique(batchAllSeqY, sorted=True)
        seqName = [val for ind, val in enumerate(sdData.dataInfo["vocabSeqY"]) if ind in indBatchSeqY]
        print(classification_report(truthSeqC, predSeqC,digits=4, target_names=seqName))

        perform = {}
        perform["logC"] = precision_recall_fscore_support(truthC, predC)
        perform["logavgC"] = precision_recall_fscore_support(truthC, predC,
                                                             average="weighted")
        perform["logSeq"] = precision_recall_fscore_support(truthSeqC, predSeqC)
        perform["logavgC"] = precision_recall_fscore_support(truthSeqC, predSeqC,
                                                             average="weighted")
        return perform

def main(config):

    # get gpu flag
    gpuAvaliable = torch.cuda.is_available()
    gpuFlag = gpuAvaliable and config.get('Train','use_gpu')=="True"
    cudaID = None
    if gpuFlag:
        print("__ use gpu __")
        cudaID = int(config.get("Train","cuda_id"))

    # load data
    dataPath = config.get("path", "data_path")
    filenameW2V = config.get("path","filename_w2v")
    dataset = config.get("path","dataset")
    sdData=sdText.sdText(path=dataPath, dataset=dataset, nameW2V=filenameW2V,
                         cudaID=cudaID,ctxPretrained="BERT",setting="supervised")

    Perform = []
    # Training start
    batchSize = int(config.get("Train", "batch_size"))
    noBatch = int(sdData.dataInfo["noXtr"] / batchSize)
    noEpoch = int(config.get("Train", "no_epoch"))

    for iRep in range(int(config.get("experimentVer","no_rep"))):

        print("__ build model __")
        # load model
        # input X[BSZ, T], output Hx[BSZ, 2u], Ht[sum(lens), 2u]
        utteranceEncoder = modelBasic.attTest(config,sdData.dataInfo)
        if gpuFlag:
            print("__ move models to GPU __")
            utteranceEncoder=utteranceEncoder.cuda(cudaID)
        Params = list(utteranceEncoder.parameters())
        optimizer = optim.Adam(Params, lr=float(config.get("Train","learning_rate")))
        perform = []

        print("__ training start __")
        for iEpoch in range(noEpoch):
            avg_loss = 0.0
            time_start = time.time()

            utteranceEncoder.train()
            for iBatch in range(noBatch):
                # generate batch
                batchX, batchLen, \
                batchY, batchSeqY, \
                batchBIOY = sdData.generateBatch(batchSize=batchSize)

                utteranceEncoder.forward(batchX,batchLen,sdData.embedding)
                loss = utteranceEncoder.lossCE(batchY,batchSeqY,batchLen)
                loss.backward()
                optimizer.step()

                # make prediction
                # utterance level
                batchPred = utteranceEncoder.predY
                batchYid = torch.argmax(batchY, dim=1)
                # token level
                # batchSeqYBI = seqProjector.pred
                # if gpuFlag:
                #     batchSeqYBI = batchSeqYBI.cpu()
                # batchSeqPred = torch.cat((seqBIOEncoder.pred, batchSeqYBI))
                # batchSeqYid = torch.argmax(batchSeqY, dim =1)
                # batchSeqYid = batchSeqYid[seqBIOEncoder.indNew]
                batchSeqPred = utteranceEncoder.predSeqY
                # batchSeqY = sdUtils.seqUnsqueeze(batchSeqPred, batchLen)
                batchSeqYid = torch.argmax(batchSeqY, dim=1)

                # training evaluate for batch
                # move2CPU
                if gpuFlag:
                    batchPred = batchPred.cpu()
                    batchYid = batchYid.cpu()
                    batchSeqPred = batchSeqPred.cpu()
                    batchSeqYid = batchSeqYid.cpu()

                # print(classification_report(batchYid, batchPred, digits=4,
                #                             target_names=sdData.dataInfo["vocabY"]))
                batchAllSeqY = torch.cat((batchSeqYid, batchSeqPred))
                indBatchSeqY = torch.unique(batchAllSeqY,sorted=True)
                seqName = [val for ind, val in enumerate(sdData.dataInfo["vocabSeqY"]) if ind in indBatchSeqY]
                # print(classification_report(batchSeqYid, batchSeqPred,digits=4, target_names=seqName))

                accC = precision_recall_fscore_support(batchYid, batchPred,
                                                       average="weighted")
                accSeq = precision_recall_fscore_support(batchSeqYid, batchSeqPred,
                                                average="weighted")
                avg_loss += float(loss)
                time_train = time.time()
                print("lossSeq: {:.4f}".format(loss),
                      "accC: {:.4f}".format(accC[1]),
                      "accSeq: {:.4f}".format(accSeq[1]),
                      "epoch: ", iEpoch+1,"/", noEpoch,
                      "batch: ",iBatch+1,"/", noBatch,
                      "training time: {:.2f}".format(time_train-time_start))

            # evaluation
            rst = evaluate(sdData,utteranceEncoder, batchSize,gpuFlag)
            perform.append(rst)
        Perform.append(perform)
    print("Normal Ending")
    return perform,sdData.dataInfo

if __name__ == '__main__':

    testCode = "helloW"
    outpPath = "./rst/" + testCode + "/"

    configPath = ["../config/templateZSJ.ini"]
    if not os.path.isdir(outpPath):
        os.makedirs(outpPath)

    for configpath in configPath:

        rst = {}
        outpFilename = time.strftime('%Y%m%d_%H%M%S') + ".pkl"

        config = configparser.ConfigParser()
        config.read(configPath)

        rst["config"] = config._sections
        rst["perform"], rst["dataInfo"] = main(config)

        with open(outpPath+outpFilename,"wb") as f:
            dump(rst, f)
