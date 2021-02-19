import os
import sys
import time
import configparser
# import argparse
from pickle import dump

sys.path.append("..")
sys.path.append("/sddo")

import torch
import torch.optim as optim
from sklearn.metrics import classification_report, precision_recall_fscore_support

from sddo.sdmodel import modelBasic, modelJoint, sdZSStrategy
import sddo.sdworld.sdUtils as sdUtils
from sddo.sdworld import sdText

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
                         cudaID=cudaID,ctxPretrained="BERT")
    indO = sdData.dataInfo["indO"]

    Perform = []

    # Training start
    batchSize = int(config.get("Train", "batch_size"))
    noBatch = int(sdData.dataInfo["noXtr"] / batchSize)
    noEpoch = int(config.get("Train", "no_epoch"))

    for iRep in range(int(config.get("experimentVer","no_rep"))):

        print("__ build model __")
        # load model
        # input X[BSZ, T], output Hx[BSZ, 2u], Ht[sum(lens), 2u]
        utteranceEncoder = modelBasic.BiLSTM(config,sdData.dataInfo)
        # input Y[noY, Ty], output Hy[noY, 2u]
        classYEncoder = modelBasic.Bert(config,sdData.dataInfo)
        # input seqY[noSeqY, TSeq], output Hseq[noSeqY, 2u]
        seqYEncoder = modelBasic.Bert(config,sdData.dataInfo)
        # input Ht[sum(lens), 2u] output HtBI[noNotO, 2u]
        seqBIOEncoder = modelBasic.classifierBIO(config)

        # Projection
        # input Hx[BSZ, 2u], Hy[noY, 2u], output score Sy[BSZ, noY]
        classProjector = modelJoint.ProjectClass(config)
        # input Ht[BSZ, T, 2u], Hseq[noSeqY, 2u]
        # output score Sseq[BSZ, T, noSeqY]
        seqProjector = modelJoint.ProjectClass(config, isSeq=True)

        if gpuFlag:
            print("__ move models to GPU __")
            utteranceEncoder=utteranceEncoder.cuda(cudaID)
            classYEncoder=classYEncoder.cuda(cudaID)
            seqYEncoder=seqYEncoder.cuda(cudaID)
            classProjector=classProjector.cuda(cudaID)
            seqProjector=seqProjector.cuda(cudaID)
            seqBIOEncoder = seqBIOEncoder.cuda(cudaID)

        Params = list(utteranceEncoder.parameters()) + \
                 list(classProjector.parameters()) + \
                 list(seqProjector.parameters()) + \
                 list(seqBIOEncoder.parameters())

        optimizer = optim.Adam(Params, lr=float(config.get("Train","learning_rate")))

        # tokenY, lensY, _, _, _ = sdUtils.sortBatch(sdData.tokenY[sdData.dataInfo["indCS"]],
        #                                       sdData.lensY[sdData.dataInfo["indCS"]])
        # tokenSeqY, lensSeqY, _, _, _ = sdUtils.sortBatch(
        #                                             sdData.tokenSeqY[sdData.dataInfo["indSeqCS"]],
        #                                             sdData.lensSeqY[sdData.dataInfo["indSeqCS"]])

        tokenY = sdData.tokenY[sdData.dataInfo["indCS"]]
        tokenSeqY = sdData.tokenSeqY[sdData.dataInfo["indSeqCS"]]
        with torch.no_grad():
            classYEncoder.forward(tokenY)
            seqYEncoder.forward(tokenSeqY)
            Hy = classYEncoder.outputHt
            Hseq = seqYEncoder.outputHt
        perform = []

        # del classYEncoder, seqYEncoder

        print("__ training start __")
        for iEpoch in range(noEpoch):
            avg_loss = 0.0
            time_start = time.time()

            utteranceEncoder.train()
            seqBIOEncoder.train()
            classProjector.train()
            seqProjector.train()

            for iBatch in range(noBatch):
                # generate batch
                batchX, batchLen, \
                batchY, batchSeqY, \
                batchBIOY = sdData.generateBatch(batchSize=batchSize)

                # make O flag
                # batchSeqYid = torch.argmax(batchSeqY, dim =1)
                # batchOFlag = torch.eq(batchSeqYid,indO)

                # encode utterance
                utteranceEncoder.forward(batchX,batchLen,sdData.embedding)
                Hx = utteranceEncoder.outputHt # Hx[BSZ, 2u]
                Ht = utteranceEncoder.outputHlen # Ht[sum(lens), 2u]

                # NER
                seqBIOEncoder.forward(Ht, indO)
                HtBI = Ht[seqBIOEncoder.indBI]
                batchSeqYBI = batchSeqY[seqBIOEncoder.indBI]
                # print(seqBIOEncoder.indOther.shape, seqBIOEncoder.indBI.shape)

                # projection
                classProjector.forward(Hx, Hy, batchY)
                seqProjector.forward(HtBI, Hseq, batchSeqYBI)

                # loss and backward
                lossClass = classProjector.lossHinge()
                lossBIO = seqBIOEncoder.lossBCE(batchBIOY, batchLen)
                lossSeq = seqProjector.lossHinge()
                # lossSeqClassifier = seqProjector.lossCE()
                loss = lossClass + lossBIO + lossSeq
                loss.backward()
                optimizer.step()

                # make prediction
                # utterance level
                batchPred = classProjector.pred
                batchYid = torch.argmax(batchY, dim=1)
                # token level
                batchSeqYBI = seqProjector.pred
                if gpuFlag:
                    batchSeqYBI = batchSeqYBI.cpu()
                batchSeqPred = torch.cat((seqBIOEncoder.pred, batchSeqYBI))
                batchSeqYid = torch.argmax(batchSeqY, dim =1)
                batchSeqYid = batchSeqYid[seqBIOEncoder.indNew]

                # training evaluate for batch
                # move2CPU
                if gpuFlag:
                    batchPred = batchPred.cpu()
                    batchYid = batchYid.cpu()
                    batchSeqPred = batchSeqPred.cpu()
                    batchSeqYid = batchSeqYid.cpu()

                # print(classification_report(batchYid, batchPred, digits=4,
                #                             target_names=sdData.dataInfo["CS"]))
                batchAllSeqY = torch.cat((batchSeqYid, batchSeqPred))
                indBatchSeqY = torch.unique(batchAllSeqY,sorted=True)
                seqName = [sdData.dataInfo["SeqCS"][i] for i in indBatchSeqY]
                # print(classification_report(batchSeqYid, batchSeqPred,
                #                             digits=4, target_names=seqName))

                accC = precision_recall_fscore_support(batchYid, batchPred,
                                                       average="weighted")
                accSeq = precision_recall_fscore_support(batchSeqYid, batchSeqPred,
                                                average="weighted")
                avg_loss += float(loss)
                time_train = time.time()
                print("lossSeq: {:.4f}".format(lossSeq),
                      "lossClass: {:.4f}".format(lossClass),
                      "lossBIO: {:.4f}".format(lossBIO),
                      "accC: {:.4f}".format(accC[1]),
                      "accSeq: {:.4f}".format(accSeq[1]),
                      "epoch: ", iEpoch+1,"/", noEpoch,
                      "batch: ",iBatch+1,"/", noBatch,
                      "training time: {:.2f}".format(time_train-time_start))

            # evaluation
            rst = sdZSStrategy.evalZSJ(sdData,batchSize,gpuFlag,
                                       utteranceEncoder, seqBIOEncoder,
                                       classYEncoder, seqYEncoder,
                                       classProjector, seqProjector)
            perform.append(rst)
        Perform.append(perform)
    print("Normal Ending")
    return perform,sdData.dataInfo

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description="ZSJ")
    # parser.add_argument("-e", "--epoch", help="the number of epoch",
    #                     default=100, type=int)

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
