import torch
import time
from sddo.sdmodel import modelBasic
from sklearn.metrics import classification_report, precision_recall_fscore_support
import sddo.sdworld.sdUtils as sdUtils

def evalZSJ(sdData, batchSize, gpuFlag,
            utteranceEncoder,seqBIOEncoder,
            classYEncoder, seqYEncoder,
            classProjector, seqProjector):

    print("__ test start __")
    timeTest0 = time.time()

    # load model
    utteranceEncoder.eval()
    classYEncoder.eval()
    seqYEncoder.eval()
    classProjector.eval()
    seqProjector.eval()
    seqBIOEncoder.eval()

    # # load data
    tokenY = sdData.tokenY
    tokenSeqY = sdData.tokenSeqY
    predC = torch.tensor([], dtype=torch.long)
    predSeqC = torch.tensor([], dtype=torch.long)
    truthC = torch.tensor([], dtype=torch.long)
    truthSeqC = torch.tensor([], dtype=torch.long)
    noBatch = int(sdData.dataInfo["noXte"]/batchSize)+1

    with torch.no_grad():

        if "BERT" == "BERT":
            classYEncoder.forward(tokenY)
            seqYEncoder.forward(tokenSeqY)

        Hy = classYEncoder.outputHt
        Hseq = seqYEncoder.outputHt

        for i in range(noBatch):

            # generate batch for test
            batchX, batchLen, \
            batchY, batchSeqY, \
            batchBIOY= sdData.generateBatch(batchSize=batchSize, i=i, isTrain=False,
                                            mode="inTurn")

            # encode utterance
            utteranceEncoder.forward(batchX, batchLen, sdData.embedding)
            Hx = utteranceEncoder.outputHt  # Hx[BSZ, 2u]
            Ht = utteranceEncoder.outputHlen  # Ht[sum(lens), 2u]

            # NER
            seqBIOEncoder.forward(Ht, indO = 0)
            HtBI = Ht[seqBIOEncoder.indBI]
            batchSeqYBI = batchSeqY[seqBIOEncoder.indBI]

            # projection
            classProjector.forward(Hx, Hy, batchY)
            seqProjector.forward(HtBI, Hseq, batchSeqYBI)

            # make prediction
            batchPred = classProjector.pred
            batchSeqYBI = seqProjector.pred
            batchSeqYO = seqBIOEncoder.pred
            batchYid = torch.argmax(batchY, dim=1)
            batchSeqYid = torch.argmax(batchSeqY, dim =1)
            batchSeqYid = batchSeqYid[seqBIOEncoder.indNew]
            if gpuFlag:
                batchPred = batchPred.cpu()
                batchSeqYBI = batchSeqYBI.cpu()
                batchSeqYO = batchSeqYO.cpu()
                batchYid = batchYid.cpu()
                batchSeqYid = batchSeqYid.cpu()
            batchSeqPred = torch.cat((batchSeqYBI, batchSeqYO))

            predC = torch.cat((predC,batchPred))
            truthC = torch.cat((truthC, batchYid))
            predSeqC = torch.cat((predSeqC,batchSeqPred))
            truthSeqC = torch.cat((truthSeqC, batchSeqYid))

    print(classification_report(truthC,predC,digits=4,
                                target_names=sdData.dataInfo["vocabY"]))
    print(classification_report(truthSeqC,predSeqC,digits=4,
                                target_names=sdData.dataInfo["vocabSeqY"]))

    perform = {}
    perform["logC"] = precision_recall_fscore_support(truthC,predC)
    perform["logavgC"] = precision_recall_fscore_support(truthC, predC,
                                                       average="weighted")
    perform["logSeq"] = precision_recall_fscore_support(truthSeqC, predSeqC)
    perform["logavgC"] = precision_recall_fscore_support(truthSeqC, predSeqC,
                                                average="weighted")
    return perform