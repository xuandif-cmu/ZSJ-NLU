import os
import sys
import json
import re
import torch
import random
import sddo.sdworld.sdUtils as sdUtils
from gensim.models.keyedvectors import KeyedVectors
from pytorch_transformers import BertModel,BertTokenizer
# from pytorch_pretrained_bert import BertModel, BertTokenizer

class sdText:
    def __init__(self,path, dataset, nameW2V=None,
                 cudaID=None, ctxPretrained=None, setting="zsj", # zsj, supervised
                 freezeSplit=None, freezeTestClass=None,
                 pClassSplit=0.3, pTrainSplit=0.7):

        # input mode
        self.path = path # str, "path/"
        self.nameW2V = nameW2V # str, the filename of W2V file, should follow the format after gensim wordvec
        self.dataset = dataset # str
        self.ctxPretrained = ctxPretrained # str, the pretrained contextual model to use
        self.cudaID = cudaID

        self.checkInputFile() # automatically extract dataset and format in the given path (TODO)
        if dataset is not None:
            self.dataset=dataset # str or omit,the name of dataset, if omited, extract in the path

        self.dataInfo = {}
        self.loadText()
        if self.nameW2V is not None:
            self.loadW2V(self.path+self.nameW2V)

        # preProcessing
        # tokenized X, Y, and sequence label
        if self.ctxPretrained == "BERT":
            self.tokenizeBERTCSide()
        else:
            self.tokenize()
        # make Y and sequence label to vectors
        self.processY()

        # move data to gpu
        if cudaID is not None:
            self.move2GPU(cudaID)

        # basic info of the dataset
        self.dataInfo["noData"] = len(self.textY)
        self.dataInfo["noY"] = len(self.dataInfo["vocabY"])
        self.dataInfo["noSeqY"] = len(self.dataInfo["vocabSeqY"])
        self.dataInfo["vocabSize"], self.dataInfo["embSize"] = self.embedding.shape

        # split dataset
        # {indTr: list of index, indTe: list of index}, indicate the training index and test index
        # or a binary array
        self.freezeSplit = freezeSplit
        self.freezeTestClass = freezeTestClass # a list of str
        self.pClassSplit = pClassSplit # a float scalar in (0,1)， select pClassSplit of class a unseen class
        self.pTrainSplit = pTrainSplit # a float scalar in (0,1), select pTrainSplit of samples from each seen class
        # generate split index
        self.setting = setting
        if self.setting == "supervised":
            self.generateSplitSupervised()
        elif self.setting == "zsj":
            self.generateSplitZS()

        print("__ data loading done __")

    def checkInputFile(self):
        files=os.listdir(self.path)
        # json has higher priority
        self.format = "json"

    def loadW2V(self,nameW2V):
        print("__ loading W2V" ,nameW2V, "__")
        self.w2v=KeyedVectors.load_word2vec_format(nameW2V,binary=False)
        self.embedding = torch.from_numpy(sdUtils.normMatrix(self.w2v.syn0))

    def loadText(self):
        print("__ loading Dataset,",self.dataset, "__")
        if self.format == "json":
            with open(self.path+"data"+self.dataset+".json", "r") as f:
                rawData = json.load(f)
        self.textX = [sample["query"].split(" ") for sample in rawData]

        self.textY = [sdUtils.addSpaceCap(sample["classLabel"]) for sample in rawData]
        self.textY = [query.lower() for query in self.textY]
        self.dataInfo["vocabY"] = list(set(self.textY))

        self.seqBIO = [sample["seqLabel"].split(" ") for sample in rawData]
        self.textSeqY = [sdUtils.removeBIO(query) for query in self.seqBIO] # ["slot_descriptoin"]
        self.textSeqY = [sdUtils.sdReplace("_"," ", query) for query in self.textSeqY]
        self.seqBIO = [sdUtils.getBIO(query) for query in self.seqBIO]

        vocabSeqY = [item for sublist in self.textSeqY for item in sublist]
        vocabSeqY = list(set(vocabSeqY))
        vocabSeqY.remove("O")
        self.dataInfo["vocabSeqY"] = ["O"]+vocabSeqY # [O, str of slot description]
        self.vocabBIO = ["B","I","O"]


    def tokenizeBERTCSide(self):

        print("__ Tokenize classes BERT __")

        # self.X [noX, maxlen], index in the dict
        # self.lensX [noX], lens of the utterance
        self.X = [sdUtils.sdTokenizerW2V(query,self.w2v) for query in self.textX]
        self.X, self.lensX = sdUtils.padding(self.X)
        self.dataInfo["maxlenX"] = max(self.lensX)

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        tokenY = [tokenizer.tokenize(query) for query in self.dataInfo["vocabY"]]
        self.tokenY = [tokenizer.convert_tokens_to_ids(query) for query in tokenY]
        self.tokenY, self.lensY = sdUtils.padding(self.tokenY)
        self.dataInfo["maxlenY"] = max(self.lensY)

        tokenSeqY = [tokenizer.tokenize(query) for query in self.dataInfo["vocabSeqY"]]
        self.tokenSeqY = [tokenizer.convert_tokens_to_ids(query) for query in tokenSeqY]
        self.tokenSeqY, self.lensSeqY = sdUtils.padding(self.tokenSeqY)
        self.dataInfo["maxlenSeqY"] = max(self.lensSeqY)


    def tokenize(self):
        print("__ Tokenize __")

        # self.X [noX, maxlen], index in the dict
        # self.lensX [noX], lens of the utterance
        self.X = [sdUtils.sdTokenizerW2V(query,self.w2v) for query in self.textX]
        self.X, self.lensX = sdUtils.padding(self.X)
        self.dataInfo["maxlenX"] = max(self.lensX)

        # self.tokenY [noY, maxlenY]
        # self.lensY [noY]
        self.tokenY = [label.split(" ") for label in self.dataInfo["vocabY"]]
        self.tokenY = [sdUtils.sdTokenizerW2V(query,self.w2v) for query in self.tokenY]
        self.tokenY, self.lensY = sdUtils.padding(self.tokenY)
        self.dataInfo["maxlenY"] = max(self.lensY)

        # self.tokenSeqY [noSeqYnoBIO, maxlenSeqYnoBIO]
        # self.lensSeqY [noSeqYnoBIO]
        self.tokenSeqY = [label.split(" ") for label in self.dataInfo["vocabSeqY"]]
        self.tokenSeqY = [sdUtils.sdTokenizerW2V(query,self.w2v) for query in self.tokenSeqY]
        self.tokenSeqY, self.lensSeqY = sdUtils.padding(self.tokenSeqY)
        self.dataInfo["maxlenSeqY"] = max(self.lensSeqY)

    def processY(self):
        print("__ Processing Y __")
        self.Y = sdUtils.getHot(self.textY,self.dataInfo["vocabY"],False)
        self.BIOY = sdUtils.getHot(self.seqBIO,self.vocabBIO, True)
        self.seqY = sdUtils.getHot(self.textSeqY,self.dataInfo["vocabSeqY"], True)

    def move2GPU(self,cudaID):
        print("__ move data to GPU __")

        # self.X = self.X.cuda(cudaID)
        # self.Y = self.Y.cuda(cudaID)
        # self.BIOY = self.BIOY.cuda(cudaID)
        # self.seqY = self.seqY.cuda(cudaID)
        self.tokenY = self.tokenY.cuda(self.cudaID)
        self.tokenSeqY = self.tokenSeqY.cuda(self.cudaID)

        # self.lensX = self.lensX.cuda(cudaID)
        self.lensY = self.lensY.cuda(self.cudaID)
        self.lensSeqY = self.lensSeqY.cuda(self.cudaID)

        self.embedding = self.embedding.cuda(self.cudaID)

    def generateSplitSupervised(self):
        print("__ Spliting Dataset with", str(self.pTrainSplit), "training set __")

        if self.freezeSplit is None:
            self.indTr = []
            self.indTe = []
            for c in self.dataInfo["vocabY"]:
                indTemp = [indx for indx, value in enumerate(self.textY) if value == c]
                noXtrThisC = int(len(indTemp)*self.pTrainSplit)
                self.indTr.extend(indTemp[0:noXtrThisC])
                self.indTe.extend(indTemp[noXtrThisC:])
        else:
            self.indTr = self.freezeSplit["indTr"]
            self.indTe = self.freezeSplit["indTe"]

        random.shuffle(self.indTr)
        random.shuffle(self.indTe)
        self.dataInfo["noXtr"] = len(self.indTr)
        self.dataInfo["noXte"] = len(self.indTe)


    def generateSplitZS(self):
        print("__ Spliting Dataset under zero-shot setting __")

        if self.freezeSplit is None:
            self.indTr = []
            self.indTe = []
            self.dataInfo["CU"] = self.freezeTestClass
            if self.freezeTestClass is None:
                self.noCU = int(self.dataInfo["noY"] * self.pClassSplit)
                self.dataInfo["CU"] = self.dataInfo["vocabY"][-self.noCU:]
            self.indTe.extend([indx for indx, j in enumerate(self.textY) if j in self.dataInfo["CU"]])
            self.dataInfo["indCU"] = [self.dataInfo["vocabY"].index(c) for c in self.dataInfo["CU"]]
            self.dataInfo["CS"] = [x for x in self.dataInfo["vocabY"] if x not in self.dataInfo["CU"]]
            self.noCS = len(self.dataInfo["CS"])
            self.dataInfo["indCS"] = [self.dataInfo["vocabY"].index(c) for c in self.dataInfo["CS"]]

            for c in self.dataInfo["CS"]:
                indTemp = [indx for indx, value in enumerate(self.textY) if value == c]
                noXtrThisC = int(len(indTemp)*self.pTrainSplit)
                self.indTr.extend(indTemp[0:noXtrThisC])
                self.indTe.extend(indTemp[noXtrThisC:])
        else:
            self.indTr = self.freezeSplit["indTr"]
            self.indTe = self.freezeSplit["indTe"]

        random.shuffle(self.indTr)
        random.shuffle(self.indTe)
        self.dataInfo["noXtr"] = len(self.indTr)
        self.dataInfo["noXte"] = len(self.indTe)

        seqYtr = self.seqY[self.indTr]
        seqYtrSumCol = torch.sum(torch.sum(seqYtr,dim=1),dim=0)

        self.dataInfo["indSeqCS"] = torch.squeeze((seqYtrSumCol!=0).nonzero())
        self.dataInfo["indSeqCU"] = torch.squeeze((seqYtrSumCol==0).nonzero())
        self.dataInfo["SeqCS"] = [self.dataInfo["vocabSeqY"][i] for i in self.dataInfo["indSeqCS"]]
        self.dataInfo["SeqCU"] = [self.dataInfo["vocabSeqY"][i] for i in self.dataInfo["indSeqCU"]]

        self.dataInfo["indO"] = self.dataInfo["vocabSeqY"].index("O")
        self.dataInfo["indOinS"] = self.dataInfo["SeqCS"].index("O")

    def generateBatch(self, isSeqToSeq=False, batchSize=50, i=None, isTrain=True, mode="random"):

        if isTrain:
            indSplit = self.indTr
            noData = self.dataInfo["noXtr"]
            if self.setting == "zsj":
                indClass = self.dataInfo["indCS"]
                indSeqClass = self.dataInfo["indSeqCS"]
        else:
            indSplit = self.indTe
            noData = self.dataInfo["noXte"]
            if self.setting == "zsj":
                indSeqClass = list(range(self.dataInfo["noSeqY"]))
                indClass = list(range(self.dataInfo["noY"]))

        if mode == "random":
            indBatch = sdUtils.generateBatch(noData,batchSize)
        elif mode == "inTurn":
            indStart = i*batchSize
            indEnd = min(i*batchSize + batchSize, noData)
            indBatch = range(indStart, indEnd)

        batchX = self.X[indSplit][indBatch] # [BSZ, T]
        batchLen = self.lensX[indSplit][indBatch] # [BSZ]
        batchY = self.Y[indSplit][indBatch]
        if self.setting == "zsj":
            batchY=batchY[:,indClass] # [BSZ, noC]
        batchBIOY = self.BIOY[indSplit][indBatch] # [BSZ, T, 3]
        batchSeqY = self.seqY[indSplit][indBatch] # [BSZ, T, noSeqC]
        if self.setting == "zsj":
            batchSeqY=batchSeqY[:,:,indSeqClass] # [BSZ, noC]

        # sort batch by lens
        batchX, batchLen, \
        batchY, batchSeqY, \
        batchBIOY = sdUtils.sortBatch(batchX,batchLen,
                                      batchY,batchSeqY,batchBIOY)
        # remove padded tokens labels
        if isSeqToSeq==False:
            batchSeqY = sdUtils.seqUnsqueeze(batchSeqY, batchLen)

        if self.cudaID is not None:
            batchX = batchX.cuda(self.cudaID)
            batchY = batchY.cuda(self.cudaID)
            batchSeqY = batchSeqY.cuda(self.cudaID)
            batchBIOY = batchBIOY.cuda(self.cudaID)
            batchLen = batchLen.cuda(self.cudaID)
            
        if isSeqToSeq==False:
            return batchX, batchLen, batchY, batchSeqY, batchBIOY
        
        return batchX, batchLen, batchY, batchSeqY