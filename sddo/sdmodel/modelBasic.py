from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from pytorch_transformers import BertModel,BertTokenizer
from sddo.sdworld import sdUtils

class LayerPass(nn.Module):
    def __init__(self):
        super(LayerPass,self).__init__()

    def forward(self,anyInput):
        return anyInput

class AggrEmb(nn.Module):
    def __init__(self,config,dataInfo):
        super(AggrEmb, self).__init__()
        self.method = "avg"
    def forward(self, inpC, inpLen, embedding):
        self.wordEmb = nn.Embedding.from_pretrained(embedding)
        self.sLens = inpLen
        embX = self.wordEmb(inpC.transpose(0, 1))
        embPacked = pack_padded_sequence(embX, self.sLens)
        # TODO


class Bert(nn.Module):
    def __init__(self,config,dataInfo):
        super(Bert,self).__init__()
        self.BERT = BertModel.from_pretrained("bert-base-uncased")
    def forward(self,inpX):

        segments_ids=[0]*inpX.shape[1]
        segments_tensors=torch.tensor([segments_ids])
        if inpX.is_cuda:
            segments_tensors = segments_tensors.cuda(inpX.device.index)

        # old version
        # embedding, _ = self.BERT(inpX, segments_tensors)
        # self.outputHt = torch.sum(embedding[-2], dim=1)

        embeddingEach, embeddingOverall = self.BERT(inpX, segments_tensors)
        self.outputHt = embeddingOverall


class BiLSTM(nn.Module):
    def __init__(self,config,dataInfo):
        super(BiLSTM, self).__init__()

        # load settings
        self.layerLSTM = int(config.get("setting_CTXEncoder","layer_lstm"))
        self.hiddenSize = int(config.get("setting_CTXEncoder","hidden_lstm"))
        self.vocabSize = dataInfo["vocabSize"]
        self.embSize = dataInfo["embSize"]
        self.maxlenX = dataInfo["maxlenX"]

        # layers
        self.wordEmb = nn.Embedding(self.vocabSize, self.embSize)
        self.biLSTM = nn.LSTM(self.embSize, self.hiddenSize, self.layerLSTM,\
                              bidirectional = True, batch_first = True)


    def forward(self,inputX, inputLen, embedding):
        self.wordEmb = nn.Embedding.from_pretrained(embedding)
        self.sLens = inputLen
        embX = self.wordEmb(inputX.transpose(0,1))
        embPacked = pack_padded_sequence(embX, self.sLens)
        outpH, (hidden, cell) = self.biLSTM(embPacked)
        self.outputHt = torch.cat((hidden[-2],hidden[-1]),dim=1) # [BSZ, 2u]
        self.outputH = pad_packed_sequence(outpH, total_length=self.maxlenX)[
            0].transpose(0, 1).contiguous()  # [BSZ, T, 2u]
        self.outputHlen = sdUtils.seqUnsqueeze(self.outputH,inputLen) # [sum(
        # len), 2u]


    def lossCE(self, truthBIO, batchLen):
        truth = sdUtils.seqUnsqueeze(truthBIO, batchLen)
        truthBIOid = torch.argmax(truth, dim=1)
        return self.CE(self.pOther, truthBIOid)

class BiLSTMAtt(nn.Module):
    def __init__(self,config,dataInfo):
        super(BiLSTMAtt, self).__init__()

        # load settings
        self.layerLSTM = int(config.get("setting_CTXEncoder","layer_lstm"))
        self.hiddenSize = int(config.get("setting_CTXEncoder","hidden_lstm"))
        self.attHeadCTX = int(config.get("setting_CTXEncoder","att_head_ctx"))
        self.attDa = int(config.get("setting_CTXEncoder","att_da_ctx"))

        self.dropoutRate = float(config.get("Train","dropout"))

        self.vocabSize = dataInfo["vocabSize"]
        self.embSize = dataInfo["embSize"]
        self.maxlenX = dataInfo["maxlenX"]

        # layers
        self.wordEmb = nn.Embedding(self.vocabSize, self.embSize)
        self.biLSTM = nn.LSTM(self.embSize, self.hiddenSize, self.layerLSTM,\
                              bidirectional = True, batch_first = True)
        self.w1Att = nn.Linear(2*self.hiddenSize,self.attDa, bias = False)
        self.w2Att = nn.Linear(self.attDa, self.attHeadCTX,  bias = False)
        self.dropout = nn.Dropout(self.dropoutRate)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        # init Layers
        self.initWeights()

    def forward(self,inputX, inputLen, embedding):

        self.wordEmb = nn.Embedding.from_pretrained(embedding)
        self.sLens = inputLen
        embX = self.wordEmb(inputX.transpose(0,1))
        embPacked = pack_padded_sequence(embX, self.sLens)
        outpH, (hidden, cell) = self.biLSTM(embPacked)
        ht = torch.cat((hidden[-2],hidden[-1]),dim=1) # [BSZ, 2u]
        self.outputH = pad_packed_sequence(outpH, total_length=self.maxlenX)[
            0].transpose(0,1).contiguous() # [BSZ, T, 2u]
        h1 = self.tanh(self.w1Att(self.dropout(self.outputH))) # [BSZ, T, da]
        h2 = self.softmax(self.w2Att(h1)) # [BSZ, T, R]
        self.outputA = torch.transpose(h2, 1, 2).contiguous() # [BSZ, R, T]
        self.outputAH = torch.bmm(self.outputA, self.outputH) # [BSZ, R, 2u]
        self.outputHt = torch.sum(self.outputAH,dim=1) # [BSZ, 2u]

    def initWeights(self):
        nn.init.xavier_uniform_(self.w1Att.weight)
        nn.init.xavier_uniform_(self.w2Att.weight)
        self.w1Att.weight.requires_grad_(True)
        self.w2Att.weight.requires_grad_(True)

class SlotGate(nn.Module):
    def __init__(self, config, dataInfo):
        super(SlotGate, self).__init__()

        # load settings
        self.layerLSTM = int(config.get("setting_CTXEncoder", "layer_lstm"))  # 3
        self.hiddenSize = int(config.get("setting_CTXEncoder", "hidden_lstm"))  # 64
        self.attHeadCTX = int(config.get("setting_CTXEncoder", "att_head_ctx"))  # 3
        self.attDa = int(config.get("setting_CTXEncoder", "att_da_ctx"))  # 16

        self.dropoutRate = float(config.get("Train", "dropout"))  # 0.5

        self.vocabSize = dataInfo["vocabSize"]  # 10895
        self.embSize = dataInfo["embSize"]  # 300
        self.maxlenX = dataInfo["maxlenX"]  # [1, 2, ..., 35]

        # layers
        self.wordEmb = nn.Embedding(self.vocabSize, self.embSize)
        self.biLSTM = nn.LSTM(self.embSize, self.hiddenSize, self.layerLSTM,\
                              bidirectional=True, batch_first=True)

        self.w1Att = nn.Linear(2 * self.hiddenSize, 2 * self.hiddenSize, bias=False)        # [2u, 2u]
        self.w2Att = nn.Linear(2 * self.hiddenSize, 2 * self.hiddenSize, bias=False)        # [2u, 2u]
        self.w3Att = nn.Linear(2 * self.hiddenSize, 2 * self.hiddenSize, bias=False)        # [2u, 2u]
        self.w4Att = nn.Linear(2 * self.hiddenSize, 2 * self.hiddenSize, bias=False)        # [2u, 2u]
        self.v = nn.Linear(2 * self.hiddenSize, 2 * self.hiddenSize, bias=False)        # [2u, 2u]

        self.dropout = nn.Dropout(self.dropoutRate)

        self.actFun = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        # init Layers
        self.initWeights()

    def forward(self, inputX, inputLen, embedding):
        self.wordEmb = nn.Embedding.from_pretrained(embedding)
        self.sLens = inputLen
        embX = self.wordEmb(inputX.transpose(0, 1))
        embPacked = pack_padded_sequence(embX, self.sLens)
        outpH, (hidden, cell) = self.biLSTM(embPacked)

        # for intent
        self.outputHt = torch.cat((hidden[-2],hidden[-1]),dim=1) # [BSZ, 2u]

        # ht = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [BSZ, 2u]
        self.outputH = pad_packed_sequence(outpH, total_length=self.maxlenX)[
            0].transpose(0, 1).contiguous()  # [BSZ, T/35, 2u]
        e = self.actFun(self.w1Att(self.dropout(self.outputH)))  # [BSZ, T/35, 2u]
        alpha = self.softmax(self.w2Att(e))  # [BSZ, T/35, 2u]

        # c = torch.zeros()
        c = alpha * self.outputH   # ?????? [BSZ, T, 2u]

        # slot gate
        g =self.v(self.actFun(c + self.w4Att(self.outputH)))
        hAddc = self.outputH + c * g # [BSZ, T, 2u]
        y = self.softmax(self.w3Att(hAddc))  # [BSZ, T, 2u]
        self.outputHlen = sdUtils.seqUnsqueeze(y, inputLen)  # [sum(len), 2u]

    def initWeights(self):
        nn.init.xavier_uniform_(self.w1Att.weight)
        nn.init.xavier_uniform_(self.w2Att.weight)
        self.w1Att.weight.requires_grad_(True)
        self.w2Att.weight.requires_grad_(True)


class classifierBIO(nn.Module):
    def __init__(self,config):
        super(classifierBIO,self).__init__()

        self.u = int(config.get("setting_CTXEncoder","hidden_lstm"))
        self.transform = nn.Sequential(
            nn.BatchNorm1d(2*self.u),
            nn.Linear(2*self.u,2*self.u),
            nn.ReLU(),
            nn.Linear(2*self.u,3),
            nn.Softmax(dim=1)
        )
        self.CE = nn.CrossEntropyLoss()

    def forward(self, inpX, indO):
        self.pOther = self.transform(inpX)
        pred = torch.argmax(self.pOther, dim=1)
        self.indOther = torch.squeeze((pred == 2).nonzero(), dim=1)
        self.indBI = torch.squeeze((pred !=2).nonzero(), dim=1)
        self.pred = indO * torch.ones(self.indOther.shape[0]).long()
        self.indNew = torch.cat((self.indOther,self.indBI))
        # self.outpX = inpX[self.indBI,] # HtBI[noNotO, 2u]

    def lossBCE(self, truthBIO, batchLen):
        truth = sdUtils.seqUnsqueeze(truthBIO, batchLen)
        truthBIOid = torch.argmax(truth,dim=1)
        return self.CE(self.pOther,truthBIOid)


class attTest(nn.Module):
    def __init__(self, config, dataInfo):
        super(attTest,self).__init__()

        # load settings
        self.hiddenSize = int(config.get("setting_CTXEncoder","hidden_lstm"))
        self.noY = len(dataInfo["vocabY"])
        self.noSeqY = len(dataInfo["vocabSeqY"])

        self.BiLSTM = BiLSTM(config, dataInfo)
        # TODO self.attention
        self.FCx = nn.Sequential(OrderedDict([
                  ('batchNorm', nn.BatchNorm1d(2*self.hiddenSize)),
                  ('linear1', nn.Linear(2*self.hiddenSize, 2*self.hiddenSize)),
                  ('activate2', nn.Tanh()),
                  ('linear2', nn.Linear(2*self.hiddenSize, self.noY)),
                  ('activate2', nn.Tanh()),
                  ('softmax', nn.Softmax())
                ])
        )
        self.FCt = nn.Sequential(OrderedDict([
                  ('batchNorm', nn.BatchNorm1d(2*self.hiddenSize)),
                  ('linear1', nn.Linear(2*self.hiddenSize, 2*self.hiddenSize)),
                  ('activate2', nn.Tanh()),
                  ('linear2', nn.Linear(2*self.hiddenSize, self.noSeqY)),
                  ('activate2', nn.Tanh()),
                  ('softmax', nn.Softmax())
                ])
        )
        # loss
        self.CE = nn.CrossEntropyLoss()

    def forward(self, batchX, batchLen, embedding):
        self.BiLSTM.forward(batchX, batchLen,embedding)
        Hx = self.BiLSTM.outputHt
        Ht = self.BiLSTM.outputH
        # TODO attention

        self.outpx = self.FCx(Hx)
        Ht = sdUtils.seqUnsqueeze(Ht,batchLen)
        self.outpt = self.FCt(Ht)
        self.predY = torch.argmax(self.outpx,dim=1)
        self.predSeqY = torch.argmax(self.outpt,dim=1)

    def lossCE(self, batchY, batchSeqY, batchLen):

        truthY = torch.argmax(batchY, dim=1)
        loss_x = self.CE(self.outpx,truthY)
        truthSeqY = torch.argmax(batchSeqY, dim=1)
        loss_t = self.CE(self.outpt,truthSeqY)
        return loss_x + loss_t