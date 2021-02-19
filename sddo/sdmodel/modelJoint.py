import torch
import torch.nn as nn
import torch.nn.functional as F
import sddo.sdworld.sdUtils as sdUtils

from sddo.sdplay.tsne import tsneBoth

class SlotAtten(nn.Module):
    def __init__(self, config):
        super(SlotAtten,self).__init__()
        #self.linear = nn.Linear(self.utter_dim+self.slot_dim, self.slot_dim, bias =False)
        
        self.u = int(config.get("setting_CTXEncoder","hidden_lstm"))
        #self.slot_dim = dataInfo["indSeqCS"]
        
        if config.get("experimentVer","ctx_encoder")=="BERT":
            seqInpDim = 768
        else:
            seqInpDim = 2*self.u
        
        if config.get("setting_SeqProjector","atten_dim")=="True":
            atten_dim = 2 * self.u
        else:
            atten_dim = 1
            
        self.W =  nn.Sequential(
            nn.Linear(6 * self.u, 2 * self.u),
            nn.ReLU(),
            nn.Linear(2 * self.u, atten_dim)
        )

        self.transformU = nn.Sequential(
            nn.Linear(2 * self.u, 2 * self.u),
            nn.ReLU(),
            nn.Linear(2 * self.u, 2 * self.u)
        )

        self.transformS = nn.Sequential(
            nn.BatchNorm1d(seqInpDim),
            nn.Linear(seqInpDim, 2 * self.u),
            nn.ReLU(),
            nn.Linear(2 * self.u, 2 * self.u)
        )

        self.CE = torch.nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, Hx, Hs,truthSeqY, inputLen):
        #Hx : utterance encoded BLSTM [BSZ, T, 2u] / BLSTMATT [BSZ, R, 2u]
        #Hs : slot encoded [K, seqInpDim] K : number of slot class
        #truthSeqY： one-hot label for Seq Y [BSZ T K]
        #inputLen: sum(lens) without pad

        self.batchSize = Hx.shape[0]
        self.utter_dim = Hx.shape[1]
        self.slot_dim = Hs.shape[0]
                
        self.transHs = self.transformS(Hs)#[k,2u]
        #self.transHx = self.transformU(Hx) #[bsz T 2u]

        matchHs = self.transHs.unsqueeze(0).unsqueeze(1).repeat(Hx.shape[0], self.utter_dim, 1,1)
        matchHx = Hx.unsqueeze(2).repeat(1, 1, self.slot_dim, 1)
        self.atten = self.W(torch.cat((torch.cat((matchHx, matchHs),dim=3), matchHx*matchHs),dim=3)) #[BSZ T K 2u]
         

        if self.atten.shape[3] == 1: #[BSZ T K 1]
            self.atten_score = self.softmax(self.atten.squeeze(3)) #[BSZ T K]
            self.G = torch.matmul(self.atten_score,self.transHs) #[BSZ T 2u]
        else: #[BSZ T K 2u]
            self.atten_score  = self.softmax(torch.sum(self.atten,dim=3)) #[Bsz, T, K]
            self.atten_dim = nn.Softmax(dim=3)(self.atten) #[Bsz, T, K,2u]
            self.G = torch.sum(self.atten_dim * self.transHs,dim=2) # [BSZ,T,2u] element-wise

        self.outputU = torch.cat((Hx,self.G + Hx),dim=2) #[BSZ, T, 4u]
        self.atten_score_flat = sdUtils.seqUnsqueeze(self.atten_score,inputLen) #[sum(lens) K]
        self.pred = torch.argmax(self.atten_score_flat,dim=1) # [sum(lens)]
        self.truth = truthSeqY  #[sum(lens) K]
   
    def lossCE(self,inputLen):
        self.truth_flat = sdUtils.seqUnsqueeze(self.truth, inputLen)
        truthSeqYid = torch.argmax(self.truth_flat, dim=1)#[sum(lens)]
        loss = self.CE(self.atten_score_flat, truthSeqYid)
        #print('loss',loss)
        return loss

    def lossProdIB(self):
        # product of seq probability without "O"
        seq_prob = self.truth.type(torch.cuda.FloatTensor) * self.atten_score #[bsz T k]
        seq_prob = torch.sum(seq_prob[:,:,1:],dim=2) #[bsz T k-1] --> #[bsz T]
        loss = torch.sum(torch.prod((1-seq_prob), dim=1))
        return loss
    
    def lossProd(self,inputLen):
        #product of seq probability
        seq_prob = self.truth.type(torch.cuda.FloatTensor) * self.atten_score #[bsz T k]
        out = torch.prod(1-seq_prob[0,:inputLen[0],:])
        for i in range(1,len(inputLen)):
            out = out+torch.prod(1-seq_prob[i,:inputLen[i],:])
        #seq_prob = torch.sum(seq_prob,dim=2) #[bsz T k] -->
        #loss = torch.sum(torch.prod((1-seq_prob), dim=1))
        return out


class ProjectClassAtt(nn.Module):
    def __init__(self, config):
        super(ProjectClassAtt, self).__init__()
       
        self.u = int(config.get("setting_CTXEncoder","hidden_lstm"))
        if config.get("experimentVer","ctx_encoder")=="BERT":
            classInpDim = 768
        else:
            classInpDim = 2*self.u
        
        self.transformU = nn.Sequential(
            nn.BatchNorm1d(4*self.u),
            nn.Linear(4*self.u, 2*self.u),
            nn.ReLU(),
            nn.Linear(2*self.u, 2*self.u)
        )
        
        self.transformC = nn.Sequential(
            nn.BatchNorm1d(classInpDim),
            nn.Linear(classInpDim, 2 * self.u),
            nn.ReLU(),
            nn.Linear(2 * self.u, 2 * self.u)
        )
         
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.CE = torch.nn.CrossEntropyLoss()
        
        self.softmax = nn.Softmax(dim=1)

        self.WA = nn.Sequential(
            nn.Linear(4 * self.u, 35),
            nn.ReLU(),
            nn.Linear(35, 1)
        )
        
    def forward(self, Ux, Hy, truthY):
        #Ux : joint utterance and slot encoding, [BSZ, T, 4u]
        #Hy : intent encoding, [noY,2u]
        
        '''
        self.max = nn.MaxPool1d(Ux.shape[1])
        self.maxUx = self.max(Ux.transpose(1,2)).squeeze(2) #[BSZ, 2u]
        '''
        AI = self.softmax(self.WA(Ux)).transpose(1,2) #[BSZ 1 T]
        self.attUx = torch.matmul(AI,Ux).squeeze(1) #[BSZ 1 T] [BSZ T 4u] ==> [BSZ 1 4u]
        
        self.transHx = self.transformU(self.attUx) #[BSZ,2u]
        matchHx = self.transHx.unsqueeze(2).repeat(1,1,Hy.shape[0]) #[BSZ, 2u, noY]
      
        self.transHy = self.transformC(Hy) # [noY,2u]
        matchHy = self.transHy.unsqueeze(0).repeat(Ux.shape[0],1,1).transpose(1,2) #[BSZ,2u, noY]
        
  
        self.score = self.cos(matchHx,matchHy)
        self.pred = torch.argmax(self.score,dim=1)
        self.truth = truthY
   
    def lossCE(self):
        truthYid = torch.argmax(self.truth, dim=1)
        loss = self.CE(self.score, truthYid)
        #print('loss',loss)
        return loss    
    
    def lossHinge(self,gamma=1):
        indTruth = torch.argmax(self.truth, dim=1)
        batchSize = indTruth.shape[0]
        scoreTrue = self.score[range(batchSize), indTruth]
        scoreOtherTemp, indOtherTemp = torch.topk(self.score,2)
        correctFlag = torch.eq(indTruth, indOtherTemp[:,0])
        scoreOther = scoreOtherTemp[range(batchSize), list(correctFlag)]
        margin = gamma - scoreTrue + scoreOther
        outp = F.relu(margin)

        return outp.mean()

class SlotGate(nn.Module):
    def __init__(self, config):
        super(SlotGate, self).__init__()

        self.u = int(config.get("setting_CTXEncoder", "hidden_lstm"))

        self.W = nn.Sequential(
            nn.Linear(2 * self.u,33),
            nn.ReLU(),
            nn.Linear(33, 1)
        )

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, Atten, Hx, Hs):
        #Atten: [BSZ T K] : before softmax
        #Hx : [BSZ T 2u]
        #Hs : [K 2u]
        if(Atten.shape[3]==1):
            Atten = Atten.squeeze(3)

        Bs = nn.Softmax(dim=1)(Atten) #[BSZ T K]
        As = nn.Softmax(dim=2) (Atten) #atten_score #[BSZ T K]

        Ai = nn.Softmax(dim=1)(self.W(Hx)) #[Bsz T 1]
        matchAi = Ai.repeat(1,1,Bs.shape[2]) #[Bsz T K]

        self.gate = self.cos(Bs, matchAi).unsqueeze(1) #[BSZ 1 K]
        self.gateHs = torch.sum(torch.matmul(self.gate * As,Hs),dim=1) #[BSZ 2u]
        self.outputU = torch.cat((torch.matmul(Ai.transpose(1,2), Hx).squeeze(1), self.gateHs), dim = 1) #[BSZ 4u]



class ProjectClass(nn.Module):
    def __init__(self, config, isSeq=False):
        super(ProjectClass, self).__init__()

        self.u = int(config.get("setting_CTXEncoder","hidden_lstm"))
        self.isSeq = isSeq
        if config.get("experimentVer","ctx_encoder")=="BERT":
            classInpDim = 768
        else:
            classInpDim = 2*self.u

        self.transformU = nn.Sequential(
            nn.BatchNorm1d(4*self.u),
            nn.Linear(4*self.u, 2*self.u),
            nn.ReLU(),
            nn.Linear(2*self.u, 2*self.u)
        )
        self.transformC = nn.Sequential(
            nn.BatchNorm1d(classInpDim),
            nn.Linear(classInpDim, 2 * self.u),
            nn.ReLU(),
            nn.Linear(2 * self.u, 2 * self.u)
        )
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, Hx, Hy, truthY):
        """
        project utterance and class to a same space
        :param Hx: [BSZ, 2u]
        :param Hy: [noY, dimY]
        :param indY: [BSZ, noY]
        :return: score[BSZ, noY]
        """
        self.transHx = self.transformU(Hx) # [BSZ, 2u]
        matchHx = self.transHx.unsqueeze(2).repeat(1,1,Hy.shape[0]) # [BSZ, 2u, noY]
        self.transHy = self.transformC(Hy) # [2u, dimY]
        transHy = self.transHy.transpose(0,1)
        matchHy = transHy.unsqueeze(0).repeat(Hx.shape[0],1,1) # [BSZ, 2u, noY]

        # tsne
        # truthID = torch.argmax(truthY,dim=1)
        # tsneBoth(self.transHx, transHy.transpose(0,1),truthID)

        self.score = self.cos(matchHx,matchHy) # [BSZ, noY]
        if self.isSeq:
            self.score[:,0] = -1
        self.pred = torch.argmax(self.score,dim=1)
        self.truth = truthY

    def loss(self):
        outp = self.truth.float().mul(self.score).sum(dim=1).mean()
        return -outp

    def lossHinge(self,gamma=1):
        indTruth = torch.argmax(self.truth, dim=1)
        batchSize = indTruth.shape[0]
        scoreTrue = self.score[range(batchSize), indTruth]
        scoreOtherTemp, indOtherTemp = torch.topk(self.score,2)
        correctFlag = torch.eq(indTruth, indOtherTemp[:,0])
        scoreOther = scoreOtherTemp[range(batchSize), list(correctFlag.int())]
        margin = gamma - scoreTrue + scoreOther
        outp = F.relu(margin)

        return outp.mean()

    def lossWorked(self,margin=0.5):
        coefTrue = self.truth.float()
        noY = coefTrue.shape[1]
        coefFalse = coefTrue-1
        rstTrue = coefTrue.mul(self.score)
        rstFalse = coefFalse.mul(self.score)
        outp = ((noY-1)*rstTrue+margin+rstFalse).sum(dim=1).mean()
        return -outp

    def lossCE(self):
        truthYid = torch.argmax(self.truth, dim=1)
        return self.CE(self.score, truthYid)
