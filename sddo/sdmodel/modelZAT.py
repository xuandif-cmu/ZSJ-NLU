
# class for ZAT and CT

import torch.nn.functional as F

'''
class ZAT(nn.Module):
    def __init__(self,config,dataInfo):
        super(BiLSTMAtt, self).__init__()
    def forward(self,inputX, inputLen, embedding):
'''

class CT(nn.Module):
    
    def __init__(self,config,dataInfo):
        super(CT, self).__init__()
        self.layerLSTM = int(config.get("setting_CT","layer_lstm"))
        self.hiddenSize_1 = int(config.get("setting_CT","hidden_lstm_1"))
        self.hiddenSize_2 = int(config.get("setting_CT","hidden_lstm_2"))
        self.lineardim = int(config.get("setting_CT","linear_dim"))
        self.label = 3 # label candidate: B I O
        
        self.biLSTM_1 = nn.LSTM(self.embSize, self.hiddenSize_1, self.layerLSTM,\
                              bidirectional = True, batch_first = True)
        
        self.biLSTM_2 = nn.LSTM(self.hiddenSize_1*2, self.hiddenSize_2, self.layerLSTM,\
                              bidirectional = True, batch_first = True)
        
        self.linear = nn.Linear(self.hiddenSize_2*2, self.lineardim,  bias = False)
        self.fc =  nn.Linear(self.hiddenSize_2*2, self.label, bias = False)
    
    def forward(self,inputX, inputLen, inputS, embedding):
        
        #inputX: utterance
        #inputS: slot description/label [BSZ, T]
        
        self.wordEmb = nn.Embedding.from_pretrained(embedding)
        self.sLens = inputLen
        embX = self.wordEmb(inputX.transpose(0,1)) # 35 100 300
        embS = self.wordEmb(inputS.transpose(0,1)) # 4 100 300
                
        embPacked = pack_padded_sequence(embX, self.sLens)
        outpH, (hidden, cell) = self.biLSTM_1(embPacked)
        self.outputH = pad_packed_sequence(outpH, total_length=self.maxlenX)[
            0].transpose(0, 1).contiguous()  # [BSZ, T, 2u] # 100 35 256
        
        slot = torch.mean(embS, dim=0) # 100 300
        slot = embS.unsqueeze(1).repeat(1,inputLen,1) # 100 35 300
        
        utterslot = torch.cat((self.outputH, slot),2) # 100 35 556
        utterslot = self.linear(utterslot) # 100 35 128
        
        embXSPacked = pack_padded_sequence(utterslot, self.sLens)
        outpH, (hidden, cell) = self.biLSTM_2(embXSPacked)
        self.outputH_2 = pad_packed_sequence(outpH, total_length=self.maxlenX)[
            0].transpose(0, 1).contiguous()  # [BSZ, T, 2u] # 100 35 256
        
        biolabel = F.softmax(self.fc(self.output_2),dim=2)
          
        return biolabel
        
        
        
        
        
        
        
        