from model.attentionMIL import Attention_Gated as attentionP
import torch.nn as nn
import torch
class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        out = self.fc(x)
        return out

class prototypeModel(nn.Module):
    def __init__(self,L=1024,D=128,K=1,num_cls=2,droprate=0):
        super(prototypeModel, self).__init__()
        self.attention=attentionP(L,D,K)
        self.classifier=Classifier_1fc(L,num_cls,droprate)
    def forward(self,x):
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x)  ## K x L
        pred = self.classifier(afeat)  ## K x num_cls
        return afeat,pred
