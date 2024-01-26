"""
This code is adapted from ABMIL.
See details in https://github.com/AMLab-Amsterdam/AttentionDeepMIL.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 1024
        self.D = 256
        self.K = 1

        # self.feature_extractor_part1 = nn.Sequential(
        #     nn.Conv2d(1, 20, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2)
        # )
        #
        # self.feature_extractor_part2 = nn.Sequential(
        #     nn.Linear(50 * 4 * 4, self.L),
        #     nn.ReLU(),
        # )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K,1 ),
            nn.Sigmoid()
        )

    def forward(self, x):
#         x = x.squeeze(0)
        #
        # H = self.feature_extractor_part1(x)
        # H = H.view(-1, 50 * 4 * 4)
        # H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(x)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N   KxN
        # print("A shape",A.shape)
        M = torch.mm(A, x)  # KxL
        # print("M shape",M.shape)
        Y_prob = self.classifier(M)
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, A

    # # AUXILIARY METHODS
    # def calculate_classification_error(self, X, Y):
    #     Y = Y.float()
    #     Y_prob, _ = self.forward(X)
    #     #
    #     # error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()
    #     # print('y_hat',Y_hat)
    #     return Y_prob
    #
    # def calculate_objective(self, X, Y):
    #     Y = Y.float()
    #     Y_prob, A = self.forward(X)
    #     Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
    #     neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
    #
    #     return neg_log_likelihood, A
class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        temp=A_V * A_U
        A = self.attention_weights(temp) # NxK
        out = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            out1 = F.softmax(out, dim=1)  # softmax over N
            return out1

        return out  ### K x N
class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.L = 1024
        self.D = 128
        self.K = 1

        # self.feature_extractor_part1 = nn.Sequential(
        #     nn.Conv2d(1, 20, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2)
        # )
        #
        # self.feature_extractor_part2 = nn.Sequential(
        #     nn.Linear(50 * 4 * 4, self.L),
        #     nn.ReLU(),
        # )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.squeeze(0)
        #
        # H = self.feature_extractor_part1(x)
        # H = H.view(-1, 50 * 4 * 4)
        # H = self.feature_extractor_part2(H)  # NxL

        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        # print("A shape",A.shape)
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, x)  # KxL

        Y_prob = self.classifier(M)
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, A

    # # AUXILIARY METHODS
    # def calculate_classification_error(self, X, Y):
    #     Y = Y.float()
    #     Y_prob,_= self.forward(X)
    #     # error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
    #
    #     return Y_prob
    #
    # def calculate_objective(self, X, Y):
    #     Y = Y.float()
    #     Y_prob,A = self.forward(X)
    #     Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
    #     neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
    #
    #     return neg_log_likelihood, A
