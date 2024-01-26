"""
This code is adapted from DSMIL.
See details in https://github.com/binli123/dsmil-wsi
"""
import math
import torch.nn as nn
import torch.nn.functional as F
import torch

class DSMIL(nn.Module):
    def __init__(self, input_feat_dim=1024, hid_feat_dim=512, num_classes=2, init=True):
        super(DSMIL, self).__init__()
        self.projecter = nn.Sequential(
            nn.Linear(input_feat_dim, input_feat_dim),
            nn.BatchNorm1d(input_feat_dim)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_feat_dim, hid_feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hid_feat_dim, hid_feat_dim),
            nn.ReLU(inplace=True)
        )

        self.fc_dsmil = nn.Sequential(nn.Linear(hid_feat_dim, 2))
        self.q_dsmil = nn.Linear(hid_feat_dim, hid_feat_dim)
        self.v_dsmil = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(hid_feat_dim, hid_feat_dim)
        )
        self.fcc_dsmil = nn.Conv1d(num_classes, num_classes, kernel_size=hid_feat_dim)

        if init:
            self._initialize_weights()

    def forward(self, x):
        """
        x is with shape of [N, D], where N denotes instance number, and D denotes instance dimension
        """
        # feature embedding at first
        x = x.view(x.size(0), -1)
        x = self.projecter(x) # [N, D]
        x = self.classifier(x) # [N, C], where C denotes embedding dimension

        feat = x
        device = feat.device
        instance_pred = self.fc_dsmil(feat)
        V = self.v_dsmil(feat)
        Q = self.q_dsmil(feat).view(feat.shape[0], -1)
        _, m_indices = torch.sort(instance_pred, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feat, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K
        q_max = self.q_dsmil(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C,
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc_dsmil(B) # 1 x C x 1
        C = C.view(1, -1)
        return instance_pred, C, A, B

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

"""
------------------------------
Pseudo-code for training DSMIL
------------------------------
self.model = DSMIL(input_feat_dim=1024, hid_feat_dim=512, num_classes=2, init=True)
criterion = torch.nn.CrossEntropyLoss()

# feat, label (bag and its label loaded by a dataloader)
self.model.train()
instance_attn_score, bag_prediction, _, _  = self.model(feat)
max_id = torch.argmax(instance_attn_score[:, 1])
bag_pred_byMax = instance_attn_score[max_id, :].squeeze(0)
bag_loss = criterion(bag_prediction, label)
bag_loss_byMax = criterion(bag_pred_byMax.unsqueeze(0), label)
loss = 0.5 * bag_loss + 0.5 * bag_loss_byMax

self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()

bag_prediction = 1.0 * torch.softmax(bag_prediction, dim=1)

--------------------------------
Pseudo-code for evaluating DSMIL
--------------------------------
self.model.eval()
# feat, label (bag and its label loaded by a dataloader)
instance_attn_score, bag_prediction, _, _ = self.model(feat)
bag_prediction = torch.softmax(bag_prediction, 1)
"""