
from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
class EarlyStoppingCLAM:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

import argparse
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
# train parameters
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--epoch_step', default='[100]', type=str)
parser.add_argument('--lrP', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--lrM', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--seed', type=int, default=32, metavar='S',
                    help='random seed (default: 32)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--fDim', type=int, default=1024)
parser.add_argument('--test_name', default='clam-mb-attentionP', type=str, help='wandb name')
parser.add_argument('--gpu_ids', default='1', type=str)
parser.add_argument('--grad_clipping', default=5, type=float)
parser.add_argument('--batch_size', default=1, type=int)
# model
parser.add_argument('--model', type=str, default='gated_attention', help='attention/gated_attention/transmil/rnn/clam-sb/clam-mb/dsmil')
# data
parser.add_argument('--data_dir', default='path/h5_files', type=str, help='h5 file dir')
parser.add_argument('--label_file', default='path/id_label.csv', type=str, help='csv file with file-label')
# bag division
parser.add_argument('--cosSplit', default=4, type=int, help='Corresponding to l in our paper')
parser.add_argument('--cosSplit_test', default=4, type=int,help='Same as in training')
parser.add_argument('--numGroup', default=6, type=int,help='number of pseudo-bags')
parser.add_argument('--numGroup_test', default=6, type=int)
# for rnn
parser.add_argument('--s', default=10, type=int, help='how many top k tiles to consider (default: 10,only for modeltype:rnn)')
#  for clam
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
# others
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--Ptype',default='attentionP',type=str,help='type of prototype:meanP/attentionP/random')
parser.add_argument('--proj',default='ablation',type=str,help='')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_ids

# CUDA_LAUNCH_BLOCKING = 1
import json
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import wandb
import random
from torch.autograd import Variable
# import os
from Early_Stopping import EarlyStopping
# from dataloader import MnistBags
from DataLoader import MIL_dataloader
from model.attentionMIL import Attention, GatedAttention
from model.prototype import prototypeModel
from model.RNN import rnn_single
import torch.nn.functional as F
from sklearn.model_selection import KFold
from model.model_clam import CLAM_SB,CLAM_MB
from model.dsmil import  DSMIL
# Training settings
from utils import eval_metric,validate_clam,Accuracy_Logger
from model.TransMIL import TransMIL
CUDA_LAUNCH_BLOCKING=1
print(torch.cuda.get_device_name())
print(torch.cuda.current_device())
print(torch.cuda.device_count())

def train(epoch, modelP, model, modelType, train_loader,  cosSplit, numGroup, lossfunc,optimizer):
    if args.Ptype=='attentionP':
        modelP.train()
    model.train()
    if args.Ptype == 'attentionP':
        train_lossP = 0.
    train_lossM = 0.

    # print("epoch",epoch)
    for batch_idx, (data, labels) in enumerate(train_loader):
        # torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())
        for ii, (tfeat, label) in enumerate(zip(data, labels)):
#             print(tfeat.shape)
            torch.cuda.empty_cache()
            slide_sub_preds = []
            slide_sub_labels = []
            instance_loss = []
            if args.cuda:
                tfeat, label = tfeat.cuda(), label.cuda()
            tfeat, label = Variable(tfeat), Variable(label)
            # print(tfeat.shape)
            # bag division
            # #fea prototype
            if args.Ptype=='random':
                # # # #random split
                feat_index = list(range(tfeat.shape[0]))
                random.shuffle(feat_index)
                index_chunk_list = np.array_split(np.array(feat_index), numGroup)
                index_chunk_list = [sst.tolist() for sst in index_chunk_list]
#                 print('random')
            else:
                if args.Ptype == 'attentionP':
                    fea_prototype, predictionP = modelP(tfeat)
                    fea_prototype = fea_prototype.cpu()
                elif args.Ptype=='meanP':
                    fea_prototype = tfeat.cpu().mean(dim=0)
#                     print('mean')
                # #cosSplit
                cos_simi = F.cosine_similarity(tfeat.cpu(), fea_prototype, dim=-1)
                values, indices = cos_simi.sort(descending=True)
                indices = np.array(indices)
                cos_split_list = np.array_split(indices, cosSplit)
                cos_split_list = [sst.tolist() for sst in cos_split_list]
                j = 0
                for level_list in cos_split_list:
                    random.shuffle(level_list)
                    level_list = np.array_split(level_list, numGroup)
                    level_list = [sst.tolist() for sst in level_list]
                    cos_split_list[j] = level_list
                    j = j + 1

                index_chunk_list = []
                for i in range(0, numGroup):
                    tt_list = []
                    for level_list in cos_split_list:
                        tt_list.extend(level_list[i])
                    # print(tt_list)
                    if modelType=='rnn':
                        tt_list.sort() #remain the order of score
                    # print('ttlist sorted',tt_list)
                    index_chunk_list.append(tt_list)
            tslideLabel = label
            # for each pseudo bag
            
            for tindex in index_chunk_list:
                slide_sub_labels.append(tslideLabel)
                subFeat_tensor = torch.index_select(tfeat, dim=0,
                                                    index=torch.LongTensor(tindex).cuda())
#                 print('shapes',subFeat_tensor.shape)
                if modelType=='rnn':# use only top s instances
                    s = min(args.s, subFeat_tensor.shape[0])
                    subFeat_tensor=subFeat_tensor[:s,:]
                # print('shapes', subFeat_tensor.shape)
                # print(torch.cuda.memory_summary())
                if modelType == 'gated_attention' or modelType == 'attention':
#                     print('sub_featTensor shape',subFeat_tensor.shape)
                    Y_prob, A = model(subFeat_tensor)
                    slide_sub_preds.append(Y_prob)
                elif modelType=='transmil':
                    results_dict = model(data=subFeat_tensor.unsqueeze(0), label=tslideLabel)
                    logits = results_dict['logits']
                    Y_prob = results_dict['Y_prob']
                    slide_sub_preds.append(logits)
                elif modelType=='rnn':
                    state = model.init_hidden(1).cuda()
                    for ss in range(0,s):
                        input = subFeat_tensor[ss,:]
                        # _, input = embedder(input)
                        output, state = model(input, state)
                    # print(output.shape)

                    # Y_prob=F.softmax(output, dim=1)
                    # print(Y_prob.shape)
                    slide_sub_preds.append(output)
                elif modelType=='clam-sb' or modelType=='clam-mb':
#                     print(subFeat_tensor.shape)
                    # print(3)
                    logits, Y_prob, Y_hat, _, instance_dict = model(subFeat_tensor, label=tslideLabel, instance_eval=True)
                    slide_sub_preds.append(logits)
                    instance_loss.append(instance_dict['instance_loss'])
                elif modelType=='dsmil':
                    instance_attn_score, bag_prediction, _, _ = model(subFeat_tensor)
                    slide_sub_preds.append(bag_prediction)
            slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
            slide_sub_labels = torch.cat(slide_sub_labels, dim=0)
            slide_sub_labels = slide_sub_labels.squeeze()
            ## train_loss
            # lossM = torch.FloatTensor().to()
            if modelType == 'gated_attention' or modelType == 'attention':
                lossM=torch.FloatTensor().cuda()
                for (Y_prob, Y) in zip(slide_sub_preds,slide_sub_labels):
                    Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
                    Y = Y.float()
                    t=(-1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob)) ) # negative log bernoulli
                    lossM=torch.cat([lossM,t])
                # print("shape",lossM.shape)
                lossM=lossM.mean()

                train_lossM = train_lossM + lossM.item()
            elif modelType == 'transmil' or modelType=='rnn':
                # train_lossM = train_lossM + lossM.item()
                lossM = lossfunc(slide_sub_preds, slide_sub_labels).mean()
                train_lossM = train_lossM + lossM.item()
            #             lossM = lossfunc(slide_sub_preds, slide_sub_labels).mean()

            elif modelType=='clam-sb' or modelType=='clam-mb':
                
                loss = lossfunc(slide_sub_preds, slide_sub_labels).mean()
                # loss_value = loss.item()
                instance_loss = torch.Tensor(instance_loss)
                instance_loss = instance_loss.mean()
                # instance_loss_value = instance_loss.item()
                lossM = args.bag_weight * loss + (1 - args.bag_weight) * instance_loss
                train_lossM = train_lossM + lossM.item()
            elif modelType=='dsmil':
                max_id = torch.argmax(instance_attn_score[:, 1])
                bag_pred_byMax = instance_attn_score[max_id, :].squeeze(0)
                bag_loss = lossfunc(bag_prediction, label).mean()
                bag_loss_byMax = lossfunc(bag_pred_byMax.unsqueeze(0), label).mean()
                lossM = 0.5 * bag_loss + 0.5 * bag_loss_byMax
                train_lossM=train_lossM+lossM.item()
            # optimization for modelP
            if args.Ptype == 'attentionP':
                #test loss
                lossP = lossfunc(predictionP, label).mean()
                optimizerP.zero_grad()
                # exit()
                lossP.backward()
                torch.nn.utils.clip_grad_norm_(modelP.parameters(), args.grad_clipping)
                optimizerP.step()
                train_lossP = train_lossP + lossP.item()
            # wandb.log({"lossP": lossP.item()})
            
            # wandb.log({"lossM": lossM.item()})
            # reset gradients
            optimizer.zero_grad()

            lossM.backward()

            if modelType == 'gated_attention' or modelType == 'attention':
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clipping)

            optimizer.step()

            # torch.cuda.empty_cache()

            


        # print('1')
    # calculate loss and error for epoch
    if args.Ptype == 'attentionP':
        train_lossP /= len(train_loader)
    train_lossM /= len(train_loader)
    if args.Ptype == 'attentionP':
        print('Epoch: {}, LossP: {:.4f},  LossM: {:.4f}'.format(epoch, train_lossP, train_lossM))
    else:
        print('Epoch: {}, LossM: {:.4f}'.format(epoch, train_lossM))
    return train_lossM


def test(modelP, model, modelType, test_loader, cosSplit, numGroup, lossfunc):
    if args.Ptype == 'attentionP':
        modelP.eval()
    model.eval()
    if args.Ptype == 'attentionP':
        test_lossP = 0.
    test_lossM = 0.
    # ='cuda:1'
    if args.Ptype == 'attentionP':
        gPredP = torch.FloatTensor().cuda()
    gtP = torch.LongTensor().cuda()
    gPredM = torch.FloatTensor().cuda()
    gtM = torch.LongTensor().cuda()

    gPred_mean = torch.FloatTensor().cuda()
    # all_slide_pred=[]
    # print('test')
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            # for each slide
            for ii, (tfeat, label) in enumerate(zip(data, labels)):
                slide_sub_preds = []
                slide_sub_labels = []
                if args.cuda:
                    tfeat, label = tfeat.cuda(), label.cuda()
                tfeat, label = Variable(tfeat), Variable(label)
                # debug
                # print(bag_label)
                # instance_labels = label[1]
                # if args.cuda:
                #     data, label = data.cuda(), label.cuda()
                # data, label = Variable(data), Variable(label)
                # bag division
                if args.Ptype=='random':
                    feat_index = list(range(tfeat.shape[0]))
                    random.shuffle(feat_index)
                    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
                    index_chunk_list = [sst.tolist() for sst in index_chunk_list]
#                     print('random')
                # #prototype
                else:
                    if args.Ptype == 'attentionP':
                        fea_prototype, predictionP = modelP(tfeat)
                        fea_prototype = fea_prototype.cpu()
                    elif args.Ptype == 'meanP':
                        fea_prototype = tfeat.cpu().mean(dim=0)
#                         print('meanP')
                    # #cosSplit
                    cos_simi = F.cosine_similarity(tfeat.cpu(), fea_prototype, dim=-1)
                    values, indices = cos_simi.sort(descending=True)
                    indices = np.array(indices)
                    cos_split_list = np.array_split(indices, cosSplit)
                    cos_split_list = [sst.tolist() for sst in cos_split_list]
                    j = 0
                    for level_list in cos_split_list:
                        random.shuffle(level_list)
                        level_list = np.array_split(level_list, numGroup)
                        level_list = [sst.tolist() for sst in level_list]
                        cos_split_list[j] = level_list
                        j = j + 1
                    index_chunk_list = []
                    for i in range(0, numGroup):
                        tt_list = []
                        for level_list in cos_split_list:
                            tt_list.extend(level_list[i])
                            if modelType == 'rnn':
                                tt_list.sort()  # remain the order of score
                        index_chunk_list.append(tt_list)
                tslideLabel = label
                # for each pseudo bag
                for tindex in index_chunk_list:
                    slide_sub_labels.append(tslideLabel)
                    subFeat_tensor = torch.index_select(tfeat, dim=0,
                                                        index=torch.LongTensor(tindex).cuda())
                    if modelType == 'rnn':  # use only top s instances
                        s = min(args.s, subFeat_tensor.shape[0])
                        subFeat_tensor = subFeat_tensor[:s, :]

                    if modelType == 'gated_attention' or modelType == 'attention':
                        Y_prob, A = model(subFeat_tensor)
                        # print(Y_prob)

                    elif modelType=='transmil':
                        results_dict = model(data=subFeat_tensor.unsqueeze(0), label=tslideLabel)
                        logits = results_dict['logits']
                        Y_prob = results_dict['Y_prob']
                    elif modelType == 'rnn':
                        state = model.init_hidden(1).cuda()
                        for ss in range(0, s):
                            input = subFeat_tensor[ss, :]
                            # _, input = embedder(input)
                            output, state = model(input, state)
                        # print('xx',output.shape)

                        Y_prob = F.softmax(output, dim=1)
                        # print('yy',Y_prob.shape)
                    elif modelType=='clam-sb' or modelType=='clam-mb':
                        logits, Y_prob, Y_hat, _, instance_dict = model(subFeat_tensor, label=tslideLabel,instance_eval=True)
                    elif modelType == 'dsmil':
                        instance_attn_score, Y_prob, _, _ = model(subFeat_tensor)
                        Y_prob = torch.softmax(Y_prob, 1)
                    slide_sub_preds.append(Y_prob)
                slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
                slide_sub_labels = torch.cat(slide_sub_labels, dim=0)
                if args.Ptype == 'attentionP':
                    gPredP = torch.cat([gPredP, predictionP])
                gtP = torch.cat([gtP, label], dim=0)
                # print("gP",gPredP.shape)
                # print("gt",gtP.shape)

                # test loss
                if args.Ptype == 'attentionP':
                    lossP = lossfunc(gPredP, gtP).mean()
                    test_lossP = test_lossP + lossP.item()
                gPredM = torch.cat([gPredM, slide_sub_preds], dim=0)
                gtM = torch.cat([gtM, slide_sub_labels], dim=0)
                lossM = torch.FloatTensor().cuda()
                if modelType == 'gated_attention' or modelType == 'attention':
                    lossM = torch.FloatTensor().cuda()
                    for (Y_prob, Y) in zip(slide_sub_preds, slide_sub_labels):
                        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
                        Y = Y.float()
                        t = (-1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(
                            1. - Y_prob)))  # negative log bernoulli
                        lossM = torch.cat([lossM, t])
#                     print("shape", lossM.shape)
                    lossM = lossM.mean()
                    test_lossM = test_lossM + lossM.item()
                elif modelType == 'transmil' or modelType=='rnn' or modelType=='clam-sb' or modelType=='clam-mb':
                    lossM = lossfunc(slide_sub_preds, slide_sub_labels).mean()
                    test_lossM=test_lossM+lossM.item()
                elif modelType == 'dsmil':
                    max_id = torch.argmax(instance_attn_score[:, 1])
                    bag_pred_byMax = instance_attn_score[max_id, :].squeeze(0)
                    bag_loss = lossfunc(Y_prob, label).mean()
                    bag_loss_byMax = lossfunc(bag_pred_byMax.unsqueeze(0), label).mean()
                    lossM = 0.5 * bag_loss + 0.5 * bag_loss_byMax
                    test_lossM = test_lossM + lossM.item()
                # print('type',type(lossM))
                # print('shape',lossM.shape)

                gSlidePred = torch.mean(slide_sub_preds, dim=0).unsqueeze(0)
                gPred_mean = torch.cat([gPred_mean, gSlidePred], dim=0)
    # if modelType == 'attention' or modelType == 'gated_attention':
    if args.Ptype == 'attentionP':
        gPredP = torch.softmax(gPredP, dim=1)
        gPredP = gPredP[:, -1]
    if modelType == 'attention' or modelType == 'gated_attention':
#         print(gPredM.shape)
        gPredM=gPredM.squeeze()
        gPred_mean=gPred_mean.squeeze()
#         print(gPredM.shape)
    elif modelType=='transmil' or modelType=='rnn' or modelType=='clam-sb' or modelType=='clam-mb' or modelType=='dsmil':
        gPredM=gPredM[:,-1]
        gPred_mean = gPred_mean[:, -1]

    # eval
    if args.Ptype == 'attentionP':
        maccP, mprecP, mrecalP, mspecP, mF1P, aucP = eval_metric(gPredP, gtP)
    maccM, mprecM, mrecalM, mspecM, mF1M, aucM = eval_metric(gPredM, gtM)
#     print(gPred_mean)
#     print(gtP)
    macc_m, mprec_m, mrecal_m, mspec_m, mF1_m, auc_m = eval_metric(gPred_mean, gtP)

    test_lossM /= len(test_loader)
    if args.Ptype == 'attentionP':
        test_lossP /= len(test_loader)
        print(f'  P acc {maccP}, precision {mprecP}, recall {mrecalP}, specificity {mspecP}, F1 {mF1P}, AUC {aucP}')
    print(f'  M acc {maccM}, precision {mprecM}, recall {mrecalM}, specificity {mspecM}, F1 {mF1M}, AUC {aucM}')
    print(f'  combined mean acc  {macc_m}, precision {mprec_m}, recall {mrecal_m}, specificity {mspec_m}, F1 {mF1_m}, AUC {auc_m}')
    if args.Ptype == 'attentionP':
        print('\nTest Set, LossP: {:.4f}, LossM: {:.4f}'.format(test_lossP, test_lossM))
    else:
        print('\nTest Set, LossM: {:.4f}'.format( test_lossM))
    return  lossM, auc_m, mF1_m, macc_m

def setSeed(seed=32):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
if __name__ == "__main__":

    torch.cuda.empty_cache()
    setSeed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    wandb.init(project=args.proj, entity='XXX', name=args.test_name)
    wandb.config = {
        "learning_rate": args.lrM,
        "epochs": args.epochs,
        "fold": 4,
        "val/train": 0.25
    }
    #  = args.
    # torch.manual_seed(args.seed)
    if args.cuda:
        # torch.cuda.manual_seed(args.seed)
        print('\nGPU is ON!')
    epoch_step = json.loads(args.epoch_step)
    print('Load Train and Test Set')
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    df = pd.read_csv(args.label_file)
    stop=False
    train_data = np.array(df['id'])  # np.ndarray()
    slide_name = train_data.tolist()  # list
    pid_ind = range(len(slide_name))
    kf = KFold(n_splits=4, random_state=666, shuffle=True)
    fold = 0
    k_best_auc = []
    # k_best_epoch = []
    k_test_auc = []
    k_best_f1 = []
    k_best_acc = []
    #  = 'cuda:1'
    data_dir = args.data_dir
    for train_index, test_index in kf.split(pid_ind):
        best_auc = 0
        test_auc = 0
        best_acc = 0
        best_f1 = 0
        best_epoch = -1
        # save_dir = os.path.join(model_dir, str(fold) + '-best_model.pth')

        test_name = [slide_name[i] for i in test_index]
        print(f'testingf pid:{len(test_name)}')
        train_val_name = [slide_name[i] for i in train_index]
        test_name_path = [os.path.join(data_dir, each_path) for each_path in test_name]
        train_name_path = [os.path.join(data_dir, each_path) for each_path in train_val_name]
        print(f'trainning length: {len(train_name_path)}')
        print(f'testing length: {len(test_name_path)}')
        # loss function
        ce_cri = torch.nn.CrossEntropyLoss(reduction='none')
        # dataLoader
        Data = MIL_dataloader(data_path=train_name_path, label_path=args.label_file, train=True)
        trainloader, valloader = Data.get_loader()
        TestData = MIL_dataloader(test_name_path, args.label_file, train=False)
        testloader = TestData.get_loader()
        if args.Ptype=='attentionP':
        # prototype model for pseudo bag
            attentionPrototype = prototypeModel(args.fDim)
            attentionPrototype = attentionPrototype.cuda()
            optimizerP = optim.Adam(attentionPrototype.parameters(), lr=args.lrP, betas=(0.9, 0.999), weight_decay=args.reg)
            schedulerP = torch.optim.lr_scheduler.MultiStepLR(optimizerP, epoch_step, gamma=args.lr_decay_ratio)
        else:
            attentionPrototype=None
        # MIL model
        print('Init Model')
        if args.model == 'attention':
            model = Attention()
        elif args.model == 'gated_attention':
            model = GatedAttention()
        elif args.model=='transmil':
            model = TransMIL(2)
        elif args.model=='rnn':
            model=rnn_single(args.fDim)
        elif args.model=='clam-sb':
            instance_loss_fn = torch.nn.CrossEntropyLoss()
            model_dict = {"dropout": args.drop_out, 'n_classes': 2}
            model_dict.update({"size_arg": args.model_size})
            model_dict.update({'subtyping': True})
            if args.B > 0:
                model_dict.update({'k_sample': args.B})
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
            # model.relocate()
        elif args.model=='clam-mb':
            instance_loss_fn = torch.nn.CrossEntropyLoss()
            model_dict = {"dropout": args.drop_out, 'n_classes': 2}
            model_dict.update({"size_arg": args.model_size})
            model_dict.update({'subtyping': True})
            if args.B > 0:
                model_dict.update({'k_sample': args.B})
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
            # model.relocate()
        elif args.model=='dsmil':
            model = DSMIL(input_feat_dim=1024, hid_feat_dim=512, num_classes=2, init=True)
        if args.cuda:
            if args.model=='clam-mb' or args.model=='clam-sb':
                model.relocate()
            else:
                model.cuda()
        early_stoppingM = EarlyStopping(patience=20, verbose=True)
        # optimizer

        if args.model=='clam-sb' or args.model=='clam-mb':
            early_stopping = EarlyStoppingCLAM(patience=25, stop_epoch=50, verbose=True)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lrM, betas=(0.9, 0.999), weight_decay=args.reg)

        print('Start Training')
        for epoch in range(1, args.epochs + 1):
            lossM = train(epoch, attentionPrototype, model, modelType=args.model, train_loader=trainloader,
                                  cosSplit=args.cosSplit, numGroup=args.numGroup, lossfunc=ce_cri,optimizer=optimizer)
            wandb.log({'loss/lossM': lossM})
            print('Start Testing')
            if args.model=='clam-sb' or args.model=='clam-mb':
                stop = validate_clam(fold, epoch, model, valloader, 2,early_stopping, ce_cri, args.results_dir)
            lossM_val, auc_val, f1_val, acc_val = test(attentionPrototype, model, modelType=args.model,
                                                                  test_loader=valloader,
                                                                  cosSplit=args.cosSplit_test,
                                                                  numGroup=args.numGroup_test, lossfunc=ce_cri)
            wandb.log({"auc/auc_val": auc_val, 'loss/M_val': lossM_val})
            lossM_test, auc_test, f1_test, acc_test = test(attentionPrototype, model, modelType=args.model,
                                                                       test_loader=testloader,
                                                                       cosSplit=args.cosSplit_test,
                                                                       numGroup=args.numGroup_test, lossfunc=ce_cri)
            wandb.log({"auc/auc_test": auc_test,'loss/M_test': lossM_test})
            if epoch > int(args.epochs * 0.1):
                if (auc_val > best_auc):
                    best_auc = auc_val
                    test_auc = auc_test
                    best_epoch = epoch
                    best_acc = acc_test
                    best_f1 = f1_test
                    if args.Ptype == 'attentionP':
                        tsave_dict = {
                        'modelP': attentionPrototype.state_dict(),
                        'modelM': model.state_dict()
                    }
                    else:
                        tsave_dict={'modelM':model.state_dict()}
                    save_name = args.test_name + "_" + str(fold) + '_best.pth'
                    torch.save(tsave_dict, save_name)
                    print(
                        f' val auc {best_auc}test auc: {test_auc}, best acc{best_acc},best f1{best_f1}from epoch {best_epoch}')
                print(f' val auc: {auc_val}test auc: {auc_test}, test acc{acc_test},test f1{f1_test}from epoch {epoch}')
            early_stoppingM(lossM)
            if early_stoppingM.early_stop:
                print("early stopping")
                break
            if stop:
                break
        fold += 1
        k_best_auc.append(best_auc)
        k_test_auc.append(test_auc)
        k_best_acc.append(best_acc)
        k_best_f1.append(best_f1)
    sum_auc = 0
    print('final auc result')
    for auc in k_best_auc:
        print(f' {auc}, ')
        sum_auc = sum_auc + auc
    mean_auc = sum_auc / len(k_best_auc)
    print(f' k_mean_auc_val{mean_auc}, ')

    sum_auc_test = 0
    for auc in k_test_auc:
        print(f' {auc}, ')
        sum_auc_test = sum_auc_test + auc
    mean_auc_test = sum_auc_test / len(k_test_auc)
    print(f' k_mean_auc_test{mean_auc_test}, ')

    sum_acc = 0
    for acc in k_best_acc:
        print(f' {acc}, ')
        sum_acc = sum_acc + acc
    mean_acc_test = sum_acc / len(k_best_acc)
    print(f' k_mean_acc_test{mean_acc_test}, ')

    sum_f1 = 0
    for f1 in k_best_f1:
        print(f' {f1}, ')
        sum_f1 = sum_f1 + f1
    mean_f1_test = sum_f1 / len(k_best_f1)
    print(f' k_mean_f1_test{mean_f1_test}, ')
