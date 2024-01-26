import random

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# import pickle
import h5py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class MIL_dataloader():
    def __init__(self, data_path, label_path, train=True):

        if train:
            X_train, X_test = train_test_split(data_path, test_size=0.2, random_state=66)  # 10% validation

            traindataset = MIL_dataset(data_path=X_train,label_path=label_path,train=train)

            traindataloader = DataLoader(traindataset, batch_size=1, shuffle=True, num_workers=4)

            valdataset = MIL_dataset(data_path=X_test,label_path=label_path, train=False)

            valdataloader = DataLoader(valdataset, batch_size=1, shuffle=False, num_workers=4)

            self.dataloader = [traindataloader, valdataloader]

        else:
            testdataset = MIL_dataset(data_path=data_path, label_path=label_path, train=False)
            testloader = DataLoader(testdataset, batch_size=1, num_workers=4)

            self.dataloader = testloader

    def get_loader(self):
        return self.dataloader

class MIL_dataset(Dataset):
    def __init__(self, data_path, label_path,train=True):
        """
        Give npz file path
        :param list_path:
        """

        self.data_path=data_path
        self.random = train
        self.label_path=label_path

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):

        fea_path = self.data_path[idx]
        # for pickle file
        # with open (fea_path,'rb')as f:
        #     raw_data=pickle.load(f)
        # f.close()
        #
        # patch_data_list= raw_data['patch_data_list']
        # label=raw_data["label"]
        # featGroup = []
        # for tpatch in patch_data_list:
        #     tfeat=tpatch['feature'].astype(np.float32)
        #     tfeat= torch.from_numpy(tfeat)
        #     featGroup.append(tfeat.unsqueeze(0))
        #     # location
        # featGroup = torch.cat(featGroup, dim=0)  ## numPatch x fs
        # # the input of torch.LongTensor() must be a list
        # label=torch.LongTensor([label])
        #
        # return featGroup,label
        with h5py.File(fea_path,'r') as f:
            raw_feature=f['features']
            fea=np.asarray(raw_feature)
        f.close()
        featGroup=torch.from_numpy(fea)
        # label
        df= pd.read_csv(self.label_path)
            # location
        _,name=os.path.split(fea_path)
        label=int(df[df['id'] == name]['label'])
        # the input of torch.LongTensor() must be a list
        label=torch.LongTensor([label])

        return featGroup,label