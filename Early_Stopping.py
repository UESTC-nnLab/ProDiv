"""
Early stopping object

from https://github.com/Bjarten/early-stopping-pytorch

"""

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=9, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.counter1=0
        self.best_score = None
        self.best_score1=None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):

        score = -val_loss
        # score1=-val_loss1
        if self.best_score is None:
            self.best_score = score
            # self.best_score1=score1
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        elif score>=self.best_score:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0