# utils/early_stopping.py
import numpy as np
import torch
import os

class EarlyStopping:
    """早停法以防止过拟合"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): 在最后一次验证损失改善后等待多少个epoch。
            verbose (bool): 如果为True，则为每次验证损失改善打印一条消息。
            delta (float): 验证损失的最小变化，以符合改善的条件。
            path (str): 保存模型的路径。
            trace_func (function): 用于打印消息的函数。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''保存模型当验证损失下降时。'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # 确保模型保存目录存在
        save_dir = os.path.dirname(self.path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss