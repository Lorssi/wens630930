import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss 实现，用于解决类别不平衡问题
    
    参数:
        alpha: 类别权重参数，可以是标量或者张量
        gamma: 聚焦参数，值越大，对易分类样本的惩罚越强
        reduction: 'none'|'mean'|'sum' 损失聚合方式
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 聚焦参数
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        计算Focal Loss
        
        参数:
            inputs: 模型输出 [N, C] 其中C是类别数
            targets: 标签 [N]
        """
        # 应用log_softmax获取log概率
        log_prob = F.log_softmax(inputs, dim=-1)
        prob = torch.exp(log_prob)
        
        # 提取目标类别的概率
        targets = targets.view(-1, 1)
        log_pt = log_prob.gather(1, targets)
        pt = prob.gather(1, targets)
        
        # 计算focal权重并应用于log概率
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # 针对每个样本应用特定的alpha
                batch_alpha = self.alpha.gather(0, targets.view(-1))
                batch_alpha = batch_alpha.view(-1, 1)
                focal_weight = batch_alpha * (1 - pt) ** self.gamma
            else:
                # 使用统一的alpha
                focal_weight = self.alpha * (1 - pt) ** self.gamma
        else:
            focal_weight = (1 - pt) ** self.gamma
            
        focal_loss = -focal_weight * log_pt
        
        # 应用reduction
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        else:  # sum
            return focal_loss.sum()


class MultitaskFocalLoss(nn.Module):
    """
    针对多任务学习的Focal Loss
    
    参数:
        alphas: 每个任务的权重列表
        gammas: 每个任务的聚焦参数列表
        task_weights: 每个任务对总损失的贡献权重
    """
    def __init__(self, alphas=None, gammas=None, task_weights=None, num_classes=8):
        super(MultitaskFocalLoss, self).__init__()
        
        # 默认为三个任务设置相同的参数
        if alphas is None:
            self.alphas = [None, None, None]
        else:
            self.alphas = alphas
            
        if gammas is None:
            self.gammas = [2.0, 2.0, 2.0]
        else:
            self.gammas = gammas
            
        if task_weights is None:
            self.task_weights = [1.0/3, 1.0/3, 1.0/3]  # 默认平均权重
        else:
            self.task_weights = task_weights
        
        # 创建三个Focal Loss实例
        self.focal_loss_1 = FocalLoss(alpha=self.alphas[0], gamma=self.gammas[0])
        self.focal_loss_2 = FocalLoss(alpha=self.alphas[1], gamma=self.gammas[1])
        self.focal_loss_3 = FocalLoss(alpha=self.alphas[2], gamma=self.gammas[2])
    
    def forward(self, days_1_7_output, days_1_7, days_8_14_output, days_8_14, days_15_21_output, days_15_21):
        """计算三个任务的加权Focal Loss"""
        loss1 = self.focal_loss_1(days_1_7_output, days_1_7)
        loss2 = self.focal_loss_2(days_8_14_output, days_8_14)
        loss3 = self.focal_loss_3(days_15_21_output, days_15_21)
        
        # 计算加权总损失
        total_loss = self.task_weights[0] * loss1 + \
                     self.task_weights[1] * loss2 + \
                     self.task_weights[2] * loss3
                     
        return total_loss, loss1, loss2, loss3