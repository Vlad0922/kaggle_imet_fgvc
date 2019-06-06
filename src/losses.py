import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.epsilon = 1e-10
    
    def forward(self, y_pred, y_true):
        pt = torch.clamp(y_pred * y_true + (1-y_pred) * (1-y_true), self.epsilon, 1-self.epsilon)
        CE = -torch.log(pt)
        FL = torch.pow(1 - pt, self.gamma)*CE
        
        loss = torch.sum(FL, dim=1)
        return torch.mean(loss)
    
class CustomF1Loss(nn.Module):
    def forward(self, predict, target):
        loss = 0
        lack_cls = target.sum(dim=0) == 0
        if lack_cls.any():
            loss += F.binary_cross_entropy(
                predict[:, lack_cls], target[:, lack_cls])
        predict = torch.clamp(predict * (1-target), min=0.01) + predict * target
        tp = predict * target
        tp = tp.sum(dim=0)
        precision = tp / (predict.sum(dim=0) + 1e-8)
        recall = tp / (target.sum(dim=0) + 1e-8)
        f1 = 2 * (precision * recall / (precision + recall + 1e-8))
        return 1 - f1.mean() + loss