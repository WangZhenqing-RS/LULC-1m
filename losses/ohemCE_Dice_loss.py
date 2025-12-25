# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 10:23:33 2023

@author: DELL
"""
import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss,SoftCrossEntropyLoss
from pytorch_toolbelt import losses as L

def ohemCE_Dice_loss(pred, target):
    
    # diceloss在一定程度上可以缓解类别不平衡,但是训练容易不稳定
    DiceLoss_fn = DiceLoss(mode='multiclass', from_logits=True)
    # 交叉熵
    CrossEntropy_fn = nn.CrossEntropyLoss()
    
    loss_ce = CrossEntropy_fn(pred, target)
    loss_dice = DiceLoss_fn(pred, target)

    # OHEM
    loss_ce_, ind = loss_ce.contiguous().view(-1).sort()
    min_value = loss_ce_[int(0.5*loss_ce.numel())]
    loss_ce = loss_ce[loss_ce>=min_value]
    loss_ce = loss_ce.mean()
    loss = loss_ce + loss_dice
    return loss

def ohemBCE_Dice_loss(pred, target):
    
    # diceloss在一定程度上可以缓解类别不平衡,但是训练容易不稳定
    DiceLoss_fn = DiceLoss(mode='binary', from_logits=False)
    # 交叉熵
    BinaryCrossEntropy_fn = nn.BCELoss(reduction="none")
    
    loss_bce = BinaryCrossEntropy_fn(pred, target)
    loss_dice = DiceLoss_fn(pred, target)

    # OHEM
    loss_bce_, ind = loss_bce.contiguous().view(-1).sort()
    min_value = loss_bce_[int(0.5*loss_bce.numel())]
    loss_bce = loss_bce[loss_bce>=min_value]
    loss_bce = loss_bce.mean()
    loss = loss_bce + loss_dice
    return loss

def BCE_Dice_loss(pred, target):
    
    # diceloss在一定程度上可以缓解类别不平衡,但是训练容易不稳定
    DiceLoss_fn = DiceLoss(mode='binary', from_logits=False)
    # 交叉熵
    BinaryCrossEntropy_fn = nn.BCELoss()
    
    loss_bce = BinaryCrossEntropy_fn(pred, target)
    loss_dice = DiceLoss_fn(pred, target)
    
    loss = loss_bce + loss_dice
    return loss

def softCE_Dice_loss(pred, target):
    DiceLoss_fn = DiceLoss(mode='multiclass', from_logits=True)
    SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.01)
    loss_fn = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn,
                              first_weight=0.5, second_weight=0.5).cuda()
    return loss_fn(pred, target)

if __name__ == '__main__':
    target=torch.ones((2,256,256),dtype=torch.long)
    input=(torch.ones((2,9,256,256))*0.9)
    input[0,0,0,0] = 0.99
    loss=ohemCE_Dice_loss(input,target)
    print(loss)