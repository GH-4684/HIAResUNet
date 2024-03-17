import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


def dice_coeff(input: Tensor, target: Tensor, weight = None, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask

    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    if weight is None:
        weight = torch.ones(input.shape, device=input.device)

    inter = 2 * (weight * input * target).sum(dim=sum_dim)
    sets_sum = (weight * input).sum(dim=sum_dim) + (weight * target).sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, weight = None, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), weight=weight, reduce_batch_first=reduce_batch_first, epsilon=epsilon)


def dice_loss(input: Tensor, target: Tensor, weight: bool = None, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, weight=weight, reduce_batch_first=True)


def dice_crossentropy_loss(model, masks_pred: Tensor, true_masks: Tensor, classes: int, beta: float = 0.5, norm_order=2, weight_decay=1e-4) -> Tensor:
    """ loss = beta*dice + (1-beta)*cross_entropy + weight_decay*L2_loss """
    crossentropy_loss = nn.CrossEntropyLoss() if classes > 1 else nn.BCEWithLogitsLoss()

    if classes == 1:
        loss = (1 - beta) * crossentropy_loss(masks_pred.squeeze(1), true_masks.float())    # masks_pred: 4D, true_masks: 3D
        loss += beta * dice_loss(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
    else:
        loss = (1 - beta) * crossentropy_loss(masks_pred, true_masks)
        loss += beta * dice_loss(
            F.softmax(masks_pred, dim=1).float(),
            F.one_hot(true_masks, classes).permute(0, 3, 1, 2).float(),
            multiclass=True
        )

    # regularization_loss = 0
    # for name, param in model.named_parameters():
    #     if 'weight' in name:
    #         regularization_loss += torch.norm(param, p=norm_order) # p是范数的阶数
    # loss += weight_decay * regularization_loss

    return loss


max_pool = lambda ks: nn.MaxPool2d(kernel_size=ks, stride=1, padding=(ks-1)//2)
dilate = lambda x, ks=3: max_pool(ks)(x.float()).long()
erode = lambda x, ks=3: -max_pool(ks)(-x.float()).long()


def _gaussian_filter(input, sigma=1):
    kernel_size = int(3 * sigma)
    kernel = torch.Tensor([[np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) for x in range(-kernel_size, kernel_size + 1)] for y in range(-kernel_size, kernel_size + 1)])
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel.size(0), kernel.size(1)).to(input.device)

    filtered = F.conv2d(input, kernel, padding=kernel_size)
    return filtered

def _weighted_focal_loss(masks_pred: Tensor, true_masks: Tensor, weight: Tensor) -> Tensor:
    loss = FocalLoss(num_classes=1)(masks_pred, true_masks, weight=weight)
    return loss

def _weighted_dice_loss(masks_pred: Tensor, true_masks: Tensor, weight: Tensor) -> Tensor:
    loss = dice_loss(masks_pred.squeeze(1), true_masks.squeeze(1), weight=weight, multiclass=False)
    assert not loss.isnan()
    return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=2):
        """
        :param alpha:       各类别权重, None则使用相同权重
        :param gamma:       难易样本调节参数. retainnet中设置为2
        :param num_classes: 类别数量
        """
        super(FocalLoss, self).__init__()

        if alpha is None:
            self.alpha = torch.ones(num_classes).reshape(1, -1, 1, 1).float()
        else:
            assert len(alpha) == num_classes
            self.alpha = torch.tensor(alpha).reshape(1, -1, 1, 1).float()
        self.gamma = gamma

    def forward(self, preds, masks, weight=None):
        alpha = self.alpha.to(preds.device)
        if weight is None:
            weight = torch.ones_like(preds)
        weight = weight.float().to(preds.device)

        preds = torch.clamp(preds, min=1e-7, max=1-1e-7)

        loss = - weight * alpha * masks * (1 - preds)**self.gamma * torch.log(preds)
        assert not loss.isnan().sum()
        loss += - weight * alpha * (1 - masks) * preds**self.gamma * torch.log(1 - preds)
        assert not loss.isnan().sum()
        loss = loss.mean()
        return loss

def cml_loss(model, masks_pred: Tensor, true_masks: Tensor, classes: int, beta: float = 0.5, K: float=0.0, norm_order=2, weight_decay=1e-4) -> Tensor:
    if classes != 1:
        raise Exception('classes should be 1')

    masks_pred_ = torch.sigmoid(masks_pred)
    true_masks_ = true_masks.unsqueeze(1).float()
    edge = dilate(true_masks_, ks=3) - erode(true_masks_, ks=3)
    edge_ = (_gaussian_filter(K * edge.float()) + 1) / (K + 1)

    loss = (1 - beta) * _weighted_focal_loss(masks_pred_, true_masks_, weight=edge_)
    loss += beta * _weighted_dice_loss(masks_pred_, true_masks_, weight=edge_)

    return loss
