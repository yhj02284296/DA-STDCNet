
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import cv2
import numpy as np
import json

def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()
#更换损失函数试试？
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1, reduction='mean', ignore_index=None):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        targets = targets.float()
        inputs = torch.sigmoid(inputs)

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            targets = targets[mask]
            inputs = inputs[mask]

        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()

        tversky = (tp + 1e-10) / (tp + self.alpha * fp + self.beta * fn + 1e-10)
        focal_tversky = (1 - tversky) ** self.gamma

        loss = focal_tversky

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)   #
    ones = torch.sparse.torch.eye(N).cuda()
    ones = ones.index_select(0, label.long())   #
    size.append(N)  #
    return ones.view(*size)

def get_boundary(gtmasks):

    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=gtmasks.device).reshape(1, 1, 3, 3).requires_grad_(False)
    # boundary_logits = boundary_logits.unsqueeze(1)
    boundary_targets = F.conv2d(gtmasks.unsqueeze(1), laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0
    return boundary_targets


class DetailAggregateLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DetailAggregateLoss, self).__init__()
        
        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)
        
        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6./10], [3./10], [1./10]],
            dtype=torch.float32).reshape(1, 3, 1, 1).type(torch.cuda.FloatTensor))
        self.ce_loss = nn.BCEWithLogitsLoss()
        self.ft_loss = FocalTverskyLoss()

    def forward(self, boundary_logits, gtmasks):

        # boundary_logits = boundary_logits.unsqueeze(1)
        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)
        
        boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=8, padding=1)
        boundary_targets_x8 = boundary_targets_x8.clamp(min=0)
    
        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')
        
        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0
        
        
        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0
       
        
        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0
        
        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim=1)
        
        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)

        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0
        
        
        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)
        # Focal Tversky Loss
        alpha = 0.5
        gamma = 0.5
        prob = torch.sigmoid(boundary_logits)
        tp = torch.sum(boudary_targets_pyramid * prob, dim=(2, 3))
        fp = torch.sum((1 - boudary_targets_pyramid) * prob, dim=(2, 3))
        fn = torch.sum(boudary_targets_pyramid * (1 - prob), dim=(2, 3))
        tversky = (tp + 1e-10) / (tp + alpha * fp + (1 - alpha) * fn + 1e-10)
        focal_tversky = torch.pow((1 - tversky), gamma)
        ft_loss = torch.mean(focal_tversky)

        # 二元交叉熵损失
        ce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid, reduction='mean')

        # 总损失
        total_loss = ce_loss + ft_loss

        return  ce_loss, ft_loss

        # bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid)#两类，容易不均衡
        # dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boudary_targets_pyramid)#dice系数，度量集合的相识度
        # return bce_loss,  dice_loss

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
                nowd_params += list(module.parameters())
        return nowd_params

if __name__ == '__main__':
    torch.manual_seed(15)
    with open('../cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
    lb_map = {el['id']: el['trainId'] for el in labels_info}

    img_path = 'E:/datasets/CityScapes/gtFine/val/frankfurt/frankfurt_000001_037705_gtFine_labelIds.png'
    img = cv2.imread(img_path, 0)
 
    label = np.zeros(img.shape, np.uint8)
    for k, v in lb_map.items():
        label[img == k] = v

    img_tensor = torch.from_numpy(label).cuda()
    img_tensor = torch.unsqueeze(img_tensor, 0).type(torch.cuda.FloatTensor)
   #temp to print img_tensor?

    detailAggregateLoss = DetailAggregateLoss()
    for param in detailAggregateLoss.parameters():
        print(param)

    bce_loss,  dice_loss = detailAggregateLoss(torch.unsqueeze(img_tensor, 0), img_tensor)
    print(bce_loss,  dice_loss)
