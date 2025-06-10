import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CenterLoss(nn.Module):
    """Center loss.
    # 参考
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    # 参数
    Args:
        num_classes (int): number of classes. # 类别数
        feat_dim (int): feature dimension.  # 特征维度
    """
    # 初始化        默认参数：类别数为10 特征维度为2 使用GPU
    def __init__(self, num_classes=100, feat_dim=768, smoothing=0.1, temp=1.):
        super(CenterLoss, self).__init__() # 继承父类的所有属性
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.smoothing = smoothing
        self.temp = temp
        self.confidence = 1 - self.smoothing
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        self.classification = nn.Linear(feat_dim, num_classes)

    def forward(self, x, labels): # 前向传播
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim). # 特征矩阵
            labels: ground truth labels with shape (batch_size). # 真实标签
        """

        pred = self.classification(x)

        if self.training:
            batch_size = x.shape[0]
            idx = torch.arange(batch_size)

            # main loss
            pred_cls = torch.log_softmax(pred, dim=-1)
            centers_sim = self.centers.clone().detach()
            x_centers = centers_sim[labels]
            label_sim = torch.matmul(F.normalize(x_centers, dim=-1), F.normalize(centers_sim, dim=-1).T)
            label_sim[idx, labels] = float('-inf')
            # label_sim = label_sim / self.temp
            label_sim = F.softmax(label_sim, dim=-1) * self.smoothing
            label_sim[idx, labels] = self.confidence
            loss_main = torch.mean(torch.sum(-label_sim * pred_cls, dim=-1))

            # center loss
            cos_sim = torch.matmul(F.normalize(x, dim=-1), F.normalize(self.centers, dim=-1).T)
            # cos_sim /= self.temp
            cos_sim = -1 * cos_sim.log_softmax(dim=-1)
            cos_sim_info = cos_sim[idx, labels]
            loss_center = cos_sim_info.clamp(min=1e-12, max=1e+12).sum() / batch_size

        else:
            loss_main = None
            loss_center = None

        return {'gloss_logits': pred,
                'loss_main': loss_main,
                'loss_center': loss_center
                }
