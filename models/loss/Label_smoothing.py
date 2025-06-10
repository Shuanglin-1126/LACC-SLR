import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0., dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.fc = nn.Linear(768, classes)

    def forward(self, x, target):
        pred = self.fc(x)
        pred_log = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # return torch.mean(torch.sum(-true_dist * pred_log, dim=self.dim))
        return {'gloss_logits': pred,
                'loss_main': torch.mean(torch.sum(-true_dist * pred_log, dim=self.dim)),
                'loss_center': None
                }
