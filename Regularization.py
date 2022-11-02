import torch.nn as nn
import torch

class Loss(nn.Module):
    def __init__(self, args=None):
        super(Loss, self).__init__()
        self.param_loss = nn.CrossEntropyLoss()
    def forward(self, output, target):
        tloss = self.param_loss(output, target)
        return tloss



class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * \
            targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss