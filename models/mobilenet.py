# models/mobilenet.py

import torch
import torch.nn as nn
from torchvision import models

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        self.base_model.classifier[1] = nn.Linear(self.base_model.last_channel, 1)  # Sortie binaire

    def forward(self, x):
        return self.base_model(x)
