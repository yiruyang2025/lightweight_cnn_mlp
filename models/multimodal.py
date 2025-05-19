import torch
import torch.nn as nn
from models.cnn import CNN
from models.mlp import MLP

class MultiModalModel(nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()
        self.cnn = CNN()
        self.mlp = MLP()
        self.fc = nn.Linear(10 + 10, 10)

    def forward(self, image, spectrum):
        image_feat = self.cnn(image)
        spectrum_feat = self.mlp(spectrum)
        combined = torch.cat((image_feat, spectrum_feat), dim=1)
        return self.fc(combined)
