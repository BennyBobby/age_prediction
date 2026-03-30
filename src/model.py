import torch
import torch.nn as nn
from torchvision import models


class AgeEstimator(nn.Module):
    def __init__(self, pretrained=True):
        super(AgeEstimator, self).__init__()
        self.backbone = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        num_features = self.backbone.fc.in_features

        self.backbone.fc = (
            nn.Identity()
        )  # On remplace 'fc' (Fully Connected) par un bloc vide pour supprimer la couche de classification originale
        self.regression_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        age = self.regression_head(features)
        return age


def get_model():
    return AgeEstimator(pretrained=True)
