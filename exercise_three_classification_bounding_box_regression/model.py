import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class SimpleYOLO(nn.Module):
    def __init__(self, num_classes):
        super(SimpleYOLO, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.num_classes = num_classes

        # Remove the final classification layer of ResNet
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # For classification

        # Separate heads for each patch
        self.classifiers = nn.ModuleList([nn.Linear(2048, num_classes) for _ in range(4)])
        self.regressors = nn.ModuleList([nn.Linear(2048, 4) for _ in range(4)])  # 4 values for bbox

    def forward(self, x):
        # x shape: (batch_size, 4, C, H, W) 4 patches
        batch_size = x.shape[0]
        features = []

        for i in range(4):
            patch_features = self.backbone(x[:, i, :, :, :])  # (batch_size, C, H', W')
            features.append(patch_features)

        class_outputs = []
        reg_outputs = []

        for i in range(4):
            # Classification head
            class_feat = self.avgpool(features[i]).view(batch_size, -1)  # Global average pooling
            class_output = self.classifiers[i](class_feat)
            class_outputs.append(class_output)

            # Regression head
            reg_output = self.regressors[i](features[i].mean(dim=[2, 3]))
            reg_outputs.append(reg_output)

        return class_outputs, reg_outputs  # Return list of tensors with length = 4
