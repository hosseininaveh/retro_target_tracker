import torch
import torch.nn as nn

class CoordRegressor(nn.Module):
    """Directly predicts (x,y) coordinates"""
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 2)  # Direct (x,y) prediction
        )
        
    def forward(self, x):
        return self.head(self.backbone(x))
