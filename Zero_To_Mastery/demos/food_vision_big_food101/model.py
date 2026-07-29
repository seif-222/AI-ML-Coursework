#making the model.py folder
import torchvision
import torch

from torch import nn

def create_eff_b2_model (num_classes=3):#this 3 is a default that we can change it to 101 when we need to
    eff_b2_model_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT

    eff_b2_model_transform = eff_b2_model_weights.transforms()

    eff_b2_model = torchvision.models.efficientnet_b2(weights=eff_b2_model_weights)

    for param in eff_b2_model.parameters():
        param.requires_grad = False

    eff_b2_model.classifier = nn.Sequential(
    nn.Dropout(p=0.3,inplace=True),
    nn.Linear(in_features=1408, out_features=num_classes))

    return eff_b2_model , eff_b2_model_transform
