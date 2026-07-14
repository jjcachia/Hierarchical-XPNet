import torch
import torch.nn as nn
from torchvision import models
import numpy as np

from src.models.backbone_models import denseNet121, denseNet201, resNet34, resNet152, vgg16, vgg19, efficientNet3D

# Dictionary of supported backbone models
BACKBONE_DICT = {
    'denseNet121': denseNet121,
    'denseNet201': denseNet201,
    'resNet34': resNet34,
    'resNet152': resNet152,
    'vgg16': vgg16,
    'vgg19': vgg19,
    'efficientNet3D': efficientNet3D
}


############################################################################################################################################################################
################################################################### Base Model for Malignancy Prediction ###################################################################
############################################################################################################################################################################

class BaseModel(nn.Module):
    def __init__(self, backbone, weights, common_channel_size, hidden_layers, indeterminate=False):
        super(BaseModel, self).__init__()        
        # self.backbone = backbone(weights=weights)
        self.backbone = backbone(weights=weights)
        self.indeterminate = indeterminate
        
        features_dims = np.array(self.backbone.get_output_dims()) # [C, (D), H, W]
        
        self.feature_size = features_dims.size
        if self.feature_size == 3:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=features_dims[0], out_channels=512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        elif self.feature_size == 4:
            self.add_on_layers = nn.Sequential(
                nn.Conv3d(in_channels=features_dims[0], out_channels=512, kernel_size=1),
                nn.BatchNorm3d(512),
                nn.SiLU(),
                nn.Dropout3d(0.2)
            )
        
        out_dims = np.array([512] + list(features_dims[1:])) # [512, (D), H, W]
        num_features = np.prod(out_dims) # 512 * (D) * H * W
        
        num_logits = 1 if not self.indeterminate else 3
        self.final_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, hidden_layers),
            nn.BatchNorm1d(hidden_layers),
            nn.ReLU(),
            nn.Dropout(0.2), # 0.2
            nn.Linear(hidden_layers, 1)
        )
        
    def forward(self, x):
        # Feature Extraction
        x = self.backbone(x)
        
        x = self.add_on_layers(x)
        
        # Final Malignancy Prediction
        if self.indeterminate:
            final_output = self.final_classifier(x)
        else:
            final_output = torch.sigmoid(self.final_classifier(x))
        
        return final_output


def construct_baseModel(backbone_name='denseFPN_121', weights='DEFAULT', common_channel_size=None, hidden_layers=1024, indeterminate=False,
                        num_tasks=None, num_classes=None):
    """
    Constructs a base model for Malignancy Prediction.

    Args:
        backbone_name (str): Name of the backbone model. Default is 'densetFPN_121'.
        weights (str): Weights to be used for the model. Default is 'DEFAULT'.
        input_dim (tuple): Dimensions of the input data. Default is (256, 12, 12).
        hidden_layers (int): Number of hidden layers in the model. Default is 1024.

    Returns:
        BaseModel: The constructed base model.

    Raises:
        ValueError: If the specified backbone_name is not supported.
    """
    if backbone_name not in BACKBONE_DICT:
        raise ValueError(f"Unsupported model name {backbone_name}")
    backbone = BACKBONE_DICT[backbone_name]
    return BaseModel(backbone=backbone, weights=weights, common_channel_size=common_channel_size, hidden_layers=hidden_layers, indeterminate=indeterminate)
