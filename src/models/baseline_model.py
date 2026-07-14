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
##### Hierarchical Multi-Task Learning Model with shared feature extraction and task-specific classifiers for nodule characteristics, and final malignancy prediction. #####
############################################################################################################################################################################

class BaselineModel(nn.Module):
    def __init__(self, backbone, weights, hidden_layers, num_tasks, num_classes, indeterminate=False):
        super(BaselineModel, self).__init__()
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        self.indeterminate = indeterminate
        
        self.backbone = backbone(weights=weights)
        
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
        self.task_specific_layers = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_features, hidden_layers),
                nn.BatchNorm1d(hidden_layers),
                nn.ReLU(),
                nn.Dropout(0.2)
            ) for _ in range(num_tasks)
        ])
        
        self.task_specific_classifier = nn.ModuleList([
            nn.Linear(hidden_layers, num_classes) for _ in range(num_tasks)
        ])
        
        num_logits = 1 if not self.indeterminate else 3
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden_layers * num_tasks, hidden_layers),
            nn.BatchNorm1d(hidden_layers),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_layers, num_logits)
        )
        
    def forward(self, x):
        # Feature Extraction
        x = self.backbone(x)
        
        # Add-on Layers
        x = self.add_on_layers(x)
                
        # Process intermediate outputs
        intermediate_outputs = [layer(x) for layer in self.task_specific_layers]
        
        # Concatenate intermediate outputs
        concatenated_outputs = torch.cat(intermediate_outputs, dim=1)
        
        # Nodule Characteristics Prediction
        task_outputs = [self.task_specific_classifier[i](intermediate_outputs[i]) for i in range(len(intermediate_outputs))]
        
        # Final Malignancy Prediction
        if self.indeterminate:
            final_output = self.final_classifier(concatenated_outputs)
        else:
            final_output = torch.sigmoid(self.final_classifier(concatenated_outputs))
                    
        return final_output, task_outputs
    
    
def construct_baselineModel(backbone_name='denseNet121', 
                            weights='DEFAULT', 
                            hidden_layers=256, 
                            num_tasks=5,
                            num_classes=2,
                            indeterminate=False):
    """
    Constructs a Hierarchical Multi-Task Learning Baseline Model.

    Args:
        backbone_name (str): Name of the backbone model.
        weights (str): Weights to initialize the model with. Default is 'DEFAULT'.
        common_channel_size (int): Size of the common channel in the model. Default is 256.
        output_channel_size (int): Size of the output channel in the model. Default is 256.
        output_feature_size (int): Size of the output feature in the model. Default is 25.
        hidden_layers (int): Number of hidden layers in the model. Default is 256.
        num_tasks (int): Number of tasks for the model. Default is 5.

    Returns:
        BaselineModel: The constructed baseline model.

    Raises:
        ValueError: If the specified backbone name is not supported.
    """
    
    if backbone_name not in BACKBONE_DICT:
        raise ValueError(f"Unsupported model name {backbone_name}")
    backbone = BACKBONE_DICT[backbone_name]
    
    return BaselineModel(backbone=backbone, 
                         weights=weights,
                         hidden_layers=hidden_layers, 
                         num_tasks=num_tasks,
                         num_classes=num_classes,
                         indeterminate=indeterminate)

