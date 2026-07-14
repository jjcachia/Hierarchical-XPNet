import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.utils.receptive_field import compute_proto_layer_rf_info_v2
from src.models.backbone_models import denseNet121, denseNet201, denseFPN_121, denseFPN_201

# Dictionary of supported backbone models
BACKBONE_DICT = {
    'denseNet121': denseNet121,
    'denseNet201': denseNet201,
    'denseFPN_121': denseFPN_121,
    'denseFPN_201': denseFPN_201,
}

class PPNet(nn.Module):
    def __init__(self, features, img_size, prototype_shape, num_characteristics, num_classes, proto_layer_rf_info=None, init_weights=True, prototype_activation_function='log', add_on_layers_type='bottleneck'):
        super(PPNet, self).__init__()
        # Define the input configurations
        self.img_size = img_size # size of the input images (e.g. (3, 224, 224))
        self.prototype_shape = prototype_shape # shape of the prototype vectors (e.g. (2000, 512, 1, 1))
        self.num_characteristics = num_characteristics # number of characteristics to predict (e.g. shape, margin, etc.)
        self.num_classes = num_classes # binary classification
        
        self.num_prototypes = self.prototype_shape[0] # total number of prototypes
        self.prototypes_per_characteristic = self.num_prototypes // self.num_characteristics # number of prototypes per characteristic
        self.prototypes_per_class = self.prototypes_per_characteristic // self.num_classes # number of prototypes per class
        
        self.proto_layer_rf_info = proto_layer_rf_info
        self.epsilon = 1e-4 # small value to avoid numerical instability

        self.prototype_activation_function = prototype_activation_function # activation function for the prototypes
        
        self.prototype_class_identity = self._get_prototype_class_identity() # class identity of the prototypes
        
        # Define the feature extractor
        self.features = features
        
        # Define the add-on layers
        first_add_on_layer_in_channels, _, _ = features.get_output_dims()
        
        # self.add_on_layers = self.initialize_add_on_layers(first_add_on_layer_in_channels, add_on_layers_type)
        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.BatchNorm2d(self.prototype_shape[1]),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )
        
        # Define separate prototype vectors for each characteristic
        self.prototype_vectors = nn.ParameterList([
            nn.Parameter(torch.rand(self.prototypes_per_characteristic, prototype_shape[1], prototype_shape[2], prototype_shape[3]), requires_grad=True) for _ in range(self.num_characteristics)
        ])
        
        # Define a tensor of ones for the l2-convolution
        self.ones = nn.Parameter(torch.ones(self.prototypes_per_characteristic, prototype_shape[1], prototype_shape[2], prototype_shape[3]), requires_grad=False)

        # Define a separate classifier for each characteristic
        self.task_specific_classifier = nn.ModuleList([
            nn.Linear(self.prototypes_per_characteristic, self.num_classes, bias=False) for _ in range(self.num_characteristics)   # Apply softmax to get confidence scores for each class of each characteristic
        ])
        
        self.final_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_characteristics*self.prototypes_per_characteristic, self.num_characteristics*self.num_classes),
            nn.BatchNorm1d(self.num_characteristics*self.num_classes),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_characteristics*self.num_classes, 1)
        )

        if init_weights:
            self._initialize_weights()
            self._set_last_layer_incorrect_connection(-0.5)
    
    def _get_prototype_class_identity(self):
        """
        Initialize the class identities of the prototypes structured by characteristics.
        Each characteristic has a tensor of size [num_prototypes_per_characteristic, num_classes].
        """
        prototype_class_identity = []
        num_prototypes_per_class = self.prototypes_per_characteristic // self.num_classes
        
        # Create a separate class identity matrix for each characteristic
        for _ in range(self.num_characteristics):
            # Initialize a zero matrix for current characteristic
            class_identity = torch.zeros(self.prototypes_per_characteristic, self.num_classes)
            
            # Assign prototypes to each class (binary: two classes per characteristic)
            for j in range(self.prototypes_per_characteristic):
                class_index = j // num_prototypes_per_class
                class_identity[j, class_index] = 1
            
            prototype_class_identity.append(class_identity)
        
        return prototype_class_identity
    
    def _set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        for i in range(self.num_characteristics):
            positive_one_weights_locations = torch.t(self.prototype_class_identity[i])
            negative_one_weights_locations = 1 - positive_one_weights_locations
            
            correct_class_connection = 1
            incorrect_class_connection = incorrect_strength
            self.task_specific_classifier[i].weight.data.copy_(
                correct_class_connection * positive_one_weights_locations
                + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _l2_convolution(self, x, prototype_vector):
        '''
        apply prototype_vector as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = prototype_vector ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes_per_characteristic,)
        # then we reshape it to (num_prototypes_per_characteristic, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=prototype_vector)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)
        
    def forward(self, x):
        # Extract features using the backbone
        x = self.features(x) # B x 256 x H x W
        
        # Apply add-on layers to the features
        x = self.add_on_layers(x) # B x 512 x H x W
        
        # Compute distances and task logits for each characteristic
        task_logits = []
        # task_probabilities = []
        similarities = []
        min_distances = []
        for i in range(self.num_characteristics):
            distance = self._l2_convolution(x, self.prototype_vectors[i]) # B x num_prototypes_per_characteristic x H x W, B x num_prototypes_per_characteristic x 1 x 1
            min_distance = -F.max_pool2d(-distance, kernel_size=(distance.size()[2], distance.size()[3])) # B x num_prototypes_per_characteristic x 1 x 1
            min_distance = min_distance.view(-1, self.prototypes_per_characteristic) # B x num_prototypes_per_characteristic
            similarity = self.distance_2_similarity(min_distance) # B x num_prototypes_per_characteristic
            
            task_logit = self.task_specific_classifier[i](similarity) # B x 2
            # task_probability = F.softmax(task_logit, dim=1)
                        
            similarities.append(similarity)
            min_distances.append(min_distance)
            task_logits.append(task_logit)
            # task_probabilities.append(task_probability)
        
        # Concatenate task distances for the final classifier
        final_output = torch.sigmoid(self.final_classifier(torch.cat(similarities, dim=1)))
        return final_output, task_logits, min_distances
    
    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        x = self.features(x)
        x = self.add_on_layers(x)
        distances = []
        for i in range(self.num_characteristics):
            distance = self._l2_convolution(x, self.prototype_vectors[i])
            distances.append(distance)
        return x, distances
    
    
    # TODO: Implement the pruning operation


def construct_PPNet(
    base_architecture='denseNet121', 
    weights='DEFAULT', 
    img_size=224,
    prototype_shape=(50*5*2, 224, 1, 1), 
    num_characteristics=5,
    num_classes=2,
    prototype_activation_function='log',
    add_on_layers_type='bottleneck'
):
    
    features = BACKBONE_DICT[base_architecture](weights=weights)
    
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(
        img_size=img_size,  
        layer_filter_sizes=layer_filter_sizes,
        layer_strides=layer_strides,
        layer_paddings=layer_paddings,
        prototype_kernel_size=prototype_shape[2]
    )
    
    return PPNet(
        features=features,
        img_size=img_size,
        prototype_shape=prototype_shape,
        num_characteristics=num_characteristics,
        proto_layer_rf_info=proto_layer_rf_info,
        num_classes=num_classes,
        prototype_activation_function=prototype_activation_function,
        add_on_layers_type=add_on_layers_type,
        init_weights=True
    )
    
    
