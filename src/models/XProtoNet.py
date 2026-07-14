import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.receptive_field import compute_proto_layer_rf_info_v2
from src.models.ProtoPNet import PPNet, BACKBONE_DICT


class XProtoNet(PPNet):
    def __init__(self, **kwargs):
        super(XProtoNet, self).__init__(**kwargs)

        self.cnn_backbone = self.features
        del self.features
        
        cnn_backbone_out_channels, _, _ = self.cnn_backbone.get_output_dims()
        
        # feature extractor module
        self.add_on_layers = torch.nn.Sequential(*list(self.add_on_layers.children())[:-1])
        # self.add_on_layers_module = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(in_channels=cnn_backbone_out_channels, out_channels=self.prototype_shape[1], kernel_size=1),
        #         nn.BatchNorm2d(self.prototype_shape[1]),
        #         nn.ReLU(),
        #         nn.Dropout(0.2),
        #         nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
        #         nn.Sigmoid()
        #     ) for _ in range(self.num_characteristics)
        # ])
        self._initialize_layer_weights(self.add_on_layers)

        # Occurrence map module
        self.occurrence_module = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=cnn_backbone_out_channels,
                    out_channels=self.prototype_shape[1],
                    kernel_size=1,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=self.prototype_shape[1],
                    out_channels=self.prototype_shape[1] // 2,
                    kernel_size=1,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=self.prototype_shape[1] // 2,
                    out_channels=self.prototypes_per_characteristic,
                    kernel_size=1,
                    bias=False,
                ),
                # nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[0], kernel_size=1, bias=False),
            ) for _ in range(self.num_characteristics)
        ])
        self._initialize_layer_weights(self.occurrence_module)

        # Last classification layer, redefine to initialize randomly
        self.task_specific_classifier = nn.ModuleList([
            nn.Linear(self.prototypes_per_characteristic, self.num_classes, bias=False) for _ in range(self.num_characteristics)   # Apply softmax to get confidence scores for each class of each characteristic
        ])
        
        self.final_add_on_layers = nn.Sequential(
            # nn.Conv2d(in_channels=cnn_backbone_out_channels, out_channels=self.prototype_shape[1], kernel_size=1),
            nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototypes_per_characteristic*self.num_characteristics, kernel_size=2),
            nn.BatchNorm2d(self.prototypes_per_characteristic*self.num_characteristics),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.final_classifier = nn.Sequential(
            # nn.flatten(),
            nn.Linear(2*self.prototypes_per_characteristic*self.num_characteristics, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        self._set_last_layer_incorrect_connection(incorrect_strength=-0.5)

        self.om_softmax = nn.Softmax(dim=-1)
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def forward(self, x):
        # Feature Extractor Layer
        x = self.cnn_backbone(x)
        feature_map = self.add_on_layers(x).unsqueeze(1)  # shape (N, 1, 128, H, W)
        
        # Hierarchical Prototype Layer
        task_logits = []
        similarities = []
        occurrence_maps = []
        for i in range(self.num_characteristics):
            # feature_map = self.add_on_layers_module[i](x).unsqueeze(1)  # shape (N, 1, 128, H, W)
            
            occurrence_map = self.get_occurence_map_absolute_val(x, i)  # shape (N, P, 1, H, W)
            
            features_extracted = (occurrence_map * feature_map).sum(dim=3).sum(dim=3)  # shape (N, P, 128)
            
            similarity = self.cosine_similarity(
                features_extracted, self.prototype_vectors[i].squeeze().unsqueeze(0)
            )  # shape (N, P)
            similarity = (similarity + 1) / 2.0  # normalizing to [0,1] for positive reasoning

            # classification layer
            task_logit = self.task_specific_classifier[i](similarity)
            
            occurrence_maps.append(occurrence_map)
            similarities.append(similarity)
            task_logits.append(task_logit)
        
        # Prepare similarity vector
        similarity_vector = torch.cat(similarities, dim=1)

        # Process through final add-on layers
        final_layer_feature_map = self.final_add_on_layers(feature_map.squeeze()).squeeze()

        # Concatenate all features
        final_layer_input = torch.cat((similarity_vector, final_layer_feature_map), dim=1)
        
        # Final Classification Layer
        final_output = torch.sigmoid(self.final_classifier(final_layer_input))

        return final_output, task_logits, similarities, occurrence_maps

    def compute_occurence_map(self, x, characteristic_index):
        # Feature Extractor Layer
        x = self.cnn_backbone(x)
        occurrence_map = self.get_occurence_map_absolute_val(x, characteristic_index)  # shape (N, P, 1, H, W)
        return occurrence_map

    def get_occurence_map_softmaxed(self, x, characteristic_index):
        occurrence_map = self.occurrence_module[characteristic_index](x)  # shape (N, P, H, W)
        n, p, h, w = occurrence_map.shape
        occurrence_map = occurrence_map.reshape((n, p, -1))
        occurrence_map = self.om_softmax(occurrence_map).reshape((n, p, h, w)).unsqueeze(2)  # shape (N, P, 1, H, W)
        return occurrence_map

    def get_occurence_map_absolute_val(self, x, characteristic_index):
        occurrence_map = self.occurrence_module[characteristic_index](x)  # shape (N, P, H, W)
        occurrence_map = torch.abs(occurrence_map).unsqueeze(2)  # shape (N, P, 1, H, W)
        return occurrence_map
    
    def push_forward(self, x):
        """
        this method is needed for the pushing operation
        """
        # Feature Extractor Layer
        x = self.cnn_backbone(x)
        feature_map = self.add_on_layers(x).unsqueeze(1)  # shape (N, 1, 128, H, W)
        
        features_extracted_list = []
        inverted_similarity_list = []
        occurrence_map_list = []
        preds_list = []
        for characteristic_index in range(self.num_characteristics):
            # feature_map = self.add_on_layers_module[characteristic_index](x).unsqueeze(1)  # shape (N, 1, 128, H, W)
            occurrence_map = self.get_occurence_map_absolute_val(x,characteristic_index)  # shape (N, P, 1, H, W)
            features_extracted = (occurrence_map * feature_map).sum(dim=3).sum(dim=3)  # shape (N, P, 128)

            # Prototype Layer
            similarity = self.cosine_similarity(
                features_extracted, self.prototype_vectors[characteristic_index].squeeze().unsqueeze(0)
            )  # shape (N, P)
            similarity = (similarity + 1) / 2.0  # normalizing to [0,1] for positive reasoning

            # classification layer
            logits = self.task_specific_classifier[characteristic_index](similarity)
            preds = logits.softmax(dim=1)
            
            features_extracted_list.append(features_extracted)
            inverted_similarity_list.append(1-similarity)
            occurrence_map_list.append(occurrence_map)
            preds_list.append(preds)

        # return features_extracted, 1 - similarity, occurrence_map, logits
        return features_extracted_list, inverted_similarity_list, occurrence_map_list, preds_list
    
    def _initialize_layer_weights(self, layer):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
def construct_XPNet(
    base_architecture,
    weights='DEFAULT',
    img_size=100,
    prototype_shape=(10*4*2, 128, 1, 1),
    num_characteristics=4,
    num_classes=2,
    prototype_activation_function="log",
    add_on_layers_type="regular",
):
    features = BACKBONE_DICT[base_architecture](weights=weights)
    # layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    # proto_layer_rf_info = compute_proto_layer_rf_info_v2(
    #     img_size=img_size,
    #     layer_filter_sizes=layer_filter_sizes,
    #     layer_strides=layer_strides,
    #     layer_paddings=layer_paddings,
    #     prototype_kernel_size=prototype_shape[2],
    # )
    return XProtoNet(
        features=features,
        img_size=img_size,
        prototype_shape=prototype_shape,
        # proto_layer_rf_info=proto_layer_rf_info,
        num_characteristics=num_characteristics,
        num_classes=num_classes,
        init_weights=True,
        prototype_activation_function=prototype_activation_function,
        add_on_layers_type=add_on_layers_type,
    )