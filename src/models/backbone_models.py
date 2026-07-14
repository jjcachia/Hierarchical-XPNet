import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch_3d import EfficientNet3D

############################################################################################################################################################################
############################################################################ Base CNN Networks #############################################################################
############################################################################################################################################################################

class denseNet121(nn.Module):
    """ DenseNet121-based Feature Extractor for feature extraction.
        Total number of parameters:  6953856 (6.95 million)
        Returns a feature map of size 1024x3x3. """
    def __init__(self, weights='DEFAULT', common_channel_size=None):
        """ 
        Initializes the denseNet121 class. 
            
        Args:
            weights (str): The weights to use for the DenseNet121 features. Default is 'DEFAULT'.
            common_channel_size (int): The size of the common channel. Default is None.
        """
        super(denseNet121, self).__init__()
        densenet = models.densenet121(weights=weights)
        self.features = densenet.features  

    def forward(self, x):
        return self.features(x)
    
    #def get_output_channels(self):
    #    """ Returns the number of output channels from the final convolutional layer. """
    #    final_bn_layer = [layer for layer in self.features.modules() if isinstance(layer, nn.BatchNorm2d)][-1]
    #    final_conv_layer = [layer for layer in self.features.modules() if isinstance(layer, nn.Conv2d)][-1]
    #    return final_bn_layer.weight.shape[0]
    
    def get_output_dims(self):
        return 1024, 3, 3
    
    def conv_info(self):
        """ Returns a list of dicts containing kernel sizes, strides, and paddings for each convolutional layer. """
        conv_layers = [layer for layer in self.features.modules() if isinstance(layer, nn.Conv2d)]
        kernel_sizes, strides, paddings = [], [], []
        for conv in conv_layers:
            kernel_sizes.append(conv.kernel_size[0])
            strides.append(conv.stride[0])
            paddings.append(conv.padding[0])
        return kernel_sizes, strides, paddings

class denseNet201(nn.Module):
    """ DenseNet201-based Feature Extractor for feature extraction.
        Total number of parameters:  18092928 (18.09 million) 
        Returns a feature map of size 1920x3x3. """
    def __init__(self, weights='DEFAULT', common_channel_size=None):
        """ 
        Initializes the denseNet201 class.
        
        Args:
            weights (str): The weights to use for the DenseNet201 features. Default is 'DEFAULT'.
            common_channel_size (int): The size of the common channel. Default is None.
        """
        super(denseNet201, self).__init__()
        densenet = models.densenet201(weights=weights)
        self.features = densenet.features

    def forward(self, x):
        return self.features(x)
    
    def get_output_dims(self):
        """ Returns the number of output channels from the final convolutional layer. """
        return 1920, 3, 3
    
    def conv_info(self):
        """ Returns a list of dicts containing kernel sizes, strides, and paddings for each convolutional layer. """
        conv_layers = [layer for layer in self.features.modules() if isinstance(layer, nn.Conv2d)]
        kernel_sizes, strides, paddings = [], [], []
        for conv in conv_layers:
            kernel_sizes.append(conv.kernel_size[0])
            strides.append(conv.stride[0])
            paddings.append(conv.padding[0])
        return kernel_sizes, strides, paddings
    
class resNet34(nn.Module):
    """ ResNet34-based Feature Extractor for feature extraction.
        Total number of parameters:  21797672 (21.80 million) 
        Returns a feature map of size 512x7x7. """
    def __init__(self, weights='DEFAULT', common_channel_size=None):
        super(resNet34, self).__init__()
        resnet = models.resnet34(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.features(x)
    
    def get_output_dims(self):
        """ Returns the number of output channels from the final convolutional layer. """
        return 512, 4, 4

class resNet152(nn.Module):
    """ ResNet152-based Feature Extractor for feature extraction.
        Total number of parameters:  60192808 (60.19 million) 
        Returns a feature map of size 2048x7x7. """
    def __init__(self, weights='DEFAULT', common_channel_size=None):
        super(resNet152, self).__init__()
        resnet = models.resnet152(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.features(x)
    
    def get_output_dims(self):
        """ Returns the number of output channels from the final convolutional layer. """
        return 2048, 4, 4

class vgg16(nn.Module):
    """ VGG16-based Feature Extractor for feature extraction.
        Total number of parameters:  138357544 (138.36 million) 
        Returns a feature map of size 512x7x7. """
    def __init__(self, weights='DEFAULT', common_channel_size=None):
        super(vgg16, self).__init__()
        vgg = models.vgg16(weights=weights)
        self.features = nn.Sequential(*list(vgg.children())[:-1])

    def forward(self, x):
        return self.features(x)
    
    def get_output_dims(self):
        """ Returns the number of output channels from the final convolutional layer. """
        return 512, 7, 7
    
class vgg19(nn.Module):
    """ VGG19-based Feature Extractor for feature extraction.
        Total number of parameters:  143667240 (143.67 million) 
        Returns a feature map of size 512x7x7. """
    def __init__(self, weights='DEFAULT', common_channel_size=None):
        super(vgg19, self).__init__()
        vgg = models.vgg19(weights=weights)
        self.features = nn.Sequential(*list(vgg.children())[:-1])

    def forward(self, x):
        return self.features(x)
    
    def get_output_dims(self):
        """ Returns the number of output channels from the final convolutional layer. """
        return 512, 7, 7

############################################################################################################################################################################
########################################################################## Base 3D CNN Networks ############################################################################
############################################################################################################################################################################ 

class efficientNet3D(nn.Module):
    """EfficientNet3D-based Feature Extractor for feature extraction.
       Returns a feature map of size 1280x4x4."""
    def __init__(self, weights='DEFAULT', common_channel_size=None):
        super(efficientNet3D, self).__init__()
        self.features = EfficientNet3D.from_name("efficientnet-b0", override_params={'include_top': False}, in_channels=1)

    def forward(self, x):
        x = self.features(x)
        return x
    
    def get_output_dims(self):
        """ Returns the number of output channels from the final convolutional layer. """
        dummy_input = torch.randn(1, 1, 128, 128, 128)
        return self.features(dummy_input).shape[1:]
        # return 1280, 2, 2, 2

class CNNBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0 , groups=1, act=True, bn=True, bias=False):
        super(CNNBlock3d, self).__init__()
        self.cnn = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias) #bias set to False as we are using BatchNorm
        
        # if groups = in_channels then it is for Depth wise convolutional; For each channel different Convolutional kernel
        # very limited change in loss but a very high decrease in number of paramteres
        # if groups = 1 : normal_conv kernel of size kernel_size**3

        self.bn = nn.BatchNorm3d(out_channels) if bn else nn.Identity() 
        self.silu = nn.SiLU() if act else nn.Identity() ##SiLU <--> Swish same Thing
        # 1 layer in MBConv doesn't have activation function
    
    def forward(self, x):
        out = self.cnn(x)
        out = self.bn(out)
        out = self.silu(out)
        return out

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),    # input C x H x W --> C x 1 X 1  ONE value of each channel
            nn.Conv3d(in_channels, reduced_dim, kernel_size=1), # expansion
            nn.SiLU(), # activation
            nn.Conv3d(reduced_dim, in_channels, kernel_size=1), # brings it back
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return x*self.se(x)

class StochasticDepth(nn.Module):
    def __init__(self, survival_prob=0.8):
        super(StochasticDepth, self).__init__()
        self.survival_prob =survival_prob
        
    def forward(self, x): #form of dropout , randomly remove some layers not during testing
        if not self.training:
            return x
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, 1, device= x.device) < self.survival_prob # maybe add 1 more here
        return torch.div(x, self.survival_prob) * binary_tensor

class MBConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 expand_ratio = 6, 
                 reduction = 4, #squeeze excitation 1/4 = 0.25
                 survival_prob =0.8 # for stocastic depth
                 ):
        super(MBConv3d, self).__init__()
        
        self.survival_prob = 0.8
        self.use_residual = in_channels == out_channels and stride == 1 # Important if we downsample then we can't use skip connections
        hidden_dim = int(in_channels * expand_ratio)
        self.expand = in_channels != hidden_dim # every first layer in MBConv
        reduced_dim = int(in_channels/reduction)
        self.padding = padding
        
        ##expansion phase

        self.expand = nn.Identity() if (expand_ratio == 1) else CNNBlock3d(in_channels, hidden_dim, kernel_size = 1)
        
        ##Depthwise convolution phase
        self.depthwise_conv = CNNBlock3d(hidden_dim, hidden_dim,
                                        kernel_size = kernel_size, stride = stride, 
                                        padding = padding, groups = hidden_dim
                                       )
        
        # Squeeze Excitation phase
        self.se = SqueezeExcitation(hidden_dim, reduced_dim = reduced_dim)
        
        #output phase
        self.pointwise_conv = CNNBlock3d(hidden_dim, out_channels, kernel_size = 1, stride = 1, act = False, padding = 0)
        # add Sigmoid Activation as mentioned in the paper
        
        # drop connect
        self.drop_layers = StochasticDepth(survival_prob = survival_prob)
    

    
    def forward(self, x):
        
        residual = x
        x = self.expand(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.pointwise_conv(x)
        
        if self.use_residual:  #and self.depthwise_conv.stride[0] == 1:
            x = self.drop_layers(x)
            x += residual
        return x

from math import ceil
class EfficeientNet3d(nn.Module):
    def __init__(self, width_mult=1, depth_mult=1, dropout_rate=0.1, num_classes=2):
        super(EfficeientNet3d, self).__init__()
        last_channels = ceil(512 * width_mult)

        self.first_layer = CNNBlock3d(1, 64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool3d(1, stride=2)
        self.features = self._feature_extractor(width_mult, depth_mult, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels * 3 * 4 * 3, 400),
            nn.Linear(400, 64),
            nn.Linear(64, num_classes),  # Adjust the output size based on the number of classes
        )

    def _feature_extractor(self, width_mult, depth_mult, last_channel):
        # Your previous code for scaling channels and layers

        layers = []
        in_channels = 64  # Initial input channels after the first layer
        final_in_channel = 0 #Initialzse

        # Define configurations for the custom MBConv blocks
        mbconv_configurations = [
            (3, 1, 64, 64, 1),
            (5, 2, 64, 96, 1),
            (5, 2, 96, 128, 2),
            (5, 2, 128, 192, 3),
            (3, 1, 192, 256, 1),
        ]

        for kernel_size, stride, in_channels, out_channels, repeats in mbconv_configurations:
            layers += [
                MBConv3d(in_channels if repeat == 0 else out_channels,
                         out_channels,
                         kernel_size=kernel_size,
                         stride=stride if repeat == 0 else 1,
                         expand_ratio=1,  # Assuming you want expansion factor 1 for these blocks
                         padding=kernel_size // 2
                         )
                for repeat in range(repeats)
            ]
            final_in_channel = out_channels
            print(f'in_channels : {in_channels}, out_channels: {out_channels}, kernelsize : {kernel_size}, stride: {stride}, repeats: {repeats}')
#         print(f'final_in_channels : {final_in_channel}')    
        layers.append(MBConv3d(final_in_channel, last_channel, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        out = self.first_layer(inputs)
        out = self.pool(out)
        x = self.features(out)
        return out

class efficientFPN3D(nn.Module):
    def __init__(self):
        super(efficientFPN3D, self).__init__()
        width_mult = 1
        depth_mult = 1
        efficientnet = EfficeientNet3d()
        self.first_layer = nn.Sequential(
                efficientnet.first_layer,
                efficientnet.pool,
                # features.features
        )
        
        features = efficientnet._feature_extractor(width_mult=width_mult, depth_mult=depth_mult, last_channel=ceil(width_mult * 512))
        self.features = nn.ModuleList([
                    nn.Sequential(*list(features.children())[0:2]), # 96x16x16x16
                    nn.Sequential(*list(features.children())[2:4]), # 128x8x8x8
                    nn.Sequential(*list(features.children())[4:7]), # 192x4x4x4
                    nn.Sequential(*list(features.children())[7:8]), # 256x4x4x4
                    nn.Sequential(*list(features.children())[8:])  # 512x4x4x4
        ])
        
        def forward(self, x):
            x = self.first_layer(x)
            outputs = []
            for feature in self.features:
                x = feature(x)
                outputs.append(x)
            return outputs
        
############################################################################################################################################################################
######################################################################### Feature Pyramid Networks #########################################################################
############################################################################################################################################################################

class denseFPN_121(nn.Module):
    """ DenseNet121-based Feature Pyramid Network (FPN) for feature extraction. 
        Total number of parameters:  8232320 (8.23 million) 
        Returns a feature map of size 256x12x12. """ 
    def __init__(self, weights='DEFAULT', common_channel_size=256):
        """
        Initializes the denseFPN_121 class.

        Args:
            weights (str): The weights to use for the denseFPN_121 features. Default is 'DEFAULT'.
            common_channel_size (int): The size of the FPN common channel. Default is 256.
        """
        super(denseFPN_121, self).__init__()
        original_densenet = models.densenet121(weights=weights)
        self.common_channel_size = common_channel_size
        
        # Initial layers: extract features without modification
        self.encoder = nn.ModuleList([
            nn.Sequential(*list(original_densenet.features.children())[:6], nn.Dropout(0.1)),   # 128x12x12
            nn.Sequential(*list(original_densenet.features.children())[6:8], nn.Dropout(0.2)),  # 256x6x6
            nn.Sequential(*list(original_densenet.features.children())[8:10], nn.Dropout(0.3)), # 896x3x3
            nn.Sequential(*list(original_densenet.features.children())[10:], nn.Dropout(0.3))   # 1920x3x3
        ])
        
        # Define convolutional layers for adapting channel sizes
        fpn_channels = [128, 256, 512, 1024]
        self.adaptation_layers = nn.ModuleDict({
            f'adapt{i+1}': nn.Conv2d(fpn_channels[i], common_channel_size, kernel_size=1)
            for i in range(4)
        })

        # Define FPN layers
        self.fpn = nn.ModuleDict({
            f'fpn{i+1}': nn.Conv2d(common_channel_size, common_channel_size, kernel_size=1)
            for i in range(3)
        })

    def forward(self, x):
        # Encoder
        features = []
        for encoder in self.encoder:
            x = encoder(x)
            features.append(x)
        
        # Merge channels using 1x1 convolutions
        adapted_features = [self.adaptation_layers[f'adapt{i+1}'](features[i]) for i in range(4)]
        
        # FPN integration using top-down pathway
        fpn_output = adapted_features.pop()  # Start with the deepest features
        for i in reversed(range(3)):
            upsampled = F.interpolate(fpn_output, size=adapted_features[i].shape[-2:], mode='nearest')
            fpn_output = self.fpn[f'fpn{i+1}'](upsampled + adapted_features[i])
        
        return fpn_output # 256x12x12
    
    def get_output_dims(self):
        """ Returns the number of output channels from the final convolutional layer. """
        return self.common_channel_size, 12, 12

    def conv_info(self):
        """
        Returns a list of dicts containing kernel sizes, strides, and paddings for each convolutional layer.
        This function will gather information from both the DenseNet backbone and the FPN custom layers.
        """
        kernel_sizes, strides, paddings = [], [], []

        # Traverse the encoder layers which are wrapped in Sequential blocks
        for seq in self.encoder:
            for layer in seq.modules():
                if isinstance(layer, nn.Conv2d):
                    kernel_sizes.append(layer.kernel_size[0])
                    strides.append(layer.stride[0])
                    paddings.append(layer.padding[0])

        # Check adaptation layers
        for layer in self.adaptation_layers.modules():
            if isinstance(layer, nn.Conv2d):
                kernel_sizes.append(layer.kernel_size[0])
                strides.append(layer.stride[0])
                paddings.append(layer.padding[0])

        # FPN layers (each one is a single Conv2d in a ModuleList)
        for layer in self.fpn.modules():
            if isinstance(layer, nn.Conv2d):
                kernel_sizes.append(layer.kernel_size[0])
                strides.append(layer.stride[0])
                paddings.append(layer.padding[0])

        return kernel_sizes, strides, paddings

class denseFPN_201(nn.Module):
    """ DenseNet201-based Feature Pyramid Network (FPN) for feature extraction.
        Total number of parameters:  19697280 (19.70 million) 
        Returns a feature map of size 256x12x12. """ 
    def __init__(self, weights='DEFAULT', common_channel_size=256):
        """
        Initializes the denseFPN_201 class.

        Args:
            weights (str): The weights to use for the EfficientNet V2 Small features. Default is 'DEFAULT'.
            common_channel_size (int): The size of the common channel. Default is 256.
        """
        super(denseFPN_201, self).__init__()
        original_densenet = models.densenet201(weights=weights)
        self.common_channel_size = common_channel_size
        
        # Initial layers: extract features without modification
        self.encoder = nn.ModuleList([
            nn.Sequential(*list(original_densenet.features.children())[:6], nn.Dropout(0.1)),   # 128x12x12
            nn.Sequential(*list(original_densenet.features.children())[6:8], nn.Dropout(0.2)),  # 256x6x6
            nn.Sequential(*list(original_densenet.features.children())[8:10], nn.Dropout(0.4)), # 896x3x3
            nn.Sequential(*list(original_densenet.features.children())[10:], nn.Dropout(0.4))   # 1920x3x3
        ])
        
        # Define convolutional layers for adapting channel sizes
        fpn_channels = [128, 256, 896, 1920]
        self.adaptation_layers = nn.ModuleDict({
            f'adapt{i+1}': nn.Conv2d(fpn_channels[i], common_channel_size, kernel_size=1)
            for i in range(4)
        })

        # Define FPN layers
        self.fpn = nn.ModuleDict({
            f'fpn{i+1}': nn.Conv2d(common_channel_size, common_channel_size, kernel_size=1)
            for i in range(3)
        })

    def forward(self, x):
        # Encoder
        features = []
        for encoder in self.encoder:
            x = encoder(x)
            features.append(x)
        
        # Merge channels using 1x1 convolutions
        adapted_features = [self.adaptation_layers[f'adapt{i+1}'](features[i]) for i in range(4)]
        
        # FPN integration using top-down pathway
        fpn_output = adapted_features.pop()  # Start with the deepest features
        for i in reversed(range(3)):
            upsampled = F.interpolate(fpn_output, size=adapted_features[i].shape[-2:], mode='nearest')
            fpn_output = self.fpn[f'fpn{i+1}'](upsampled + adapted_features[i])
        
        return fpn_output # 256x12x12
    
    def get_output_dims(self):
        """ Returns the number of output channels from the final convolutional layer. """
        # final_conv_layer = [layer for layer in self.fpn[-1].modules() if isinstance(layer, nn.Conv2d)][-1]
        return self.common_channel_size, 12, 12
    
    def conv_info(self):
        """
        Returns a list of dicts containing kernel sizes, strides, and paddings for each convolutional layer.
        This function will gather information from both the DenseNet backbone and the FPN custom layers.
        """
        kernel_sizes, strides, paddings = [], [], []

        # Traverse the encoder layers which are wrapped in Sequential blocks
        for seq in self.encoder:
            for layer in seq.modules():
                if isinstance(layer, nn.Conv2d):
                    kernel_sizes.append(layer.kernel_size[0])
                    strides.append(layer.stride[0])
                    paddings.append(layer.padding[0])

        # Check adaptation layers
        for layer in self.adaptation_layers.modules():
            if isinstance(layer, nn.Conv2d):
                kernel_sizes.append(layer.kernel_size[0])
                strides.append(layer.stride[0])
                paddings.append(layer.padding[0])

        # FPN layers (each one is a single Conv2d in a ModuleList)
        for layer in self.fpn.modules():
            if isinstance(layer, nn.Conv2d):
                kernel_sizes.append(layer.kernel_size[0])
                strides.append(layer.stride[0])
                paddings.append(layer.padding[0])

        return kernel_sizes, strides, paddings


class efficientFPN_v2_s(nn.Module):
    """ EfficientNet V2 Small-based Feature Pyramid Network (FPN) for feature extraction.
        Total number of parameters:  21008208 (21.01 million) 
        Returns a feature map of size 256x25x25. """ 
    def __init__(self, weights='DEFAULT', common_channel_size=256):
        """
        Initializes the efficientFPN_v2_s class.

        Args:
            weights (str): The weights to use for the EfficientNet V2 Small features. Default is 'DEFAULT'.
            common_channel_size (int): The size of the common channel. Default is 256.
        """
        super(efficientFPN_v2_s, self).__init__()
        self.common_channel_size = common_channel_size
        
        # Load EfficientNet V2 Small features
        efficientnet_v2_s = models.efficientnet_v2_s(weights=weights).features[:-1]

        # Modularize encoders
        self.encoder= nn.ModuleList([
            nn.Sequential(*list(efficientnet_v2_s.children())[:2], nn.Dropout(0.1)),    # 24x50x50
            nn.Sequential(*list(efficientnet_v2_s.children())[2:3], nn.Dropout(0.1)),   # 48x25x25
            nn.Sequential(*list(efficientnet_v2_s.children())[3:4], nn.Dropout(0.2)),   # 64x13x13
            nn.Sequential(*list(efficientnet_v2_s.children())[4:5], nn.Dropout(0.2)),   # 128x7x7
            nn.Sequential(*list(efficientnet_v2_s.children())[5:6], nn.Dropout(0.3)),   # 160x7x7
            nn.Sequential(*list(efficientnet_v2_s.children())[6:7], nn.Dropout(0.3))    # 256x4x4
        ])
        
        # Define convolutional layers for adapting channel sizes
        fpn_channels = [24, 48, 64, 128, 160, 256]  # example channel sizes based on architecture details
        self.adaptation_layers = nn.ModuleDict({
            f'adapt{i+1}': nn.Conv2d(fpn_channels[i], common_channel_size, kernel_size=1)
            for i in range(6)
        })

        # Define FPN layers
        self.fpn = nn.ModuleDict({
            f'fpn{i+1}': nn.Conv2d(common_channel_size, common_channel_size, kernel_size=1)
            for i in range(6)
        })

    def forward(self, x):
        # Forward pass through encoders
        features = []
        for encoder in self.encoder:
            x = encoder(x)
            features.append(x)
        
        # Merge channels using 1x1 convolutions
        adapted_features = [self.adaptation_layers[f'adapt{i+1}'](features[i]) for i in range(6)]
        
        # FPN integration using top-down pathway
        fpn_output = adapted_features.pop()  # Start with the deepest features
        for i in reversed(range(0,4)):
            upsampled = F.interpolate(fpn_output, size=adapted_features[i].shape[-2:], mode='nearest')
            fpn_output = self.fpn[f'fpn{i+1}'](upsampled + adapted_features[i])
        
        return fpn_output # 256x25x25
    
    def get_output_dims(self):
        """ Returns the number of output channels from the final convolutional layer. """
        # final_conv_layer = [layer for layer in self.fpn[-1].modules() if isinstance(layer, nn.Conv2d)][-1]
        return self.common_channel_size, 50, 50
    
    def conv_info(self):
        """
        Returns a list of dicts containing kernel sizes, strides, and paddings for each convolutional layer.
        This function will gather information from both the DenseNet backbone and the FPN custom layers.
        """
        kernel_sizes, strides, paddings = [], [], []

        # Traverse the encoder layers which are wrapped in Sequential blocks
        for seq in self.encoder:
            for layer in seq.modules():
                if isinstance(layer, nn.Conv2d):
                    kernel_sizes.append(layer.kernel_size[0])
                    strides.append(layer.stride[0])
                    paddings.append(layer.padding[0])

        # Check adaptation layers
        for layer in self.adaptation_layers.modules():
            if isinstance(layer, nn.Conv2d):
                kernel_sizes.append(layer.kernel_size[0])
                strides.append(layer.stride[0])
                paddings.append(layer.padding[0])

        # FPN layers (each one is a single Conv2d in a ModuleList)
        for layer in self.fpn.modules():
            if isinstance(layer, nn.Conv2d):
                kernel_sizes.append(layer.kernel_size[0])
                strides.append(layer.stride[0])
                paddings.append(layer.padding[0])

        return kernel_sizes, strides, paddings
    

############################################################################################################################################################################
######################################################################## Encoder-Decoder Networks ##########################################################################
############################################################################################################################################################################

class efficientDecoder_v2_s(nn.Module):
    """ EfficientNet V2 Small-based encoder-decoder architecture for feature extraction.
        Total number of parameters:  20971120 (21.00 million) 
        Returns a feature map of size 256x25x25. """ 
    def __init__(self, weights='DEFAULT', common_channel_size=None):
        """
        Initializes the efficientDecoder_v2_s class.

        Args:
            weights (str): The weights to use for the EfficientNet V2 Small features. Default is 'DEFAULT'.
            common_channel_size (int): The size of the common channel. Default is None.
        """
        super(efficientDecoder_v2_s, self).__init__()
        # Load EfficientNet V2 Small features
        efficientnet_v2_s = models.efficientnet_v2_s(weights=weights).features[:-1]

        # Modularize encoders
        self.encoders = nn.ModuleList([
            nn.Sequential(*list(efficientnet_v2_s.children())[:2], nn.Dropout(0.1)),    # 24x50x50
            nn.Sequential(*list(efficientnet_v2_s.children())[2:3], nn.Dropout(0.1)),   # 48x25x25
            nn.Sequential(*list(efficientnet_v2_s.children())[3:4], nn.Dropout(0.2)),   # 64x13x13
            nn.Sequential(*list(efficientnet_v2_s.children())[4:5], nn.Dropout(0.2)),   # 128x7x7
            nn.Sequential(*list(efficientnet_v2_s.children())[5:6], nn.Dropout(0.3)),   # 160x7x7
            nn.Sequential(*list(efficientnet_v2_s.children())[6:7], nn.Dropout(0.3))    # 256x4x4
        ])
        
        # Modularize upconvolutions
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=256, out_channels=160, kernel_size=2, stride=2, padding=1, output_padding=1),
            nn.Conv2d(in_channels=160, out_channels=128, kernel_size=1, stride=1),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=48, kernel_size=2, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=2, stride=2)
        ])
        
        # Modularize decoders
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(160*2, 160, kernel_size=3, padding=1),
                nn.BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                nn.SiLU(inplace=True),
                nn.Dropout(0.3, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128*2, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                nn.SiLU(inplace=True),
                nn.Dropout(0.3, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64*2, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                nn.SiLU(inplace=True),
                nn.Dropout(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(48*2, 48, kernel_size=3, padding=1),
                nn.BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                nn.SiLU(inplace=True),
                nn.Dropout(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(24*2, 24, kernel_size=3, padding=1),
                nn.BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                nn.SiLU(inplace=True),
                nn.Dropout(0.1, inplace=True)
            )
        ])

    def forward(self, x):
        # Encoder
        features = []
        for encoder in self.encoders:
            x = encoder(x)
            features.append(x)
        
        # Decoder
        x = features.pop()
        for upconv, decoder, feature in zip(self.upconvs, self.decoders, reversed(features)):
            x = upconv(x)
            x = torch.cat((x, feature), dim=1)
            x = decoder(x)
        
        return x # 256x25x25
    
    def get_output_channels(self):
        """ Returns the number of output channels from the final convolutional layer. """
        final_conv_layer = [layer for layer in self.decoders[-1].modules() if isinstance(layer, nn.Conv2d)][-1]
        return final_conv_layer.weight.shape[1]
    
