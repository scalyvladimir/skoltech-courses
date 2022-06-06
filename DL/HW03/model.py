import torch
from torch import nn
from torch.nn import functional as F
import functools
import math
from torch.nn.utils import spectral_norm


class AdaptiveBatchNorm(nn.BatchNorm2d):
    """
    Adaptive batch normalization layer (4 points)

    Args:
        num_features: number of features in batch normalization layer
        embed_features: number of features in embeddings

    The base layer (BatchNorm2d) is applied to "inputs" with affine = False

    After that, the "embeds" are linearly mapped to "gamma" and "bias"
    
    These "gamma" and "bias" are applied to the outputs like in batch normalization
    with affine = True (see definition of batch normalization for reference)
    """
    def __init__(self, num_features: int, embed_features: int):
        super(AdaptiveBatchNorm, self).__init__(num_features, affine=False)
        
        self.gamma_map = nn.Linear(embed_features, num_features, bias=False)
        self.bias_map = nn.Linear(embed_features, num_features, bias=False)

    def forward(self, inputs, embeds):
        gamma = self.gamma_map(embeds)
        bias = self.bias_map(embeds)

        assert gamma.shape[0] == inputs.shape[0] and gamma.shape[1] == inputs.shape[1]
        assert bias.shape[0] == inputs.shape[0] and bias.shape[1] == inputs.shape[1]

        outputs = super().forward(inputs)

        return outputs * gamma[..., None, None] + bias[..., None, None]


class PreActResBlock(nn.Module):
    """
    Pre-activation residual block (6 points)

    Paper: https://arxiv.org/pdf/1603.05027.pdf
    Scheme: materials/preactresblock.png
    Review: https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e

    Args:
        in_channels: input number of channels
        out_channels: output number of channels
        batchnorm: this block is with/without adaptive batch normalization
        upsample: use nearest neighbours upsampling at the beginning
        downsample: use average pooling after the end

    in_channels != out_channels:
        - first conv: in_channels -> out_channels
        - second conv: out_channels -> out_channels
        - use 1x1 conv in skip connection

    in_channels == out_channels: skip connection is without a conv
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int = None,
                 batchnorm: bool = False,
                 upsample: bool = False,
                 downsample: bool = False):
        super(PreActResBlock, self).__init__()
        # TODO: define pre-activation residual block
        # TODO: apply spectral normalization to conv layers
        # Don't forget that activation after residual sum cannot be inplace!
        
        if batchnorm:
            self.ad_bn1 = AdaptiveBatchNorm(in_channels, embed_channels)
            self.ad_bn2 = AdaptiveBatchNorm(out_channels, embed_channels)
            
        if upsample:
            self.up = nn.UpsamplingNearest2d(scale_factor=2)
    
        if downsample:
            self.down = nn.AvgPool2d(kernel_size=2)
        
        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=False),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, padding=1))
        )
        
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=False),
            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, padding=1))
        )
        
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))

    def forward(self, 
                inputs, # regular features 
                embeds=None): # embeds used in adaptive batch norm
        
        inputs = self.up(inputs) if hasattr(self, 'up') else inputs
            
        outputs = self.ad_bn1(inputs, embeds) if hasattr(self, 'ad_bn1') else inputs
        
        outputs = self.conv1(outputs)
            
        outputs = self.ad_bn2(outputs, embeds) if hasattr(self, 'ad_bn2') else outputs
        
        outputs = self.conv2(outputs)
        
        outputs = outputs + self.skip(inputs)
        
        outputs = self.down(outputs) if hasattr(self, 'down') else outputs
        
        return outputs


class Generator(nn.Module):
    """
    Generator network (8 points)
    
    TODO:

      - Implement an option to condition the synthesis on trainable class embeddings
        (use nn.Embedding module with noise_channels as the size of each embed)

      - Concatenate input noise with class embeddings (if use_class_condition = True) to obtain input embeddings

      - Linearly map input embeddings into input tensor with the following dims: max_channels x 4 x 4

      - Forward an input tensor through a convolutional part, 
        which consists of num_blocks PreActResBlocks and performs upsampling by a factor of 2 in each block

      - Each PreActResBlock is additionally conditioned on the input embeddings (via adaptive batch normalization)

      - At the end of the convolutional part apply regular BN, ReLU and Conv as an image prediction head

      - Apply spectral norm to all conv and linear layers (not the embedding layer)

      - Use Sigmoid at the end to map the outputs into an image

    Notes:

      - The last convolutional layer should map min_channels to 3. With each upsampling you should decrease
        the number of channels by a factor of 2

      - Class embeddings are only used and trained if use_class_condition = True
    """    
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 noise_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_class_condition: bool):
        super().__init__()
        
        self.output_size = 4 * 2**num_blocks
        self.min_channels = min_channels
        self.max_channels = max_channels
        
        if use_class_condition:
            self.embeds = nn.Embedding(num_classes, noise_channels)
            noise_channels *= 2
            
        self.linear = spectral_norm(nn.Linear(noise_channels, self.max_channels * 4 * 4))
        
        self.blocks_list = nn.ModuleList()
        curr_channels = self.max_channels
        
        for i in range(num_blocks):
            tmp = PreActResBlock(curr_channels, curr_channels // 2, noise_channels, True, True, False)
            
            self.blocks_list.append(tmp)
            curr_channels //= 2
            
        self.last_layer = nn.Sequential(
            nn.BatchNorm2d(self.min_channels),
            nn.ReLU(inplace=False),
            spectral_norm(nn.Conv2d(self.min_channels, 3, kernel_size=1)),
            nn.Sigmoid()
        )

    def forward(self, noise, labels):
        
        if hasattr(self, 'embeds'):            
            input_embeds = torch.cat([self.embeds(labels), noise], dim=1)
        else:
            input_embeds = noise.clone()
            
        outputs = self.linear(input_embeds)
        outputs = outputs.view(-1, self.max_channels, 4, 4)
        
        for block_layer in self.blocks_list:
            outputs = block_layer(outputs, input_embeds)
        
        outputs = self.last_layer(outputs)

        assert outputs.shape == (noise.shape[0], 3, self.output_size, self.output_size)
        return outputs


class Discriminator(nn.Module):
    """
    Discriminator network (8 points)

    TODO:
    
      - Define a convolutional part of the discriminator similarly to
        the generator blocks, but in the inverse order, with downsampling, and
        without batch normalization
    
      - At the end of the convolutional part apply ReLU and sum pooling
    
    TODO: implement projection discriminator head (https://arxiv.org/abs/1802.05637)
    
    Scheme: materials/prgan.png
    
    Notation:
    
      - phi is a convolutional part of the discriminator
    
      - psi is a vector
    
      - y is a class embedding
    
    Class embeddings matrix is similar to the generator, shape: num_classes x max_channels

    Discriminator outputs a B x 1 matrix of realism scores

    Apply spectral norm for all layers (conv, linear, embedding)
    """
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_projection_head: bool):
        super().__init__()
        
        if use_projection_head:
            self.head = spectral_norm(nn.Embedding(max_channels, 1))
        
        self.blocks_list = nn.ModuleList([PreActResBlock(3, min_channels, None, False, False, True)])
        curr_channels = min_channels
        
        for i in range(num_blocks):
            tmp = PreActResBlock(curr_channels, curr_channels * 2, None, False, False, True)
            curr_channels *= 2
            self.blocks_list.append(tmp)
                
        self.embeds = spectral_norm(nn.Embedding(num_classes, max_channels))

        self.act = nn.ReLU(inplace=False)
        self.last_layer = spectral_norm(nn.Linear(max_channels, 1))

    def forward(self, inputs, labels):
        embeds = self.embeds(labels)
        
        scores = inputs
        
        for block_layer in self.blocks_list:
            scores = block_layer(scores, embeds)
            
        scores = self.act(scores)
        
        scores = torch.sum(scores.view(scores.shape[0], scores.shape[1], -1), dim=2)
        
        scores = self.last_layer(scores)
        
        if hasattr(self, 'head'):
            tmp = torch.mul(self.head(labels), scores)
            scores = scores + torch.sum(tmp, dim=1, keepdim=True)
        
        scores = scores.view((inputs.shape[0],))
        
        assert scores.shape == (inputs.shape[0],)
        return scores


