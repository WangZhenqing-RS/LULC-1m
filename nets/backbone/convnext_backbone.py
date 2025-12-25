# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 20:16:28 2023

@author: DELL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from torch.nn.parameter import Parameter
from .convnext import convnext_tinier, convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge, LayerNorm

convnext_models = {
    "convnext_tinier": convnext_tinier,
    "convnext_tiny": convnext_tiny,
    "convnext_small": convnext_small,
    "convnext_base": convnext_base,
    "convnext_large": convnext_large,
    "convnext_xlarge": convnext_xlarge,
}

convnext_dims = {
    "convnext_tinier": [48, 96, 192, 384],
    "convnext_tiny": [96, 192, 384, 768],
    "convnext_small": [96, 192, 384, 768],
    "convnext_base": [128, 256, 512, 1024],
    "convnext_large": [192, 384, 768, 1536],
    "convnext_xlarge": [256, 512, 1024, 2048],
}

class ConvNeXt_Backbone(nn.Module):
    def __init__(self, backbone_name = 'convnext_base', pretrained = True, in_channels = 3, out_indices =[0, 1, 2, 3]):
        super(ConvNeXt_Backbone, self).__init__()
        self.model    = convnext_models[backbone_name](pretrained)
        if in_channels > 3:
            with torch.no_grad():
                pretrained_conv1 = self.model.downsample_layers[0][0].weight.clone()
                self.model.downsample_layers[0][0] = torch.nn.Conv2d(in_channels, 
                                                                     self.model.downsample_layers[0][0].out_channels, 
                                                                     kernel_size=4, 
                                                                     stride=4)
                torch.nn.init.kaiming_normal_(
                self.model.downsample_layers[0][0].weight, mode='fan_out', nonlinearity='relu')
                self.model.downsample_layers[0][0].weight[:, :3] = pretrained_conv1
        if in_channels < 3:
            with torch.no_grad():
                pretrained_conv1 = self.model.downsample_layers[0][0].weight.clone()
                self.model.downsample_layers[0][0] = torch.nn.Conv2d(in_channels, 
                                                                     self.model.downsample_layers[0][0].out_channels, 
                                                                     kernel_size=4, 
                                                                     stride=4)
                torch.nn.init.kaiming_normal_(
                self.model.downsample_layers[0][0].weight, mode='fan_out', nonlinearity='relu')
                self.model.downsample_layers[0][0].weight = Parameter(pretrained_conv1[:, :in_channels])
        
        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(convnext_dims[backbone_name][i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            
        del self.model.norm
        del self.model.head
        

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.model.downsample_layers[i](x)
            x = self.model.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)
    

if __name__ == '__main__':
    model = ConvNeXt_Backbone(pretrained=False, in_channels=2)
    inputs = torch.randn((2,2,512,512))
    outputs = model(inputs)