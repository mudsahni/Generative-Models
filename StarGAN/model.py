#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 00:32:22 2019

@author: musahni
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    '''Residual Block with instance normalization.'''
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        layers = []
        #Convolution1
        layers.append(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1,
                               padding=1, bias=False))
        #Normalization
        layers.append(nn.InstanceNorm2d(dim_out, affine=True,
                                         track_running_stats=True))
        #Relu
        layers.append(nn.ReLU(inplace=True))
        
        #Convolution2
        layers.append(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1,
                               padding=1, bias=False))
        #Normalization
        layers.append(nn.InstanceNorm2d(dim_out, affine=True,
                                          track_running_stats=True))
        
    
        self.main = nn.Sequential(*layers)
        
    def forward(self, x):
        return x + self.main(x)
    
    
class Generator(nn.Module):
    '''Generator Network'''
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()
        
        layers = []
        #Convolutional Layer
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1,
                                padding=3, bias=False))
        #Normalization
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True,
                                        track_running_stats=True))
        #relu
        layers.append(nn.ReLU(inplace=True))
        
        #Down Sampling
        curr_dim = conv_dim
        
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, 
                                    stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, 
                                            track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim*2
            
        