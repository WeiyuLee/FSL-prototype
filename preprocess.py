#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:30:43 2017

@author: Weiyu Lee
"""

import numpy as np
import cv2

def horizontal_flip(data):
    # image shape: [batch, 32, 32, 3]  ***[begin:end:step]
    
    data_flip = data[:, :, ::-1, :]
    
    return data_flip

def vertical_flip(data):
    # image shape: [batch, 32, 32, 3]
    
    data_flip = data[:, ::-1, :, :]
    
    return data_flip

def normalize(data, mean=[], std=[]):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    
    print('Data normalization......')
    if mean == []:
        mean = np.mean(data, axis=0)        # mean
    
    if std == []:
        data_n = (data - mean) / 255.
    else:
        data_n = (data - mean[None, None, :]) / std[None, None, :]
    
    return data_n, mean, std

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    output = np.zeros((len(x), 10))
    
    for i, j in enumerate(x):
        output[i,j] = 1
           
    return output

def reshape_image(input, size):
    
    width, height, channel = size
    
    temp_x = np.empty((len(input), width, height, channel))
    
    for i, e in enumerate(input):
        temp_x[i] = cv2.resize(e, (width, height))

    return temp_x