# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 18:09:25 2018

@author: ThangPD
"""

import use_alternate_backend
import matplotlib
if use_alternate_backend.use_alternate==True:
    try:
        matplotlib.use(use_alternate_backend.backend)
        print("Use alternate backend:",use_alternate_backend.backend)
    except:
        print("Cannot use alternate backend:",use_alternate_backend.backend)

import os
import numpy as np
import h5py
import json
import argparse
from matplotlib import pyplot as plt
import multiprocessing

def show_compare_imgs(img_path,img_arr1,img_arr2):
    plt.figure()

    plt.subplot(1,2,1)
    plt.imshow(img_arr1,cmap='gray')
    plt.title('Original image')
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.imshow(img_arr2,cmap='gray')
    plt.title('PRNU corrected image')
    plt.colorbar()

    plt.suptitle('Image path = '+img_path)
    plt.tight_layout()
    plt.show()

def read_image(img_path,channel_name):
    with h5py.File(img_path, 'r') as hf:
        img_h5 = hf.get(channel_name)
        img_arr = np.array(img_h5)
        img_arr = np.transpose(img_arr)
        return img_arr

def read_json(json_path):
    with open(json_path,'r') as file:
        data = file.read()
        return json.loads(data)

def correct_image_prnu(img_path,ds_path,prnu_path,channel_name):
    img_arr = read_image(img_path,channel_name)
    print("Image shape =",img_arr.shape)

    ds = np.array(read_json(ds_path).get(channel_name))
    prnu = np.array(read_json(prnu_path).get(channel_name))

    print("DS shape =",ds.shape)
    print("PRNU shape =",prnu.shape)

    if img_arr.shape[1] != prnu.shape[0] or img_arr.shape[1] != ds.shape[0]:
        print("Invalid image shape and/or dark signal shape and/or PRNU shape")
        return None

    img_arr_corrected = img_arr-ds
    img_arr_corrected = np.divide(img_arr_corrected,prnu)
    print("Corrected image shape = ",img_arr_corrected.shape)
    
    square_index = img_arr_corrected.shape[1]
    
    if img_arr.shape[0] > square_index:
        img_arr = img_arr[:square_index]
        
    if img_arr_corrected.shape[0] > square_index:
        img_arr_corrected = img_arr_corrected[:square_index]

    job_for_another_core = multiprocessing.Process(target=show_compare_imgs,args=(img_path, img_arr, img_arr_corrected))
    job_for_another_core.start()


if __name__=='__main__':
    """
    4 parameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', help='String path to image file')
    parser.add_argument('ds_path', help='String path to dark signal json file')
    parser.add_argument('prnu_path', help='String path to prnu json file')
    parser.add_argument('channel_name', help='String channel name')
    args = parser.parse_args()
    img_path = args.img_path
    ds_path = args.ds_path
    prnu_path = args.prnu_path
    channel_name = args.channel_name
    correct_image_prnu(img_path,ds_path,prnu_path,channel_name)