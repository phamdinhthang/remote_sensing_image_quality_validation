# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:30:01 2018

@author: ThangPD
"""

import os
import cv2
import h5py
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def show_img(img_arr):
    print('Image size =',str(img_arr.shape))
    plt.figure()
    plt.imshow(img_arr,cmap='gray')
    plt.colorbar()
    plt.show()

def read_image(img_path,grayscale_only=True):
    img = Image.open(img_path)
    img_arr = np.array(img)

    if grayscale_only==True and len(img_arr.shape)>2:
        img_grey = img.convert('L')
        img_arr = np.array(img_grey)
    return img_arr

def open_hdf5_file(file_path,dataset_name):
    with h5py.File(file_path, 'r') as hf:
        img_h5 = hf.get(dataset_name)
        img_arr = np.array(img_h5)
        img_arr = np.transpose(img_arr)
        return img_arr

def main():
    PAN_folder = 'C:/Users/admin/Desktop/DS_PRNU/DESCENDING/VNREDSAT_1_LEVEL0_97_971523933_23933_PAN_2017-10-31_09-28-28'
    MS_folder = 'C:/Users/admin/Desktop/DS_PRNU/DESCENDING/VNREDSAT_1_LEVEL0_97_971523933_23933_MS_2017-10-31_09-28-28'
    MS_files = ['B1.h5','B2.h5','B3.h5','B4.h5']

    pan_arr = open_hdf5_file(os.path.join(PAN_folder,'PAN.h5'),'PAN')
    show_img(pan_arr)

    for file in MS_files:
        ms_arr = open_hdf5_file(os.path.join(MS_folder,file),file.split('.')[0])
        show_img(ms_arr)

    tiff_path = "C:/Users/admin/Desktop/DS_PRNU/20130904_NinhThuan_19916_1A_Pan/20130904_NinhThuan_19916_1A_Pan/IMAGERY.TIF"
    img_arr = read_image(tiff_path)
    show_img(img_arr)

if __name__=='__main__':
    main()