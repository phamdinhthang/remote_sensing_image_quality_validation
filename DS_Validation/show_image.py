# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 18:06:06 2018

@author: ThangPD
"""

import os
import h5py
import numpy as np
from matplotlib import pyplot as plt

def show_img(img_arr):
    print('Image size =',str(img_arr.shape)) 
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(img_arr,cmap='gray')
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.show()


PAN_folder = 'C:/Users/admin/Desktop/DS_PRNU/ASCENDING/VNREDSAT_1_LEVEL0_97_971523254_23254_PAN_2017-00-258_00-00-00'

with h5py.File(os.path.join(PAN_folder,'PAN.h5'), 'r') as hf:
    img_h5 = hf.get('PAN')
    img = np.array(img_h5)
    img = np.transpose(img)
    show_img(img)
    

MS_folder = 'C:/Users/admin/Desktop/DS_PRNU/ASCENDING/VNREDSAT_1_LEVEL0_97_971523254_23254_MS_2017-00-258_00-00-00'

MS_files = ['B1.h5','B2.h5','B3.h5','B4.h5']
for file in MS_files:
    with h5py.File(os.path.join(MS_folder,file), 'r') as hf:
        img_h5 = hf.get(file.split('.')[0])
        img = np.array(img_h5)
        img = np.transpose(img)
        show_img(img)
