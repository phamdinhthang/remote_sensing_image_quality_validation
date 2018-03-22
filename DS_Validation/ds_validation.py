# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:48:35 2018

@author: ThangPD
"""

import os
import h5py
import numpy as np
import visualizer
import data_writer
import argparse

def remove_bright_line(img_arr,dark_threshold=7):
    #filter bright objects. The higher the threshold, the more lines is kept. in real life, threshold is about 3-10
    meanColumn = np.mean(img_arr,axis=1)
    threshold_col = np.full((img_arr.shape[0],),dark_threshold)
    darker = (meanColumn - threshold_col) < dark_threshold

    keptLine = img_arr[darker]
    removed_lines = img_arr.shape[0]-keptLine.shape[0]
    if removed_lines>0: print("    + Remove",removed_lines,"brighter than",dark_threshold,"lsb lines")
    return keptLine

def averaging_DS(img_arr):
    img_arr = remove_bright_line(img_arr)
    meanLine = np.mean(img_arr,axis=0)
    return meanLine.reshape(1,img_arr.shape[1])

def processing_file(folder_path, file_name, block_size):
    image_id = folder_path[-12:-9]
    channel = file_name.split('.')[0]
    with h5py.File(os.path.join(folder_path,file_name), 'r') as hf:
        data=hf[channel]
        original_data_shape = data.shape
        print("----- Processing. Image ID =",image_id,", channel =",channel,"-----")
        print("Image shape (h, w) =",original_data_shape[::-1],", block size =",block_size)
        ncols = original_data_shape[0]
        nrows = original_data_shape[1]

        total_block = int(nrows/block_size)
        meanLine = np.zeros((1,ncols))
        for i in range(total_block+1):
            print("- Processing block",i,"/",total_block)
            if i==total_block:
                arr = data[:,i*block_size:nrows-1]
            else:
                arr = data[:,i*block_size:(i+1)*block_size-1]

            arr = np.transpose(arr)
            meanLine += averaging_DS(arr)/total_block
    return meanLine.reshape(meanLine.shape[1])

def processing_PAN_folder(PAN_folder,block_size=1000):
    PAN_file = 'PAN.h5'
    meanLine = processing_file(PAN_folder,PAN_file,block_size)
    return meanLine

def processing_MS_folder(MS_folder,block_size=250):
    MS_files = ['B1.h5','B2.h5','B3.h5','B4.h5']
    meanLine = {}
    for file in MS_files:
        meanLine[file.split('.')[0]] = processing_file(MS_folder,file,block_size)
    return meanLine

def processing_image(PAN_folder, MS_folder):
    ds_pan = processing_PAN_folder(PAN_folder)
    ds_ms = processing_MS_folder(MS_folder)
    return ds_pan, ds_ms

def average_over_images(arrs):
    vector_len = len(arrs[0])
    average = np.zeros(vector_len)
    for arr in arrs: average+=arr
    average = average/len(arrs)
    return list(average)

def group_data(PAN_DSs,MS_DSs):
    ds_grouped = {}
    PAN_DSs['average'] = average_over_images(list(PAN_DSs.values()))
    for key,meanLine in PAN_DSs.items():
        if not isinstance(meanLine,list): PAN_DSs[key]=list(meanLine)
    ds_grouped['PAN']=PAN_DSs

    MS_channels = {'B1':{},'B2':{},'B3':{},'B4':{}}
    for image_id,meanLine_dic in MS_DSs.items():
        for channel,meanline in meanLine_dic.items():
            MS_channels.get(channel)[image_id] = list(meanline)

    for channel,meanLine_dic in MS_channels.items():
        meanLine_dic['average']=average_over_images(list(meanLine_dic.values()))

    ds_grouped.update(MS_channels)
    return ds_grouped


def calculate_DS(folder_path):
    folders = os.listdir(folder_path)
    image_ids = set([name[-12:-9] for name in folders])

    PAN_DSs = {}
    MS_DSs = {}
    for image_id in image_ids:
        PAN_folder, MS_folder = None, None
        for name in folders:
            if image_id in name and 'PAN' in name: PAN_folder=os.path.join(folder_path,name)
            if image_id in name and 'MS' in name: MS_folder=os.path.join(folder_path,name)
        if PAN_folder is not None and MS_folder is not None:
            ds_pan, ds_ms = processing_image(PAN_folder, MS_folder)
            PAN_DSs[image_id] = ds_pan
            MS_DSs[image_id] = ds_ms

    ds_grouped = group_data(PAN_DSs, MS_DSs)
    visualizer.visualize_channel_data(ds_grouped)

    return ds_grouped

def validate_dark_signal(folderpath):
    ds_grouped = calculate_DS(folderpath)

    result = {}
    for channel,value in ds_grouped.items():
        result[channel]=value.get('average')

    json_path = data_writer.write_data(result)
    return json_path

if __name__=='__main__':
    """
    single parameter: folder path to the folder that contains many ascending acquisition images. Inside each folder, ".RAW" image must be converted to HDF5 dataformat using matlab script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('folderpath', help='String Folderpath')
    args = parser.parse_args()
    folderpath = args.folderpath
    validate_dark_signal(folderpath)