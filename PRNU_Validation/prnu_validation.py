# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:08:06 2018

@author: ThangPD
"""

import os
import h5py
import numpy as np
import visualizer
import json
import copy
import math
import read_cpf
import argparse
import data_writer

def averaging_PRNU(img_arr):
    meanLine = np.mean(img_arr,axis=0)
    return meanLine.reshape(1,img_arr.shape[1])

def processing_file(folder_path, file_name, ds_dict, block_size):
    image_id = folder_path.split('_')[5]
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
            meanLine += averaging_PRNU(arr)/total_block

        #Remove dark signal elements
        meanLine -= np.array(ds_dict.get(channel))

        #Normalize meanLine
        meanValue = np.mean(meanLine)
        meanLine /= meanValue

    return meanLine.reshape(meanLine.shape[1])

def processing_PAN_folder(PAN_folder, ds_dict, block_size=1000):
    PAN_file = 'PAN.h5'
    meanLine = processing_file(PAN_folder,PAN_file,ds_dict,block_size)
    return meanLine

def processing_MS_folder(MS_folder, ds_dict, block_size=250):
    MS_files = ['B1.h5','B2.h5','B3.h5','B4.h5']
    meanLine = {}
    for file in MS_files:
        meanLine[file.split('.')[0]] = processing_file(MS_folder,file,ds_dict,block_size)
    return meanLine

def processing_image(PAN_folder, MS_folder, ds_dict):
    prnu_pan = processing_PAN_folder(PAN_folder, ds_dict)
    prnu_ms = processing_MS_folder(MS_folder, ds_dict)
    return prnu_pan, prnu_ms

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

def read_ds_json(dark_signal_json_path):
    with open(dark_signal_json_path, 'r') as file:
        txt_data = file.read()
        ds_dic = json.loads(txt_data)
        return ds_dic

def gaussian_filter(prnu_arr,sigma=20):
    #1D gaussian filter with standard deviation Sigma
    n=math.ceil(3*sigma)
    X=np.array(range(-n,n+1,1))

    Y = np.exp(-1*np.power(X,2)/(2*sigma*sigma))
    gauss_filter = np.divide(Y,np.sum(Y,axis=0))

    L = np.floor(len(gauss_filter)/2)
    prnu_arr_extended = np.concatenate((np.full(int(L), 1), prnu_arr, np.full(int(L), 1)), axis=0)
    prnu_lf = np.convolve(prnu_arr_extended, gauss_filter)
    prnu_lf = prnu_lf[2*int(L):-2*int(L)]
    prnu_hf = np.divide(prnu_arr,prnu_lf)

    return prnu_hf

def high_pass_filter(prnu_grouped):
    prnu_grouped_high_pass_filter = copy.deepcopy(prnu_grouped)
    for channel,prnu_dic in prnu_grouped_high_pass_filter.items():
        for image_id,prnu in prnu_dic.items():
            prnu_dic[image_id] = list(gaussian_filter(np.array(prnu)))
        prnu_grouped_high_pass_filter[channel] = prnu_dic
    return prnu_grouped_high_pass_filter

def low_pass_convolve(prnu_arr, prnu_ground, sigma=20):
    n=math.ceil(3*sigma)
    X=np.array(range(-n,n+1,1))

    Y = np.exp(-1*np.power(X,2)/(2*sigma*sigma))
    gauss_filter = np.divide(Y,np.sum(Y,axis=0))

    L = np.floor(len(gauss_filter)/2)
    prnu_ground_extended = np.concatenate((np.full(int(L), 1), prnu_ground, np.full(int(L), 1)), axis=0)
    prnu_ground_lf = np.convolve(prnu_ground_extended, gauss_filter)
    prnu_ground_lf = prnu_ground_lf[2*int(L):-2*int(L)]
    prnu_add_low_pass = np.multiply(prnu_arr,prnu_ground_lf)

    return prnu_add_low_pass

def add_on_ground_low_pass(prnu_grouped, ground_cpf_dict):
    prnu_grouped_add_ground = copy.deepcopy(prnu_grouped)
    for channel,prnu_dic in prnu_grouped_add_ground.items():
        for image_id,prnu in prnu_dic.items():
            prnu_dic[image_id] = list(low_pass_convolve(np.array(prnu), ground_cpf_dict.get('prnu').get(channel)))
        prnu_grouped_add_ground[channel] = prnu_dic
    return prnu_grouped_add_ground

def calculate_PRNU(folder_path, dark_signal_json_path, on_ground_cpf_path):
    ds_dict = read_ds_json(dark_signal_json_path)
    ground_cpf_dict = read_cpf.read_cpf_to_json(on_ground_cpf_path)

    folders = os.listdir(folder_path)
    image_ids = set([name.split('_')[5] for name in folders])

    PAN_PRNUs = {}
    MS_PRNUs = {}
    for image_id in image_ids:
        PAN_folder, MS_folder = None, None
        for name in folders:
            if image_id in name and 'PAN' in name: PAN_folder=os.path.join(folder_path,name)
            if image_id in name and 'MS' in name: MS_folder=os.path.join(folder_path,name)
        if PAN_folder is not None and MS_folder is not None:
            prnu_pan, prnu_ms = processing_image(PAN_folder, MS_folder, ds_dict)
            PAN_PRNUs[image_id] = prnu_pan
            MS_PRNUs[image_id] = prnu_ms

    prnu_grouped = group_data(PAN_PRNUs, MS_PRNUs)
    prnu_grouped_high_pass_filter = high_pass_filter(prnu_grouped)
    prnu_grouped_add_on_ground_low_pass = add_on_ground_low_pass(prnu_grouped_high_pass_filter, ground_cpf_dict)

#    visualizer.visualize_channel_data(prnu_grouped,'Calculated PRNU')
#    visualizer.visualize_channel_data(prnu_grouped_high_pass_filter,'High pass filtering PRNU')
    visualizer.visualize_channel_data(prnu_grouped_add_on_ground_low_pass,'Final processed PRNU')

    return prnu_grouped_add_on_ground_low_pass


def validate_prnu(folder_path, dark_signal_json_path, on_ground_cpf_path):
    prnu = calculate_PRNU(folder_path, dark_signal_json_path, on_ground_cpf_path)

    result = {}
    for channel,val in prnu.items():
        result[channel]=val.get('average')

    result_path = data_writer.write_data(result)
    return result_path

if __name__=='__main__':
    """
    three parameters:
    param1: path to the folder contains many dessert image acquisitions. In each folder, ".RAW" file must be converted to HDF5 file using matlab script
    param2: path to the file contains the calculated dark_signal.json
    param3: path to the on-ground CPF
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('folderpath', help='String Folderpath')
    parser.add_argument('dspath', help='String dark signal json path')
    parser.add_argument('cpfpath', help='String on ground cpf path')
    args = parser.parse_args()
    folderpath = args.folderpath
    dspath = args.dspath
    cpfpath = args.cpfpath
    validate_prnu(folderpath,dspath,cpfpath)