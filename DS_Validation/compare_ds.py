# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 17:12:04 2018

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

import argparse
import json
import os
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing

def read_file(filepath):
    if os.path.exists(filepath) and os.path.isfile(filepath):
        with open(filepath,'r') as file:
            data = file.read()
            return data

def compare(ds1,ds2):
    ds1_dict = json.loads(read_file(ds1))
    ds2_dict = json.loads(read_file(ds2))

    PAN_diff = np.absolute(np.array(ds1_dict.get('PAN'))-np.array(ds2_dict.get('PAN')))
    B1_diff = np.absolute(np.array(ds1_dict.get('B1'))-np.array(ds2_dict.get('B1')))
    B2_diff = np.absolute(np.array(ds1_dict.get('B2'))-np.array(ds2_dict.get('B2')))
    B3_diff = np.absolute(np.array(ds1_dict.get('B3'))-np.array(ds2_dict.get('B3')))
    B4_diff = np.absolute(np.array(ds1_dict.get('B4'))-np.array(ds2_dict.get('B4')))

    job_for_another_core = multiprocessing.Process(target=plot_differences,args=(PAN_diff,B1_diff,B2_diff,B3_diff,B4_diff))
    job_for_another_core.start()

def plot_differences(PAN_diff,B1_diff,B2_diff,B3_diff,B4_diff):
    plt.figure()

    plt.subplot(3,2,1)
    plt.plot(PAN_diff)
    plt.title("PAN differences")
    plt.xlabel('Colum index')
    plt.ylabel('lsb')
    plt.grid()

    plt.subplot(3,2,2)
    plt.plot(B1_diff)
    plt.title("B1 differences")
    plt.xlabel('Colum index')
    plt.ylabel('lsb')
    plt.grid()

    plt.subplot(3,2,3)
    plt.plot(B2_diff)
    plt.title("B2 differences")
    plt.xlabel('Colum index')
    plt.ylabel('lsb')
    plt.grid()

    plt.subplot(3,2,4)
    plt.plot(B3_diff)
    plt.title("B3 differences")
    plt.xlabel('Colum index')
    plt.ylabel('lsb')
    plt.grid()

    plt.subplot(3,2,5)
    plt.plot(B4_diff)
    plt.title("B4 differences")
    plt.xlabel('Colum index')
    plt.ylabel('lsb')
    plt.grid()

    plt.tight_layout()
    plt.suptitle("Compare dark signal from 2 json files")
    plt.show()

if __name__=='__main__':
    """
    2 parameters: 2 path to 2 ds json file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('ds1', help='String ds json file 1')
    parser.add_argument('ds2', help='String ds json file 1')
    args = parser.parse_args()
    ds1 = args.ds1
    ds2 = args.ds2
    compare(ds1,ds2)