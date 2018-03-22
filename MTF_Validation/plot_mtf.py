# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 18:24:36 2018

@author: ThangPD
"""
import argparse
import json
from matplotlib import pyplot as plt

def read_mtf_json(mtf_json_path):
    with open(mtf_json_path, 'r') as file:
        txt_data = file.read()
        mtf_dic = json.loads(txt_data)
        return mtf_dic
def plot_mtf(mtfpath):
    mtf_dict = read_mtf_json(mtfpath)
    spatial_freq = mtf_dict.get('spatial_frequency')
    mtf = mtf_dict.get('mtf')
    smooth_mtf = mtf_dict.get('smooth_mtf')
    mtf_nyquist = mtf_dict.get('mtf_nyquist')

    plt.figure
    plt.plot(spatial_freq,mtf,label='MTF raw')
    plt.plot(spatial_freq,smooth_mtf,label='MTF smoothed')
    plt.title('MTF chart. MTF@nyquist_frequency = '+str(mtf_nyquist))
    plt.xlabel('spatial frequency (cycle/pixel)')
    plt.ylabel('Modulation factor')
    plt.legend()
    plt.grid()
    plt.show()

    print("MTF value @nyquist frequency =",mtf_nyquist)

if __name__ == '__main__':
    """
    single parameter: path to the mtf.json
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    filename = args.filepath
    plot_mtf(filename)