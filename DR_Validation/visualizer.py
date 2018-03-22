# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 17:08:50 2018

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

from matplotlib import pyplot as plt
import multiprocessing

def visualize_channel_data(dark_signal_grouped):
    job_for_another_core = multiprocessing.Process(target=plot_graph,args=(dark_signal_grouped,))
    job_for_another_core.start()

def plot_graph(dark_signal_grouped):
    plt.figure(figsize=(8, 10), dpi=80)
    i=1
    for channel_name,channel_data in dark_signal_grouped.items():
        plt.subplot(5, 2, i)
        i+=1
        for image_id,meanLine in channel_data.items():
            if image_id != 'average':
                plt.plot(meanLine)
        plt.legend(list(channel_data.keys()), loc='upper left')
        plt.title(channel_name+' Dark Signal')
        plt.xlabel('Colum index')
        plt.ylabel('lsb')
        plt.grid()

        plt.subplot(5, 2, i)
        i+=1
        if channel_data.get('average') is not None:
            plt.plot(channel_data.get('average'))
            plt.title(channel_name+' Average dark Signal')
            plt.xlabel('Colum index')
            plt.ylabel('lsb')
            plt.grid()
    plt.tight_layout()
    plt.show()

