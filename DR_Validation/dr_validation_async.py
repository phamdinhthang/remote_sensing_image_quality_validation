# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:58:28 2018

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

import numpy as np
import argparse   
from matplotlib import pyplot as plt
import multiprocessing

from matplotlib.widgets import RectangleSelector
from PIL import Image

def validate_dr(img_path):
    ROI_selection(img_path)

def read_image(img_path,grayscale_only=True):
    img = Image.open(img_path)
    img_arr = np.array(img)

    if grayscale_only==True and len(img_arr.shape)>2:
        img_grey = img.convert('L')
        img_arr = np.array(img_grey)

#    img_arr = convert_10bits_to_8bits(img_arr)
    return img_arr

def convert_10bits_to_8bits(img_arr):
    if img_arr.dtype==np.uint16:
        #16bits, but only 10bits is usable
        print("Original image arr dtype:",img_arr.dtype)
        img_arr_8bits = np.divide(img_arr,4)
        img_arr_8bits = img_arr_8bits.astype(np.uint8)
        return img_arr_8bits
    else:
        return img_arr

class EventHandler(object):
    def __init__(self, filename):
        self.filename = filename

    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.roi = np.array([y1, y2, x1, x2])

    def event_exit_manager(self, event):
        if event.key in ['enter']:
            dr_estimator = Dynamic_Range(self.filename, self.roi)
            dr_estimator.calculate_DR()

class ROI_selection(object):
    def __init__(self, filename):
        self.filename = filename
        self.img_arr = read_image(self.filename)

        job_for_another_core = multiprocessing.Process(target=ROI_selection.show_selector,args=(self,))
        job_for_another_core.start()

    @staticmethod
    def show_selector(ROI_instance):
        fig_image, current_ax = plt.subplots()
        plt.imshow(ROI_instance.img_arr, cmap='gray')
        eh = EventHandler(ROI_instance.filename)
        rectangle_selector = RectangleSelector(current_ax,
                                               eh.line_select_callback,
                                               drawtype='box',
                                               useblit=True,
                                               button=[1, 2, 3],
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)
        print(rectangle_selector.center)
        plt.connect('key_press_event', eh.event_exit_manager)
        plt.colorbar()
        plt.show()

class Dynamic_Range(object):
    def __init__(self, filename, roi):
        self.roi = roi.astype(int)
        img_arr = read_image(filename)
        img_arr = img_arr[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        self.data = img_arr
        self.examine_bit_depth()

    def examine_bit_depth(self):
        if self.data.dtype==np.uint16:
            self.bins = 1024
            self.range = 1023
        else:
            self.bins = 256
            self.range = 255

    def calculate_DR(self):
        print("Image arr dtype =",self.data.dtype)
        print("Cropped sub-region shape =",self.data.shape)

        job_for_another_core = multiprocessing.Process(target=Dynamic_Range.plot_result,args=(self,))
        job_for_another_core.start()

    @staticmethod
    def plot_result(dr_instance):
        data_flat = dr_instance.data.flatten()

        plt.figure()

        plt.subplot(2,2,1)
        plt.imshow(dr_instance.data,cmap='gray')
        plt.colorbar()
        plt.title("Cropped area")

        plt.subplot(2,2,2)
        plt.hist(data_flat,bins=dr_instance.bins,range=(0,dr_instance.range))
        plt.title("Image histogram")
        plt.xlabel("Pixel intensity (lsb)")
        plt.ylabel("Pixel count")

        unique, counts = np.unique(data_flat, return_counts=True)
        first_10_unique = unique[:10]
        first_10_unique_counts = counts[:10]
        last_10_unique = unique[-10:]
        last_10_unique_counts = counts[-10:]

        plt.subplot(2,2,3)
        plt.bar(first_10_unique, first_10_unique_counts)
        plt.xticks(first_10_unique, rotation=90)
        plt.title("Histogram zoomed: ten darkest level")
        plt.xlabel("Pixel intensity (lsb)")
        plt.ylabel("Pixel count")

        plt.subplot(2,2,4)
        plt.bar(last_10_unique, last_10_unique_counts)
        plt.xticks(last_10_unique, rotation=90)
        plt.title("Histogram zoomed: ten brightest level")
        plt.xlabel("Pixel intensity (lsb)")
        plt.ylabel("Pixel count")

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    filename = args.filepath
    ROI_selection(filename)