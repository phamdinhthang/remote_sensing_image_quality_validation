# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:29:55 2018

@author: ThangPD
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image

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
        fig_image, current_ax = plt.subplots()
        plt.imshow(self.img_arr, cmap='gray')
        eh = EventHandler(self.filename)
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
        print(type(self.data))
        print("Data shape=",self.data.shape)

        data_flat = self.data.flatten()
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(self.data,cmap='gray')
        plt.colorbar()
        plt.title("Cropped area")

        plt.subplot(1,2,2)
        plt.hist(data_flat,bins=self.bins,range=(0,self.range))
        plt.title("Image histogram")
        plt.xlabel("Pixel intensity (lsb)")
        plt.ylabel("Pixel count")
        plt.show()

        percentile_90 = int(0.9*self.range)

        a = self.data>percentile_90
        b = self.data<self.range
        last_10_percentile = np.multiply(a,b)

        count = np.sum(last_10_percentile)
        print("Pixel counts in intensity range (",percentile_90,",",self.range,")=",count)
        if count==0:
            print("Image not saturated")
        else:
            print("Image has region on saturated zone")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    filename = args.filepath
    ROI_selection(filename)