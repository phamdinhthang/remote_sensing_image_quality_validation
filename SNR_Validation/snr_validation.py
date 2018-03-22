# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 18:07:02 2018

@author: ThangPD
ref: https://stats.stackexchange.com/questions/51946/how-to-calculate-the-signal-to-noise-ratio-snr-in-an-image
"""

import numpy as np
import cv2
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
            mtf_estimator = Uniform_Assessment_SNR(self.filename, self.roi)
            mtf_estimator.calculate_SNR()

class ROI_selection(object):
    def __init__(self, filename):
        self.filename = filename
        self.image_data = read_image(self.filename)
        fig_image, current_ax = plt.subplots()
        plt.imshow(self.image_data, cmap='gray')
        eh = EventHandler(self.filename)
        rect_select = RectangleSelector(current_ax,
                                        eh.line_select_callback,
                                        drawtype='box',
                                        useblit=True,
                                        button=[1, 2, 3],
                                        minspanx=5, minspany=5,
                                        spancoords='pixels',
                                        interactive=True)
        print("Rectangle Selector center:",rect_select.center)
        plt.connect('key_press_event', eh.event_exit_manager)
        plt.title('Original raw image')
        plt.colorbar()
        plt.show()

class Uniform_Assessment_SNR(object):
    def __init__(self, filename, roi):
        image_data = cv2.imread(filename, 0)
        self.roi = roi.astype(int)
        image_data = image_data[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        self.image_data = np.array(image_data)

    def calculate_SNR(self):
        fig = plt.figure()
        fig.suptitle(filename + ' SNR analysis with region (y1, y2, x1, x2) = ' + str(self.roi), fontsize=10)
        plt.subplot(2, 2, 1)
        plt.imshow(self.image_data, cmap='gray')
        plt.title("Croped Area")
        plt.colorbar()
        self.calculate_signal()
        self.calculate_noise()

#        snr = 10*math.log10(self.p_signal/self.p_noise)
        snr = self.p_signal/self.p_noise
        print("Image SNR =",snr)

    def calculate_signal(self):
        flatten = self.image_data.flatten()
        self.p_signal = np.mean(flatten)
        mean_image = np.full(self.image_data.shape,self.p_signal)

        plt.subplot(2, 2, 2)
        plt.imshow(mean_image, cmap='gray')
        plt.title("Mean signal Image. Signal power = "+str(self.p_signal))
        plt.colorbar()

    def calculate_noise(self):
        flatten = self.image_data.flatten()
        self.p_noise = np.std(flatten)

        pattern_image = np.absolute(self.image_data-self.p_signal)
        noise_image = np.absolute(np.random.normal(0,self.p_noise,self.image_data.shape))

        plt.subplot(2, 2, 3)
        plt.imshow(pattern_image, cmap='gray')
        plt.title("Noise pattern. Noise power = "+str(self.p_noise))
        plt.colorbar()

        plt.subplot(2, 2, 4)
        plt.imshow(noise_image, cmap='gray')
        plt.title("Noise level")
        plt.colorbar()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    filename = args.filepath
    ROI_selection(filename)