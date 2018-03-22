# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:41:33 2018

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
import scipy.ndimage as img_pro
from matplotlib import pyplot as plt
import multiprocessing

from matplotlib.widgets import RectangleSelector
from PIL import Image

def validate_snr(img_path,rotate):
    ROI_selection(img_path,rotate)

def read_image(img_path,grayscale_only=True):
    img = Image.open(img_path)
    img_arr = np.array(img)

    if grayscale_only==True and len(img_arr.shape)>2:
        img_grey = img.convert('L')
        img_arr = np.array(img_grey)
    return img_arr

def rotate_image(img_arr,deg):
    rotated = img_pro.rotate(img_arr, deg, reshape=False)
    return rotated

class EventHandler(object):
    def __init__(self, filename, rotate):
        self.filename = filename
        self.rotate = rotate

    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.roi = np.array([y1, y2, x1, x2])

    def event_exit_manager(self, event):
        if event.key in ['enter']:
            mtf_estimator = Uniform_Assessment_SNR(self.filename, self.rotate, self.roi)
            mtf_estimator.calculate_SNR()

class ROI_selection(object):
    def __init__(self, filename, rotate=0):
        self.filename = filename
        self.rotate = rotate
        image_data = read_image(self.filename)
        self.image_data = rotate_image(image_data,self.rotate)

        job_for_another_core = multiprocessing.Process(target=ROI_selection.show_selector,args=(self,))
        job_for_another_core.start()

    def show_selector(ROI_instance):
        fig_image, current_ax = plt.subplots()
        plt.imshow(ROI_instance.image_data, cmap='gray')
        eh = EventHandler(ROI_instance.filename,ROI_instance.rotate)
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
    def __init__(self, filename, rotate, roi):
        image_data = read_image(filename)
        self.image_data = rotate_image(image_data,rotate)
        self.roi = roi.astype(int)
        self.image_data = self.image_data[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

    def calculate_SNR(self):
        self.calculate_signal()
        self.calculate_noise()
#        snr = 10*math.log10(self.p_signal/self.p_noise)
        self.snr = 10*self.p_signal/self.p_noise

        job_for_another_core = multiprocessing.Process(target=Uniform_Assessment_SNR.plot_result,args=(self,))
        job_for_another_core.start()



    def calculate_signal(self):
        flatten = self.image_data.flatten()
        third_quantile = np.percentile(flatten,75)

        signal_region = flatten[flatten>third_quantile]
        self.p_signal = np.mean(signal_region)

    def calculate_noise(self):
        flatten = self.image_data.flatten()
        med = np.median(flatten)
        noise_region = flatten[flatten<med]
        self.p_noise = np.std(noise_region)

    def plot_result(snr_instance):
        plt.figure()

        plt.subplot(2, 2, 1)
        plt.imshow(snr_instance.image_data, cmap='gray')
        plt.title("Croped Area")
        plt.colorbar()

        mean_image = np.full(snr_instance.image_data.shape,snr_instance.p_signal)
        plt.subplot(2, 2, 2)
        plt.imshow(mean_image, cmap='gray')
        plt.title("Mean signal Image. Signal power = "+str(int(snr_instance.p_signal)))
        plt.colorbar()

        pattern_image = np.absolute(snr_instance.image_data-snr_instance.p_signal)

        plt.subplot(2, 2, 3)
        plt.imshow(pattern_image, cmap='gray')
        plt.title("Noise pattern. Noise power = "+str(int(snr_instance.p_noise)))
        plt.colorbar()

        noise_image = np.absolute(np.random.normal(0,snr_instance.p_noise,snr_instance.image_data.shape))
        plt.subplot(2, 2, 4)
        plt.imshow(noise_image, cmap='gray')
        plt.title("Noise level")
        plt.colorbar()

        plt.suptitle('Calculated SNR = '+str(int(snr_instance.snr)), fontsize=16)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    parser.add_argument('rotate', help='Integer rotate angle')
    args = parser.parse_args()
    filename = args.filepath
    rotate = args.rotate
    ROI_selection(filename,rotate)