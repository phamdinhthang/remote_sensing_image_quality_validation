# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 18:08:42 2018

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
import cv2
import argparse
from matplotlib import pyplot as plt
import multiprocessing
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector
from scipy import interpolate
from scipy.signal import savgol_filter
from PIL import Image
import data_writer


def validate_mtf(image_path):
    ROI_selection(image_path)

# Reference:
# http://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def read_image(img_path,grayscale_only=True,convert_10bits=False):
    img = Image.open(img_path)
    img_arr = np.array(img)

    if grayscale_only==True and len(img_arr.shape)>2:
        img_grey = img.convert('L')
        img_arr = np.array(img_grey)

    if convert_10bits==True: img_arr = convert_10bits_to_8bits(img_arr)
    return img_arr

def convert_10bits_to_8bits(img_arr):
    if img_arr.dtype==np.uint16:
        #16bits, but only 10bits is usable
        print("Original image arr dtype:",img_arr.dtype)
        img_arr_8bits = np.divide(img_arr,4)
        img_arr_8bits = img_arr_8bits.astype('uint8')
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
            mtf_estimator = Slanted_Edge_MTF(self.filename, self.roi)
            mtf_estimator.calculate_MTF()

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

class Slanted_Edge_MTF(object):
    def __init__(self, filename, roi):
        self.roi = roi.astype(int)

        img_arr = read_image(filename,convert_10bits=True)
        img_arr = img_arr[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

        self.data = img_arr
        self.min = np.amin(self.data)
        self.max = np.amax(self.data)
        print("Region: max intense=",self.max,",min instense=",self.min,", middle intense=",(self.min+self.max)/2)
        _, th = cv2.threshold(self.data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#        _, th = cv2.threshold(self.data, 300, 1023, cv2.THRESH_BINARY)
        self.min = np.amin(self.data)
        self.max = np.amax(self.data)
        self.threshold = th*(self.max - self.min) + self.min
        below_thresh = ((self.data >= self.min) & (self.data <= self.threshold))
        above_thresh = ((self.data >= self.threshold) & (self.data <= self.max))
        area_below_thresh = self.data[below_thresh].sum()/below_thresh.sum()
        area_above_thresh = self.data[above_thresh].sum()/above_thresh.sum()
        self.threshold = (area_below_thresh - area_above_thresh)/2 + area_above_thresh

    def calculate_MTF(self):
#        fig = plt.figure()
#        fig.suptitle(filename + ' MTF analysis with region (y1, y2, x1, x2) = ' + str(self.roi), fontsize=10)
#        plt.subplot(2, 3, 1)
#        plt.imshow(np.array(self.data), cmap='gray')
#        plt.title("Cropped area")

        self.edges = cv2.Canny(self.data, self.min, self.max-5)
#        plt.subplot(2, 3, 2)
#        plt.imshow(self.edges, cmap='gray')
#        plt.title("Detected Edge")
        row_edge, col_edge = np.where(self.edges == 255)
        z = np.polyfit(np.flipud(col_edge), row_edge, 1)
        angle_radians = np.arctan(z[0])
        angle_deg = angle_radians * (180/3.14)
        if abs(angle_deg) < 45:
            self.data = np.transpose(self.data)

        try:
            self.compute_esf()
        except:
            print("Refined computation failed. Try raw computation instead")
            self.compute_raw_mtf_estimation()


    def compute_raw_mtf_estimation(self):
        self.raw_esf = np.mean(self.data,axis=0)
        esf_max = np.full(int(len(self.raw_esf)/2),np.max(self.raw_esf))
        esf_min = np.full(int(len(self.raw_esf)/2),np.min(self.raw_esf))
        self.raw_esf_extend = None
        if self.raw_esf[0]-self.raw_esf[-1] > 0:
            self.raw_esf_extend = np.concatenate((esf_max,self.raw_esf,esf_min), axis=0)
        else:
            self.raw_esf_extend = np.concatenate((esf_min,self.raw_esf,esf_max), axis=0)

        self.raw_lsf = np.gradient(self.raw_esf_extend)
        self.raw_mtf = np.absolute(np.fft.fft(self.raw_lsf, 2048))

        max_mtf = np.max(self.raw_mtf)
        min_mtf = np.min(self.raw_mtf)
        self.normalized_mtf = np.divide(self.raw_mtf-min_mtf,max_mtf-min_mtf)

        index_of_zero = np.where(self.normalized_mtf == 0)
        indx = index_of_zero[0][0]
        self.normalized_mtf = self.normalized_mtf[:indx]

        job_for_another_core = multiprocessing.Process(target=Slanted_Edge_MTF.plot_raw_mtf_estimation,args=(self,))
        job_for_another_core.start()

    @staticmethod
    def plot_raw_mtf_estimation(slant_edge_instance):
        plt.figure()
        plt.suptitle('Calculated MTF@nyquist frequency =', fontsize=16)

        plt.subplot(2, 2, 1)
        plt.imshow(slant_edge_instance.data, cmap='gray')
        plt.title("Cropped area")

        plt.subplot(2, 2, 2)
        plt.plot(slant_edge_instance.raw_esf_extend)
        plt.title("Edge spread function (ESF) curve")
        plt.xlabel("pixel")
        plt.ylabel("lsb")

        plt.subplot(2, 2, 3)
        plt.plot(slant_edge_instance.raw_lsf)
        plt.title("Line Spread Function (LSF) curve")
        plt.xlabel("pixel")
        plt.ylabel("lsb")

        plt.subplot(2, 2, 4)
        xticks = np.arange(0.0,1.0,1/len(slant_edge_instance.normalized_mtf))
        plt.plot(slant_edge_instance.normalized_mtf,xticks)
        print("Len mtf = ",len(slant_edge_instance.normalized_mtf))
        plt.title("Modulation Transfer Function")
        plt.xlabel("cycle/pixel")
        plt.ylabel("Modulation factor")
        plt.show()

    def compute_esf(self):
        kernel = np.ones((3, 3), np.float32)/9
        smooth_img = cv2.filter2D(self.data, -1, kernel)
        row = self.data.shape[0]
        column = self.data.shape[1]
        array_values_near_edge = np.empty([row, 13])
        array_positions = np.empty([row, 13])
        edge_pos = np.empty(row)
        smooth_img = smooth_img.astype(float)
        for i in range(0, row):
            # print(smooth_img[i,:])
            diff_img = smooth_img[i, 1:] - smooth_img[i, 0:(column-1)]
            abs_diff_img = np.absolute(diff_img)
            abs_diff_max = np.amax(abs_diff_img)
            if abs_diff_max == 1:
                raise IOError('No Edge Found')
            app_edge = np.where(abs_diff_img == abs_diff_max)
            bound_edge_left = app_edge[0][0] - 2
            bound_edge_right = app_edge[0][0] + 3
            strip_cropped = self.data[i, bound_edge_left:bound_edge_right]
            temp_y = np.arange(1, 6)
            try:
                f = interpolate.interp1d(strip_cropped, temp_y, kind='cubic')
            except: continue
            edge_pos_temp = f(self.threshold)
            edge_pos[i] = edge_pos_temp + bound_edge_left - 1
            bound_edge_left_expand = app_edge[0][0] - 6
            bound_edge_right_expand = app_edge[0][0] + 7
            try:
                array_values_near_edge[i, :] = self.data[i, bound_edge_left_expand:bound_edge_right_expand]
            except: continue
            array_positions[i, :] = np.arange(bound_edge_left_expand, bound_edge_right_expand)
#        y = np.arange(0, row)
        nans, x = nan_helper(edge_pos)
        edge_pos[nans] = np.interp(x(nans), x(~nans), edge_pos[~nans])

        array_positions_by_edge = array_positions - np.transpose(edge_pos * np.ones((13, 1)))
        num_row = array_positions_by_edge.shape[0]
        num_col = array_positions_by_edge.shape[1]
        array_values_by_edge = np.reshape(array_values_near_edge, num_row*num_col, order='F')
        array_positions_by_edge = np.reshape(array_positions_by_edge, num_row*num_col, order='F')

        bin_pad = 0.0001
        pixel_subdiv = 0.10
        topedge = np.amax(array_positions_by_edge) + bin_pad + pixel_subdiv
        botedge = np.amin(array_positions_by_edge) - bin_pad
        print("Top edge =",topedge)
        print("Bottom edge =",botedge)
        binedges = np.arange(botedge, topedge+1, pixel_subdiv)
        numbins = np.shape(binedges)[0] - 1

        binpositions = binedges[0:numbins] + (0.5) * pixel_subdiv

        h, whichbin = np.histogram(array_positions_by_edge, binedges)
        whichbin = np.digitize(array_positions_by_edge, binedges)
        binmean = np.empty(numbins)

        for i in range(0, numbins):
            flagbinmembers = (whichbin == i)
            binmembers = array_values_by_edge[flagbinmembers]
            binmean[i] = np.mean(binmembers)
        nans, x = nan_helper(binmean)
        binmean[nans] = np.interp(x(nans), x(~nans), binmean[~nans])
        esf = binmean
        xesf = binpositions
        xesf = xesf - np.amin(xesf)
        self.xesf = xesf
        esf_smooth = savgol_filter(esf, 51, 3)
        self.esf = esf
        self.esf_smooth = esf_smooth
#        plt.subplot(2, 3, 3)
#        plt.title("Edge Spread Function (ESF) Curve")
#        plt.xlabel("pixel")
#        plt.ylabel("lsb")
#        plt.plot(self.xesf, self.esf, 'y-', self.xesf, self.esf_smooth)
        yellow_patch = mpatches.Patch(color='yellow', label='Raw ESF')
        blue_patch = mpatches.Patch(color='blue', label='Smooth ESF')
        plt.legend(handles=[yellow_patch, blue_patch], loc=4)

        self.compute_lsf()

    def compute_lsf(self):
        diff_esf = abs(self.esf[1:] - self.esf[0:(self.esf.shape[0] - 1)])
        diff_esf = np.append(0, diff_esf)
        lsf = diff_esf
        diff_esf_smooth = abs(self.esf_smooth[0:(self.esf.shape[0] - 1)] - self.esf_smooth[1:])
        diff_esf_smooth = np.append(0, diff_esf_smooth)
        lsf_smooth = diff_esf_smooth
        self.lsf = lsf
        self.lsf_smooth = lsf_smooth
#        plt.subplot(2, 3, 4)
#        plt.title("Line Spread Function (LSF) Curve")
#        plt.xlabel("pixel")
#        plt.ylabel("lsb")
#        plt.plot(self.xesf, lsf, 'y-', self.xesf, lsf_smooth)
        yellow_patch = mpatches.Patch(color='yellow', label='Raw LSF')
        blue_patch = mpatches.Patch(color='blue', label='Smooth LSF')
        plt.legend(handles=[yellow_patch, blue_patch])
        self.compute_mtf()

    def compute_mtf(self):
        mtf = np.absolute(np.fft.fft(self.lsf, 2048))
        mtf_smooth = np.absolute(np.fft.fft(self.lsf_smooth, 2048))
        mtf_final = np.fft.fftshift(mtf)
        mtf_final_smooth = np.fft.fftshift(mtf_smooth)

        x_mtf_final = np.arange(0,1,1./127)
        mtf_final = mtf_final[1024:1151]/np.amax(mtf_final[1024:1151])
        mtf_final_smooth = mtf_final_smooth[1024:1151]/np.amax(mtf_final_smooth[1024:1151])
        self.x_mtf_final = x_mtf_final
        self.mtf_final = mtf_final
        self.mtf_final_smooth = mtf_final_smooth
#        plt.subplot(2, 3, 5)
#        plt.plot(self.x_mtf_final, self.mtf_final, 'y-', self.x_mtf_final, self.mtf_final_smooth)
#        plt.title("Modulation Transfer Function (MTF) Curve")
#        plt.xlabel("cycles/pixel")
#        plt.ylabel("Modulation Factor")
#        yellow_patch = mpatches.Patch(color='yellow', label='Raw MTF')
#        blue_patch = mpatches.Patch(color='blue', label='Smooth MTF')
#        plt.legend(handles=[yellow_patch, blue_patch])

#        plt.show()

        mtf_final_smooth = savgol_filter(mtf_final, 101, 5)
        nyquist_region = np.logical_and(x_mtf_final > 0.5*0.99,x_mtf_final < 0.5*1.01)
        self.mtf_nyquist = np.mean(mtf_final_smooth[nyquist_region])

        self.write_mtf(x_mtf_final,mtf_final,mtf_final_smooth)

        job_for_another_core = multiprocessing.Process(target=Slanted_Edge_MTF.plot_result,args=(self,))
        job_for_another_core.start()

        return mtf

    def plot_result(slant_edge_instance):
        plt.figure()
        plt.suptitle('Calculated MTF@nyquist frequency =' + str("{:10.3f}".format(slant_edge_instance.mtf_nyquist)), fontsize=16)

        plt.subplot(2, 3, 1)
        plt.imshow(np.array(slant_edge_instance.data), cmap='gray')
        plt.title("Cropped area")

        plt.subplot(2, 3, 2)
        plt.imshow(slant_edge_instance.edges, cmap='gray')
        plt.title("Detected Edge")

        plt.subplot(2, 3, 3)
        plt.title("Edge Spread Function (ESF) Curve")
        plt.xlabel("pixel")
        plt.ylabel("lsb")
        plt.plot(slant_edge_instance.xesf, slant_edge_instance.esf, 'y-', slant_edge_instance.xesf, slant_edge_instance.esf_smooth)

        plt.subplot(2, 3, 4)
        plt.title("Line Spread Function (LSF) Curve")
        plt.xlabel("pixel")
        plt.ylabel("lsb")
        plt.plot(slant_edge_instance.xesf, slant_edge_instance.lsf, 'y-', slant_edge_instance.xesf, slant_edge_instance.lsf_smooth)

        plt.subplot(2, 3, 5)
        plt.plot(slant_edge_instance.x_mtf_final, slant_edge_instance.mtf_final, 'y-', slant_edge_instance.x_mtf_final, slant_edge_instance.mtf_final_smooth)
        plt.title("Modulation Transfer Function (MTF) Curve")
        plt.xlabel("cycles/pixel")
        plt.ylabel("Modulation Factor")

        yellow_patch = mpatches.Patch(color='yellow', label='Raw MTF')
        blue_patch = mpatches.Patch(color='blue', label='Smooth MTF')
        plt.legend(handles=[yellow_patch, blue_patch])
        plt.show()


    def write_mtf(self,spatial_freq,mtf,smooth_mtf):
        mtf_dict = {}
        mtf_dict['spatial_frequency']=list(spatial_freq)
        mtf_dict['mtf']=list(mtf)
        mtf_dict['smooth_mtf']=list(smooth_mtf)
        mtf_dict['mtf_nyquist']=self.mtf_nyquist
        data_writer.write_data(mtf_dict)

if __name__ == '__main__':
    """
    single parameter: path to the 10bits tiff image
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    filename = args.filepath
    ROI_selection(filename)