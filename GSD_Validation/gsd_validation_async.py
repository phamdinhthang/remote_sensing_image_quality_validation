# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:41:09 2018

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
from math import radians, cos, sin, asin, sqrt

def validate_gsd(img_path,lat1,long1,lat2,long2):
    coordinate_dic={'lat1':lat1,'long1':long1,'lat2':lat2,'long2':long2}
    ROI_selection(img_path,coordinate_dic)

def read_image(img_path,grayscale_only=True):
    img = Image.open(img_path)
    img_arr = np.array(img)

    if grayscale_only==True and len(img_arr.shape)>2:
        img_grey = img.convert('L')
        img_arr = np.array(img_grey)
    return img_arr

class EventHandler(object):
    def __init__(self, filename, coordinate_dic):
        self.filename = filename
        self.coordinate_dic = coordinate_dic

    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.roi = np.array([y1, y2, x1, x2])

    def event_exit_manager(self, event):
        if event.key in ['enter']:
            gsd_estimator = GSD(self.filename, self.roi, self.coordinate_dic)
            gsd_estimator.calculate_GSD()

class ROI_selection(object):
    def __init__(self, filename,coordinate_dic):
        self.filename = filename
        self.coordinate_dic = coordinate_dic
        self.image_data = read_image(self.filename,grayscale_only=False)

        job_for_another_core = multiprocessing.Process(target=ROI_selection.show_selector,args=(self,))
        job_for_another_core.start()

    @staticmethod
    def show_selector(ROI_instance):
        fig_image, current_ax = plt.subplots()
        plt.imshow(ROI_instance.image_data, cmap='gray')
        eh = EventHandler(ROI_instance.filename,ROI_instance.coordinate_dic)
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

class GSD(object):
    def __init__(self, filename, roi, coordinate_dic):
        self.roi = roi.astype(int)
        self.coordinate_dic = coordinate_dic

        img_arr = read_image(filename)
        img_arr = img_arr[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        self.data = img_arr

    def calculate_GSD(self):
        print("Pixel 1 location (x, y) = (",self.roi[2],",",self.roi[0],")")
        print("Pixel 2 location (x, y) = (",self.roi[3],",",self.roi[1],")")

        y_pixel_distance = abs(self.roi[1] - self.roi[0])
        x_pixel_distance = abs(self.roi[3] - self.roi[2])

        point1 = {'lat':self.coordinate_dic.get('lat1'),'long':self.coordinate_dic.get('long1')}
        point2 = {'lat':self.coordinate_dic.get('lat2'),'long':self.coordinate_dic.get('long2')}

        #latitude distance = y distance, longitude distance = x distance
        lat_distance, long_distance = self.calculate_coordinate_distance(point1,point2)

        y_gsd = lat_distance/y_pixel_distance
        x_gsd = long_distance/x_pixel_distance

        print("Latitude distance:",lat_distance,"(m)")
        print("Longitude distance:",long_distance,"(m)")
        print("GSD X (across track) = ",x_gsd,"(m)")
        print("GSD Y (along track) = ",y_gsd,"(m)")

        result_dic = {'pixel1_x':self.roi[2],
                      'pixel1_y':self.roi[0],
                      'pixel2_x':self.roi[3],
                      'pixel2_y':self.roi[1],
                      'y_pixel_distance':y_pixel_distance,
                      'x_pixel_distance':x_pixel_distance,
                      'latitude_distance':lat_distance,
                      'longitude_distance':long_distance,
                      'gsd_y':y_gsd,
                      'gsd_x':x_gsd}
        job_for_another_core = multiprocessing.Process(target=GSD.plot_result,args=(result_dic,))
        job_for_another_core.start()

    @staticmethod
    def plot_result(result_dict):
        pixel1_x = result_dict.get('pixel1_x')
        pixel1_y = result_dict.get('pixel1_y')
        pixel2_x = result_dict.get('pixel2_x')
        pixel2_y = result_dict.get('pixel2_y')
        y_pixel_distance = result_dict.get('y_pixel_distance')
        x_pixel_distance = result_dict.get('x_pixel_distance')
        latitude_distance = result_dict.get('latitude_distance')
        longitude_distance = result_dict.get('longitude_distance')
        gsd_y = result_dict.get('gsd_y')
        gsd_x = result_dict.get('gsd_x')

        plt.figure(figsize=(6,3))

        plt.text(-0.1, 0.8, 'Pixel 1 location (x,y) = ('+str(pixel1_x)+','+str(pixel1_y)+')',
                verticalalignment='bottom', horizontalalignment='left',
                color='blue', fontsize=12)
        plt.text(-0.1, 0.7, 'Pixel 2 location (x,y) = ('+str(pixel2_x)+','+str(pixel2_y)+')',
                verticalalignment='bottom', horizontalalignment='left',
                color='blue', fontsize=12)
        plt.text(-0.1, 0.6, 'Y pixel distance = '+str(y_pixel_distance)+' (pixel)',
                verticalalignment='bottom', horizontalalignment='left',
                color='blue', fontsize=12)
        plt.text(-0.1, 0.5, 'X pixel distance = '+str(x_pixel_distance)+' (pixel)',
                verticalalignment='bottom', horizontalalignment='left',
                color='blue', fontsize=12)
        plt.text(-0.1, 0.4, 'Latitude distance (y distance) = '+str(latitude_distance)+' (m)',
                verticalalignment='bottom', horizontalalignment='left',
                color='blue', fontsize=12)
        plt.text(-0.1, 0.3, 'Longitude distance (x distance) = '+str(longitude_distance)+' (m)',
                verticalalignment='bottom', horizontalalignment='left',
                color='blue', fontsize=12)
        plt.text(-0.1, 0.2, 'GSD along track (y GSD) = '+str(gsd_y)+' (m)',
                verticalalignment='bottom', horizontalalignment='left',
                color='blue', fontsize=12)
        plt.text(-0.1, 0.1, 'GSD across track (x GSD) = '+str(gsd_x)+' (m)',
                verticalalignment='bottom', horizontalalignment='left',
                color='blue', fontsize=12)
        plt.axis('off')
        plt.title("Ground sampling distance (GSD) validated result")
        plt.show()


    def calculate_coordinate_distance(self, point1, point2):
        lat1 = point1.get('lat')
        long1 = point1.get('long')

        lat2 = point2.get('lat')
        long2 = point2.get('long')

        lat_distance = self.haversine_distance(long1, lat1, long1, lat2 )
        long_distance = self.haversine_distance(long1, lat1, long2, lat1 )
        return lat_distance, long_distance

    def haversine_distance(self, long1, lat1, long2, lat2):
        """
        long1, lat1, long2, lat2 must be measured in decimal degree
        """

        long1, lat1, long2, lat2 = map(radians, [long1, lat1, long2, lat2])


        dlong = long2 - long1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlong/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371 # Radius of earth in kilometers

        #Return result in meters
        return 1000*c*r

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    parser.add_argument('lat1', help='String Latitude of control point 1')
    parser.add_argument('long1', help='String Longitude of control point 1')
    parser.add_argument('lat2', help='String Latitude of control point 2')
    parser.add_argument('long2', help='String Longitude of control point 2')
    args = parser.parse_args()
    filename = args.filepath
    lat1 = args.lat1
    long1 = args.long1
    lat2 = args.lat2
    long2 = args.long2
    validate_gsd(filename, lat1, long1, lat2, long2)