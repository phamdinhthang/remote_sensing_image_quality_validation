# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 18:17:05 2018

@author: ThangPD
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image
from math import radians, cos, sin, asin, sqrt

def validate_gsd(img_path):
    ROI_selection(img_path)

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
            gsd_estimator = GSD(self.filename, self.roi)
            gsd_estimator.calculate_GSD()

class ROI_selection(object):
    def __init__(self, filename):
        self.filename = filename
        self.image_data = read_image(self.filename)
        fig_image, current_ax = plt.subplots()
        plt.imshow(self.image_data, cmap='gray')
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

class GSD(object):
    def __init__(self, filename, roi):
        self.roi = roi.astype(int)

        img_arr = read_image(filename)
        img_arr = img_arr[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        self.data = img_arr

    def calculate_GSD(self):
        print("Pixel 1 location (y, x) = (",self.roi[0],",",self.roi[2],")")
        print("Pixel 2 location (y, x) = (",self.roi[1],",",self.roi[3],")")

        y_pixel_distance = abs(self.roi[1] - self.roi[0])
        x_pixel_distance = abs(self.roi[3] - self.roi[2])

        top_left = {'lat':12.743824,'long':108.067473}
        bottom_right = {'lat':12.742926,'long':108.068393}

        #latitude distance = y distance, longitude distance = x distance
        lat_distance, long_distance = self.calculate_coordinate_distance(bottom_right,top_left)

        y_gsd = lat_distance/y_pixel_distance
        x_gsd = long_distance/x_pixel_distance

        print("Latitude distance:",lat_distance,"(m)")
        print("Longitude distance:",long_distance,"(m)")
        print("GSD X (across track) = ",x_gsd,"(m)")
        print("GSD Y (along track) = ",y_gsd,"(m)")


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
    args = parser.parse_args()
    filename = args.filepath
    validate_gsd(filename)