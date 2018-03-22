# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 17:09:50 2018

@author: ThangPD
"""
import os
import json

def write_data(ds_grouped,file_path=None):
    try:
        if file_path==None:
            src_path = os.path.abspath(os.path.dirname(__file__))
            file_path = os.path.join(src_path,'dark_signal.json')

        if os.path.exists(file_path) and os.path.isfile(file_path):
            os.remove(file_path)

        with open(file_path, 'w') as f:
            f.write(json.dumps(ds_grouped))
        print("Dark signal saved to file. Check file at:",file_path)
        return file_path
    except:
        print("Cannot save dark signal data to file")
        return None