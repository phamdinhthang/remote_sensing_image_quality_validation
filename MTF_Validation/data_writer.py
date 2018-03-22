# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 18:20:00 2018

@author: ThangPD
"""

import os
import json
def write_data(mtf,file_path=None):
    try:
        if file_path==None:
            src_path = os.path.abspath(os.path.dirname(__file__))
            file_path = os.path.join(src_path,'mtf.json')

        if os.path.exists(file_path) and os.path.isfile(file_path):
            os.remove(file_path)
        with open(file_path, 'w') as f:
            f.write(json.dumps(mtf))
        print("MTF saved to file. Check file at:",file_path)
    except:
        print("Cannot save MTF data to file")