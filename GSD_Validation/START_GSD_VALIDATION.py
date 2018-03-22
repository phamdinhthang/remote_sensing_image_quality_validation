# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:33:49 2018

@author: ThangPD
"""

from flask import Flask
from flask_cors import CORS
from flask_restplus import Resource, Api
import gsd_validation_async
import os

app = Flask(__name__)
api = Api(app, version='1.0', title='Module kiểm định độ phân giải ảnh (Ground Sampling Distance (GSD))', description='Tính toán và kiểm định độ phân giải từ dữ liệu ảnh', validate=True)
CORS(app)

def to_number(s):
    if s is None:
        return None
    try:
        float(s)
        try:
            return int(s)
        except ValueError:
            return float(s)
    except ValueError:
        print(s, ' is not a number')
        return None

@api.route('/validate_gsd/image_path=<image_path>&lat1=<lat1>&long1=<long1>&lat2=<lat2>&long2=<long2>', endpoint = 'validate-ground-sampling-distance')
@api.doc(params={'image_path': 'Đường dẫn đến tập tin dữ liệu ảnh định dạng TIFF 10bits',
                 'lat1':'Vĩ độ điểm thứ nhất (decimal degrees)',
                 'long1':'Kinh độ điểm thứ nhất (decimal degrees)',
                 'lat2':'Vĩ độ điểm thứ hai (decimal degrees)',
                 'long2':'Kinh độ điểm thứ hai (decimal degrees)'})
class ValidateGroundSamplingDistanceResource(Resource):
    def get(self,image_path,lat1,long1,lat2,long2):
        if not os.path.exists(image_path) or not os.path.isfile(image_path) or not image_path.lower().endswith('.tif') or to_number(lat1) is None or to_number(long1) is None or to_number(lat2) is None or to_number(long2) is None:
            return {'Lỗi':"Tham số truyền vào không đúng"}

        gsd_validation_async.validate_gsd(image_path, to_number(lat1), to_number(long1), to_number(lat2), to_number(long2))
        return {"Thông báo":"Chọn 2 điểm trên ảnh hiện ra bằng cách kéo chuột"}

if __name__ == '__main__':
    app.run(debug=True, host = 'localhost', port= 7004)