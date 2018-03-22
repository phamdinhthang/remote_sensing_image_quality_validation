# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:40:11 2018

@author: ThangPD
"""

from flask import Flask
from flask_cors import CORS
from flask_restplus import Resource, Api
import snr_validation_async
import os

app = Flask(__name__)
api = Api(app, version='1.0', title='Module kiểm định độ nhiễu ảnh (Signal to Noise Ratio (SNR))', description='Tính toán và kiểm định độ nhiễu từ dữ liệu ảnh', validate=True)
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

@api.route('/validate_snr/image_path=<image_path>&rotate=<rotate>', endpoint = 'validate-signal-to-noise-ratio')
@api.doc(params={'image_path': 'Đường dẫn đến tập tin dữ liệu ảnh định dạng TIFF 10bits',
                 'rotate':'Góc quay ảnh theo chiều ngược kim đồng hồ'})
class ValidateSNRResource(Resource):
    def get(self,image_path,rotate):
        if not os.path.exists(image_path) or not os.path.isfile(image_path) or not image_path.lower().endswith('.tif') or to_number(rotate) is None:
            return {'Lỗi':"Tham số truyền vào không đúng"}

        snr_validation_async.validate_snr(image_path,to_number(rotate))
        return {"Thông báo":"Lựa chọn vùng kiểm định trên ảnh hiện lên"}

if __name__ == '__main__':
    app.run(debug=True, host = 'localhost', port= 7003)