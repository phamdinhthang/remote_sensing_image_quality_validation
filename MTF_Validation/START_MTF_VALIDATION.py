# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 18:07:13 2018

@author: ThangPD
"""

from flask import Flask
from flask_cors import CORS
from flask_restplus import Resource, Api
import mtf_validation_async_normalized
import os

app = Flask(__name__)
api = Api(app, version='1.0', title='Module kiểm định độ nét ảnh (Modulation Transfer Function (MTF))', description='Tính toán và kiểm định MTF từ dữ liệu ảnh', validate=True)
CORS(app)


@api.route('/validate_across_track_mtf/image_path=<image_path>', endpoint = 'validate-modulation-transfer-function-across-track')
@api.doc(params={'image_path': 'Đường dẫn đến tập tin dữ liệu ảnh định dạng TIFF 10bits'})
class ValidateModulationTransferFunctionAcrossTrackResource(Resource):
    def get(self,image_path):
        if not os.path.exists(image_path) or not os.path.isfile(image_path) or not image_path.lower().endswith('.tif'):
            return {'Lỗi':"Tham số truyền vào không đúng"}

        mtf_validation_async_normalized.validate_mtf(image_path,along_track=False)
        return {"Thông báo":"Chọn vùng chuyển đổi từ trắng - đen hoặc đen - trắng theo chiều ngang trên ảnh"}


@api.route('/validate_along_track_mtf/image_path=<image_path>', endpoint = 'validate-modulation-transfer-function-along-track')
@api.doc(params={'image_path': 'Đường dẫn đến tập tin dữ liệu ảnh định dạng TIFF 10bits'})
class ValidateModulationTransferFunctionAlongTrackResource(Resource):
    def get(self,image_path):
        if not os.path.exists(image_path) or not os.path.isfile(image_path) or not image_path.lower().endswith('.tif'):
            return {'Lỗi':"Tham số truyền vào không đúng"}

        mtf_validation_async_normalized.validate_mtf(image_path,along_track=True)
        return {"Thông báo":"Chọn vùng chuyển đổi từ trắng - đen hoặc đen - trắng theo chiều dọc trên ảnh"}

if __name__ == '__main__':
    app.run(debug=True, host = 'localhost', port= 7002)