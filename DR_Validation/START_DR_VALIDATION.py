# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:57:00 2018

@author: ThangPD
"""

from flask import Flask
from flask_cors import CORS
from flask_restplus import Resource, Api
import dr_validation_async
import os

app = Flask(__name__)
api = Api(app, version='1.0', title='Module kiểm định dải động bức xạ ảnh (Dynamic Range (DR))', description='Kiểm định dải động bức xạ từ dữ liệu ảnh', validate=True)
CORS(app)

@api.route('/validate_dr/image_path=<image_path>', endpoint = 'validate-dynamic-range')
@api.doc(params={'image_path': 'Đường dẫn đến tập tin dữ liệu ảnh định dạng TIFF 10bits'})
class ValidateDynamicRangeResource(Resource):
    def get(self,image_path):
        if not os.path.exists(image_path) or not os.path.isfile(image_path) or not image_path.lower().endswith('.tif'):
            return {'Lỗi':"Tham số truyền vào không đúng"}

        dr_validation_async.validate_dr(image_path)
        return {"Thông báo":"Lựa chọn một vùng trên ảnh hiện ra và kiểm định dải động bức xạ bằng histogram"}

if __name__ == '__main__':
    app.run(debug=True, host = 'localhost', port= 7005)