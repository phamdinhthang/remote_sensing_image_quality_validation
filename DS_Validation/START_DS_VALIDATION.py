# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:07:02 2018

@author: ThangPD
"""

from flask import Flask
from flask_cors import CORS
from flask_restplus import Resource, Api
import ds_validation
import cpf_to_ds_json
import compare_ds
import image_ds_correction
import os

app = Flask(__name__)
api = Api(app, version='1.0', title='Module kiểm định và hiệu chỉnh dòng tối (Dark signal)', description='Tính toán dòng tối từ dữ liệu ảnh và thực hiện hiệu chỉnh ảnh', validate=True)
CORS(app)

@api.route('/validate_dark_signal/images_folders_path=<images_folders_path>', endpoint = 'validate-dark-signal')
@api.doc(params={'images_folders_path': 'Đường dẫn đến thư mục chứa dữ liệu ảnh'})
class ValidateDarkSignalResource(Resource):
    def get(self,images_folders_path):
        if not os.path.exists(images_folders_path) or not os.path.isdir(images_folders_path):
            return {'Lỗi':"Các tham số truyền vào không đúng"}

        result_path = ds_validation.validate_dark_signal(images_folders_path)
        if result_path is not None:
            return {"Đường dẫn đến tập tin kết quả":result_path}
        else:
            return {"Lỗi":"Không kiểm định được dòng tối"}

@api.route('/convert_cpf_to_json/cpf_path=<cpf_path>', endpoint = 'convert_cpf_to_json')
@api.doc(params={'cpf_path': 'Đường dẫn đến tập tin CPF'})
class ConvertCPFToJsonResource(Resource):
    def get(self,cpf_path):
        if not os.path.exists(cpf_path) or not os.path.isfile(cpf_path) or not cpf_path.lower().endswith('.cpf'):
            return {'Lỗi':"Các tham số truyền vào không đúng"}

        result_path = cpf_to_ds_json.convert(cpf_path)
        if result_path is not None:
            return {"Đường dẫn đến file Json":result_path}
        else:
            return {"Lỗi":"Không chuyển đổi được tập tin CPF"}

@api.route('/compare_dark_signal/json_path1=<json_path1>&json_path2=<json_path2>', endpoint = 'compare-dark-signal')
@api.doc(params={'json_path1': 'Đường dẫn đến thập tin dòng tối (.json) thứ nhất',
                 'json_path2': 'Đường dẫn đến thập tin dòng tối (.json) thứ hai'})
class CompareDarkSignalResource(Resource):
    def get(self,json_path1,json_path2):
        if not os.path.exists(json_path1) or not os.path.isfile(json_path1) or not json_path1.lower().endswith('.json') or not os.path.exists(json_path2) or not os.path.isfile(json_path2) or not json_path2.lower().endswith('.json'):
            return {'Lỗi':"Các tham số truyền vào không đúng"}

        compare_ds.compare(json_path1,json_path2)
        return {"Thông báo":"Xem đồ thị so sánh dòng tối trên cửa sổ hiện ra"}

@api.route('/image_dark_signal_correction/img_path=<img_path>&json_path=<json_path>&channel_name=<channel_name>', endpoint = 'image-dark-signal-correction')
@api.doc(params={'img_path': 'Đường dẫn đến tập tin dữ liệu ảnh dạng .h5',
                 'json_path': 'Đường dẫn đến thập tin dòng tối (.json)',
                 'channel_name':'Tên kênh phổ: [PAN hoặc B1 hoặc B2 hoặc B3 hoặc B4]'})
class ImageDarkSignalCorrectionResource(Resource):
    def get(self,img_path,json_path,channel_name):
        if not os.path.exists(img_path) or not os.path.isfile(img_path) or not img_path.lower().endswith('.h5') or not os.path.exists(json_path) or not os.path.isfile(json_path) or not json_path.lower().endswith('.json') or channel_name not in ['PAN','B1','B2','B3','B4']:
            return {'Lỗi':"Các tham số truyền vào không đúng"}

        image_ds_correction.correct_image_ds(img_path,json_path,channel_name)
        return {"Thông báo":"Xem so sánh ảnh trước và sau khi hiệu chỉnh trên cửa sổ hiện ra"}

if __name__ == '__main__':
    app.run(debug=True, host = 'localhost', port= 7000)