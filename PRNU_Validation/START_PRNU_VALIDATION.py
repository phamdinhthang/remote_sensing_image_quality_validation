# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 17:36:20 2018

@author: ThangPD
"""

from flask import Flask
from flask_cors import CORS
from flask_restplus import Resource, Api
import os

import prnu_validation
import cpf_to_prnu_json
import compare_prnu
import image_prnu_correction

app = Flask(__name__)
api = Api(app, version='1.0', title='Module kiểm định và hiệu chỉnh độ khuếch đại điểm ảnh (Pixel Response Non-Uniformity)', description='Tính toán độ khuếch đại điểm ảnh từ dữ liệu ảnh và thực hiện hiệu chỉnh ảnh', validate=True)
CORS(app)

@api.route('/validate_prnu/images_folders_path=<images_folders_path>&dark_signal_json_path=<dark_signal_json_path>&on_ground_cpf_path=<on_ground_cpf_path>', endpoint = 'validate-prnu')
@api.doc(params={'images_folders_path': 'Đường dẫn đến thư mục chứa ảnh',
                 'dark_signal_json_path': 'Đường dẫn đến tập tin dòng tối (.json)',
                 'on_ground_cpf_path': 'Đường dẫn đến tập tin CPF của vệ tinh đo trên mặt đất'})
class ValidatePRNUResource(Resource):
    def get(self,images_folders_path, dark_signal_json_path, on_ground_cpf_path):
        if not os.path.exists(images_folders_path) or not os.path.isdir(images_folders_path) or not os.path.exists(dark_signal_json_path) or not os.path.isfile(dark_signal_json_path) or not dark_signal_json_path.lower().endswith('.json') or not os.path.exists(on_ground_cpf_path) or not os.path.isfile(on_ground_cpf_path) or not on_ground_cpf_path.lower().endswith('.cpf'):
            return {'Lỗi':"Các tham số truyền vào không đúng"}

        result_path = prnu_validation.validate_prnu(images_folders_path, dark_signal_json_path, on_ground_cpf_path)
        if result_path is not None:
            return {"Đường dẫn đến tập tin chứa kết quả":result_path}
        else:
            return {"Lỗi":"Không kiểm định được PRNU"}

@api.route('/convert_cpf_to_json/cpf_path=<cpf_path>', endpoint = 'convert_cpf_to_json')
@api.doc(params={'cpf_path': 'Đường dẫn đến tập tin CPF'})
class ConvertCPFToJsonResource(Resource):
    def get(self,cpf_path):
        if not os.path.exists(cpf_path) or not os.path.isfile(cpf_path) or not cpf_path.lower().endswith('.cpf'):
            return {'Lỗi':"Các tham số truyền vào không đúng"}

        result_path = cpf_to_prnu_json.convert(cpf_path)
        if result_path is not None:
            return {"Đường dẫn đến tập tin chứa kết quả":result_path}
        else:
            return {"Lỗi":"Không chuyển đổi được định dạng tập tin CPF"}
#
@api.route('/compare_prnu/json_path1=<json_path1>&json_path2=<json_path2>', endpoint = 'compare-prnu')
@api.doc(params={'json_path1': 'Đường dẫn đến tập tin PRNU (.json) thứ nhất',
                 'json_path2': 'Đường dẫn đến tập tin PRNU (.json) thứ hai'})
class ComparePRNUResource(Resource):
    def get(self,json_path1,json_path2):
        if not os.path.exists(json_path1) or not os.path.isfile(json_path1) or not json_path1.lower().endswith('.json') or not os.path.exists(json_path2) or not os.path.isfile(json_path2) or not json_path2.lower().endswith('.json'):
            return {'Lỗi':"Các tham số truyền vào không đúng"}

        compare_prnu.compare(json_path1,json_path2)
        return {"Thông báo":"Xem đồ thị so sánh PRNU trên cửa sổ hiện ra"}

@api.route('/image_prnu_correction/img_path=<img_path>&ds_json_path=<ds_json_path>&prnu_json_path=<prnu_json_path>&channel_name=<channel_name>', endpoint = 'image-prnu-correction')
@api.doc(params={'img_path': 'Đường dẫn đến tập tin dữ liệu ảnh .h5',
                 'ds_json_path': 'Đường dẫn đến tập tin chứa dòng tối (.json)',
                 'prnu_json_path': 'Đường dẫn đến tập tin chứa PRNU (.json)',
                 'channel_name':'Tên kênh phổ: [PAN hoặc B1 hoặc B2 hoặc B3 hoặc B4]'})
class ImagePRNUCorrectionResource(Resource):
    def get(self,img_path,ds_json_path,prnu_json_path,channel_name):
        if not os.path.exists(img_path) or not os.path.isfile(img_path) or not img_path.lower().endswith('.h5') or not os.path.exists(ds_json_path) or not os.path.isfile(ds_json_path) or not ds_json_path.lower().endswith('.json') or not os.path.exists(prnu_json_path) or not os.path.isfile(prnu_json_path) or not prnu_json_path.lower().endswith('.json') or channel_name not in ['PAN','B1','B2','B3','B4']:
            return {'Lỗi':"Các tham số truyền vào không đúng"}

        image_prnu_correction.correct_image_prnu(img_path,ds_json_path,prnu_json_path,channel_name)
        return {"Thông báo":"Xem so sánh dữ liệu ảnh trước và sau khi hiệu chỉnh trên cửa sổ hiện ra"}

if __name__ == '__main__':
    app.run(debug=True, host = 'localhost', port= 7001)