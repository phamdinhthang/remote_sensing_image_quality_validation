# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 17:04:06 2018

@author: ThangPD
"""

import xml.etree.ElementTree
import json
import os
import argparse

def read_cpf_to_json(cpf_path, write_to_file=False):
    tree = xml.etree.ElementTree.parse(cpf_path)
    root = tree.getroot()

    pan_ds = [float(x) for x in root.find('RadiometricParameters').find('PAN').find('G1').find('DarkCurrents').text.split(',')]
    b1_ds = [float(x) for x in root.find('RadiometricParameters').find('B1').find('G1').find('DarkCurrents').text.split(',')]
    b2_ds = [float(x) for x in root.find('RadiometricParameters').find('B2').find('G1').find('DarkCurrents').text.split(',')]
    b3_ds = [float(x) for x in root.find('RadiometricParameters').find('B3').find('G1').find('DarkCurrents').text.split(',')]
    b4_ds = [float(x) for x in root.find('RadiometricParameters').find('B4').find('G1').find('DarkCurrents').text.split(',')]

    ds_dic = {'PAN':pan_ds,'B1':b1_ds,'B2':b2_ds,'B3':b3_ds,'B4':b4_ds}

    if write_to_file==True:
        src_path = os.path.abspath(os.path.dirname(__file__))
        json_path = os.path.join(src_path,'cpf_ds.json')

        if os.path.exists(json_path) and os.path.isfile(json_path):
            os.remove(json_path)

        with open(json_path, 'w') as f:
            f.write(json.dumps(ds_dic))
        print("CPF converted to dark signal json. Check file at:", json_path)
        return json_path
    else: return None


def convert(cpfpath):
    json_path = read_cpf_to_json(cpfpath,write_to_file=True)
    return json_path

if __name__=='__main__':
    """
    single parameter: path to CPF files
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('cpfpath', help='String CPFpath')
    args = parser.parse_args()
    cpfpath = args.cpfpath
    convert(cpfpath)