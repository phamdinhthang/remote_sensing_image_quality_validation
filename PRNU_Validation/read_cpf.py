# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:13:16 2018

@author: ThangPD
"""
import xml.etree.ElementTree
import json
import os

def read_cpf_to_json(cpf_path, write_to_file=False):
    tree = xml.etree.ElementTree.parse(cpf_path)
    root = tree.getroot()
    
    pan_ds = [float(x) for x in root.find('RadiometricParameters').find('PAN').find('G1').find('DarkCurrents').text.split(',')]
    b1_ds = [float(x) for x in root.find('RadiometricParameters').find('B1').find('G1').find('DarkCurrents').text.split(',')]
    b2_ds = [float(x) for x in root.find('RadiometricParameters').find('B2').find('G1').find('DarkCurrents').text.split(',')]
    b3_ds = [float(x) for x in root.find('RadiometricParameters').find('B3').find('G1').find('DarkCurrents').text.split(',')]
    b4_ds = [float(x) for x in root.find('RadiometricParameters').find('B4').find('G1').find('DarkCurrents').text.split(',')]
    
    pan_prnu = [float(x) for x in root.find('RadiometricParameters').find('PAN').find('G1').find('DetectorRelativeGains').text.split(',')]
    b1_prnu = [float(x) for x in root.find('RadiometricParameters').find('B1').find('G1').find('DetectorRelativeGains').text.split(',')]
    b2_prnu = [float(x) for x in root.find('RadiometricParameters').find('B2').find('G1').find('DetectorRelativeGains').text.split(',')]
    b3_prnu = [float(x) for x in root.find('RadiometricParameters').find('B3').find('G1').find('DetectorRelativeGains').text.split(',')]
    b4_prnu = [float(x) for x in root.find('RadiometricParameters').find('B4').find('G1').find('DetectorRelativeGains').text.split(',')]
    
    cpf_dict = {'ds':{'PAN':pan_ds,'B1':b1_ds,'B2':b2_ds,'B3':b3_ds,'B4':b4_ds},'prnu':{'PAN':pan_prnu,'B1':b1_prnu,'B2':b2_prnu,'B3':b3_prnu,'B4':b4_prnu}}
    
    if write_to_file==True:
        cpf_folder = os.path.dirname(cpf_path)
        json_path = os.path.join(cpf_folder,'cpf.json')
        with open(json_path, 'w') as f: 
            f.write(json.dumps(cpf_dict))
        print("CPF converted to json. Check file at:", json_path)
    
    return cpf_dict

def main():
    cpf_path = 'C:/Users/admin/Google Drive/Python_NAOMI125_Validation/Python_Validation/PRNU_Validation/on_ground_cpf/VNREDSAT_1_20130321_090000_20130319_000000.CPF'
    cpf_dict = read_cpf_to_json(cpf_path,write_to_file=True)
    return cpf_dict

if __name__=='__main__':
    main()