import os,time,glob,sys,time,tqdm,cv2, serial, json, shutil
import numpy as np
import lights.labjackU3_control
import lights.coolLed_control
import settings.get_settings
import movement.simple_stream
import camera.camera_control
import analysis.fluor_postprocess

from Run_yolo_model import yolo_model

import pandas as pd
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy

# read in settings from files
s_plate_names_and_opts = settings.get_settings.get_plate_names_and_opts()
s_plate_positions = settings.get_settings.get_plate_positions()
s_terasaki_positions = settings.get_settings.get_terasaki_positions()
s_wm_positions = settings.get_settings.get_wm_positions()
s_wm_4pair_positions = settings.get_settings.get_wm_4pair_positions()
s_machines = settings.get_settings.get_machine_settings()
s_camera_settings = settings.get_settings.get_basic_camera_settings()
s_todays_runs = settings.get_settings.get_todays_runs()


N = 25

output_dir = os.path.join(r"C:\Users\LabPC2\Desktop\tiff_testing",'saving_test')
os.makedirs(output_dir, exist_ok = True)
folder = output_dir
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
os.makedirs(output_dir, exist_ok = True)

s_camera_settings['fluorescence'][7] = 'png'
start_time = time.time()
this_plate_parameters,this_plate_position = settings.get_settings.get_indexed_dict_parameters(s_plate_names_and_opts,s_plate_positions,0)
for i in tqdm.tqdm(range(N)):

    this_plate_parameters['name'] = str(i)
    this_plate_parameters['well_name'] = str(i)

    camera.camera_control.simple_capture_data_fluor(s_camera_settings, plate_parameters=this_plate_parameters, testing=False, output_dir=output_dir, cap = None, return_cap = True)

    # if i == 0:
    #     cap = camera.camera_control.simple_capture_data_fluor(s_camera_settings, plate_parameters=this_plate_parameters, testing=False, output_dir=output_dir, cap = None, return_cap = True)
    # else:
    #     cap = camera.camera_control.simple_capture_data_fluor(s_camera_settings, plate_parameters=this_plate_parameters, testing=False, output_dir=output_dir, cap = cap, return_cap = True)

print('NOT returning cap WITH PNG')
print('elapsed', time.time()-start_time)
print('')
print('')
folder = output_dir
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
os.makedirs(output_dir, exist_ok = True)

s_camera_settings['fluorescence'][7] = 'png'
s_camera_settings['fluorescence'][6] = 3
start_time = time.time()
this_plate_parameters,this_plate_position = settings.get_settings.get_indexed_dict_parameters(s_plate_names_and_opts,s_plate_positions,0)
for i in tqdm.tqdm(range(N)):

    this_plate_parameters['name'] = str(i)
    this_plate_parameters['well_name'] = str(i)

    # camera.camera_control.simple_capture_data_fluor(s_camera_settings, plate_parameters=this_plate_parameters, testing=False, output_dir=output_dir, cap = None, return_cap = True)

    if i == 0:
        cap = camera.camera_control.simple_capture_data_fluor(s_camera_settings, plate_parameters=this_plate_parameters, testing=False, output_dir=output_dir, cap = None, return_cap = True)
    else:
        cap = camera.camera_control.simple_capture_data_fluor(s_camera_settings, plate_parameters=this_plate_parameters, testing=False, output_dir=output_dir, cap = cap, return_cap = True)

print('YES returning cap WITH ' + str(s_camera_settings['fluorescence'][7]))
print('elapsed', time.time()-start_time)