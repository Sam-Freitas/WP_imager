import os,time,glob,sys,time,tqdm,cv2, serial, json
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
s_machines = settings.get_settings.get_machine_settings()
s_camera_settings = settings.get_settings.get_basic_camera_settings()
s_todays_runs = settings.get_settings.get_todays_runs()

################### THIS IS FOR testing autofocus scripts ONLY

s_positions = s_wm_positions.copy()

cmap = plt.colormaps['Spectral']

plt.figure()
for well_index,this_well_location_xy in enumerate(zip(s_positions['x_relative_pos_mm'].values(),s_positions['y_relative_pos_mm'].values())):
    this_c = cmap(well_index/240)
    plt.plot(this_well_location_xy[0],this_well_location_xy[1],'o', c = this_c)

plt.show(block=False)
plt.pause(0.5)

row_names = ['A','B','C','D','E','F','G','H','I','J','K','L']
col_names = list(range(1,21))

names_of_wells = list(s_positions['name'].values())
x_positions = list(s_positions['x_relative_pos_mm'].values())
y_positions = list(s_positions['y_relative_pos_mm'].values())
for i in range(0,len(row_names),2):
    for j in range(0,len(col_names),2):

        first_well = str(row_names[i]) + str(col_names[j]) 
        fourth_well = str(row_names[i+1]) + str(col_names[j+1]) 

        export_name = str(row_names[i]) + str(col_names[j]) + '_' + str(row_names[i]) + str(col_names[j+1]) + '_' + str(row_names[i+1]) + str(col_names[j]) + '_' + str(row_names[i+1]) + str(col_names[j+1])

        idx_1 = names_of_wells.index(first_well)
        idx_4 = names_of_wells.index(fourth_well)

        x_pos = (x_positions[idx_1] + x_positions[idx_4])/2
        y_pos = (y_positions[idx_1] + y_positions[idx_4])/2

        plt.plot(x_pos,y_pos,'ko')

        print(export_name, x_pos,y_pos)
        print('')
plt.show(block = True)