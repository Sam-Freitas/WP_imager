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

################### THIS IS FOR testing autofocus scripts ONLY

starting_location_xyz = [-500,-300,-103] # center of where you want to measure [-191.4,-300,-86]
pixels_per_mm = 192

FOV = 5

autofocus_min_max = [5,-5] # remember that down (towards sample) is negative
autofocus_delta_z = 0.5 # mm 
autofocus_steps = int(abs(np.diff(autofocus_min_max) / autofocus_delta_z)) + 1

z_positions = np.linspace(starting_location_xyz[2]+autofocus_min_max[0],starting_location_xyz[2]+autofocus_min_max[1],num = autofocus_steps)

print('eof')