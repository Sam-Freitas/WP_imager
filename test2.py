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

s_camera_settings = settings.get_settings.get_basic_camera_settings()

################### THIS IS FOR testing autofocus scripts ONLY

starting_location_xyz = [-500,-300,-103] # center of where you want to measure [-191.4,-300,-86]
pixels_per_mm = 192

FOV = 5

autofocus_min_max = [5,-5] # remember that down (towards sample) is negative
autofocus_delta_z = 0.5 # mm 
autofocus_steps = int(abs(np.diff(autofocus_min_max) / autofocus_delta_z)) + 1

z_limit = [-5,-108]

z_positions = np.linspace(starting_location_xyz[2]+autofocus_min_max[0],starting_location_xyz[2]+autofocus_min_max[1],num = autofocus_steps)

starting_location = dict()
starting_location['x_pos'] = round(starting_location_xyz[0],4)
starting_location['y_pos'] = round(starting_location_xyz[1],4)
starting_location['z_pos'] = round(starting_location_xyz[2],4)

# controller.move_XY_at_Z_travel(starting_location, z_travel_height)

images = []
for counter,z_pos in enumerate(z_positions):
    print(counter)
    this_location = starting_location.copy()
    this_location['z_pos'] = z_pos

    if z_pos < z_limit[0] and z_pos > z_limit[1]:

        # controller.move_XYZ(position = this_location)
        print(this_location)
            
        if counter == 0:
            frame, cap = camera.camera_control.capture_fluor_img_return_img(s_camera_settings, return_cap = True)
        else:
            frame, cap = camera.camera_control.capture_fluor_img_return_img(s_camera_settings, cap = cap,return_cap = True)
        images.append(frame)


# cv2.imwrite('square_test ' + str(int(pixels_per_mm)) + '.bmp', large_img_norm.astype(np.uint8))
np.save('autofocus_stack.npy',np.asarray(images))

lights.labjackU3_control.turn_off_everything(d)
lights.coolLed_control.turn_everything_off(coolLED_port)
print('eof')