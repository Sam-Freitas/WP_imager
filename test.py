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

print('data processing')

starting_location_xyz = [-500,-300,-103] # center of where you want to measure [-191.4,-300,-86]
pixels_per_mm = 1453.5353/5.0
pixels_per_mm = 980/5
pixels_per_mm = 192
FOV = 5
extent_x = 20 #mm
extent_y = 20 #mm
delta_x = -5 # mm # start at the top left and snake across and down
delta_y = 5 # mm this difference is to move across x and down y to make the image processing easier
y_images = int( ((extent_y-FOV)/np.abs(delta_y)) + 1 ) # left to right
x_images = int( ((extent_x-FOV)/np.abs(delta_x)) + 1 ) # up and down ########## I know that this is a wrong name but total image = x_image*y_images and i cant think of a better term right now
starting_location = dict()
starting_location['x_pos'] = round(starting_location_xyz[0],4)
starting_location['y_pos'] = round(starting_location_xyz[1],4)
starting_location['z_pos'] = round(starting_location_xyz[2],4)

images = np.load(r"Y:\Users\Sam Freitas\test.npy") # use with extent = 20
# images = np.load(r"Y:\Users\Sam Freitas\images_before.npy") # use with extend = 35
# large_img_initial = cv2.imread(r"Y:\Users\Sam Freitas\square_test.bmp")

# # # large_img = np.zeros((int(extent_y*pixels_per_mm),int(extent_x*pixels_per_mm)))

# # # # the term row and col are not exact as the system might have overlapping images, this is just nomeclature if there was zero overlap
# # # counter = 0
# # # for row in range(y_images): # rows
# # #     if row % 2:
# # #         cols = range(x_images-1,-1,-1) # odd flag
# # #     else:
# # #         cols = range(0,x_images) # even flag (includes zero)

# # #     for col in cols:
# # #         print(row,col)

# # #         frame = images[counter]
# # #         img_data_cropped = analysis.fluor_postprocess.crop_center_numpy_return(frame,pixels_per_mm*FOV)+1
# # #         temp_large_img = analysis.fluor_postprocess.put_frame_in_large_img(extent_y,extent_x,pixels_per_mm,FOV,delta_x,delta_y, counter, img_data_cropped, row, col)

# # #         # for overlapping images (still needs work)
# # #         if (row == 0) and (col == 0):
# # #             large_img = temp_large_img
# # #         else:
# # #             large_img = analysis.fluor_postprocess.average_arrays_ignore_zeros(large_img, temp_large_img)
        
# # #         # camera.camera_control.imshow_resize('img',large_img.astype(np.uint8), resize_size = [640,640])
# # #         counter += 1
# # # large_img_norm = ((large_img - large_img.min()) / (large_img.max() - large_img.min()))*255
# # # cv2.imwrite('test_unregistered.bmp', large_img_norm.astype(np.uint8))

img_num = int(np.sqrt(images.shape[0]))
images, img_data_cropped, large_img = analysis.fluor_postprocess.align_frames_register(images,pixels_per_mm,FOV,extent_x,extent_y,delta_x,delta_y,overlap = 1)

large_img_norm = analysis.fluor_postprocess.normalize_to_smallest_nonzero(large_img)*255
camera.camera_control.imshow_resize('img',large_img_norm.astype(np.uint8), resize_size = [640,640])
cv2.imwrite('test_registered.bmp',large_img_norm.astype(np.uint8))

print('eof')