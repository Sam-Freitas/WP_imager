import cv2, os, tqdm, glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted
from skimage import filters
import scipy


good_wells = ['a4','a8','a12','c7','d2','d12','e9','e11','f4','g6','g9','g11']

output = 'analysis\grid_output'

img_file_type = "png"
overarching_dir = r"Y:\Users\Sam Freitas\WP_imager\output\GLS371_testing\t_1"  # os.path.join(os.getcwd(),'captured_images')

last_n_images_to_use = 6

days_dirs = natsorted(glob.glob(os.path.join(overarching_dir,'*')))[-last_n_images_to_use:]

for i,this_well in enumerate(good_wells):

    print(this_well)

    for j,each_day in enumerate(days_dirs):

        name = os.path.split(each_day)[-1]

        imgs = np.asarray(glob.glob(os.path.join(each_day,'fluorescent_data','*.png')))
        idx = np.char.find(imgs,this_well + '_')>0
        
        this_img = cv2.imread(str(imgs[idx][0]))

        cv2.imwrite(os.path.join(output,str(this_well) + '_' + name + '.png'), this_img)

    print(this_well)