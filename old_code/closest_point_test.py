import os,time,glob,sys,time,tqdm,cv2, serial, json, torch, scipy
import numpy as np
import lights.labjackU3_control
import lights.coolLed_control
import settings.get_settings
# import movement.simple_stream
import camera.camera_control
import analysis.fluor_postprocess
import atexit
from Run_yolo_model import yolo_model, sort_rows
from wand.image import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt

# this is the main python file to control the WP imager (Worm Paparazzi imager) imaging system
# functions:
# 1) image and stimulate the plates for lifespan and healthspan anaylsis
# 2) fluorescently image the plate

# you will need z_starting_point_array local python environment setup with cuda-torch 

def find_closest_points(points_array, new_coordinate, num_closest=1):
    # Calculate the Euclidean distances between each point and the new coordinate
    distances = np.linalg.norm(points_array - new_coordinate, axis=1)
    
    # Sort the distances and get the indices of the closest points
    closest_indices = np.argsort(distances)[:num_closest]
    
    # Get the closest points and their distances
    closest_points = points_array[closest_indices]
    closest_distances = distances[closest_indices]
    
    return closest_points, closest_indices, closest_distances

def jprint(input):
    print(json.dumps(input,indent=4))

if __name__ == "__main__":
    # this first block of code is to set up the system and make sure that it is homed and ready for all parts of the experiment

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'output')

    # read in settings from files
    s_plate_names_and_opts = settings.get_settings.get_plate_names_and_opts()
    s_plate_positions = settings.get_settings.get_plate_positions()
    s_terasaki_positions = settings.get_settings.get_terasaki_positions()
    s_wm_positions = settings.get_settings.get_wm_positions()
    s_wm_4pair_positions = settings.get_settings.get_wm_4pair_positions()
    s_machines = settings.get_settings.get_machine_settings()
    s_camera_settings = settings.get_settings.get_basic_camera_settings()
    s_todays_runs = settings.get_settings.get_todays_runs()

    s_positions = s_wm_4pair_positions.copy()
    n_image_locs = 60

    z_starting_point_array = np.zeros(shape=(n_image_locs,3))

    for well_index,this_well_location_xy in enumerate(tqdm.tqdm(zip(s_positions['x_relative_pos_mm'].values(),s_positions['y_relative_pos_mm'].values()), total = n_image_locs)):

        if well_index != 0:    
            closest_points, closest_indices, closest_distances = find_closest_points(z_starting_point_array[0:well_index,0:2], this_well_location_xy)
            # z_starting_point_array[well_index,-1] = closest_indices
        else:
            closest_indices = 0
            z_starting_point_array[well_index,0:2] = this_well_location_xy
            z_starting_point_array[well_index,2] = -89 - np.random.rand(1)

        z_starting_point_array[well_index,0:2] = this_well_location_xy
        plt.scatter(z_starting_point_array[0:well_index,0],z_starting_point_array[0:well_index,1])
        plt.plot(this_well_location_xy[0],this_well_location_xy[1],'ro')
        plt.plot(z_starting_point_array[closest_indices,0],z_starting_point_array[closest_indices,1],'ko')
        plt.show(block = False)
        plt.pause(0.25)

        print(well_index)
        if well_index < n_image_locs-1:
            plt.clf()

    print('eof')