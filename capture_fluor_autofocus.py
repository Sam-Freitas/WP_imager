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

def turn_everything_off_at_exit():
    lights.labjackU3_control.turn_off_everything()
    cv2.destroyAllWindows()
    # lights.coolLed_control.turn_everything_off()

class CNCController:
    def __init__(self, port, baudrate):
        import re
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)

    def wait_for_movement_completion(self,cleaned_line):

        # print("waiting on: " + str(cleaned_line))

        if ('$X' not in cleaned_line) and ('$$' not in cleaned_line) and ('?' not in cleaned_line):
            idle_counter = 0
            time.sleep(0.1)
            while True:
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
                time.sleep(0.1)
                command = str.encode("?"+ "\n")
                self.ser.write(command)
                time.sleep(0.1)
                grbl_out = self.ser.readline().decode().strip()
                grbl_response = grbl_out.strip()

                if 'ok' not in grbl_response.lower():   
                    if 'idle' in grbl_response.lower():
                        idle_counter += 1
                    else:
                        if grbl_response != '':
                            pass
                            # print(grbl_response)
                if idle_counter == 1 or idle_counter == 2:
                    # print(grbl_response)
                    pass
                if idle_counter > 5:
                    break
                if 'alarm' in grbl_response.lower():
                    raise ValueError(grbl_response)

    def send_command(self, command):
        self.ser.reset_input_buffer() # flush the input and the output 
        self.ser.reset_output_buffer()
        time.sleep(0.1)
        self.ser.write(command.encode())
        time.sleep(0.1)

        CNCController.wait_for_movement_completion(self,command)
        out = []
        for i in range(50):
            time.sleep(0.01)
            response = self.ser.readline().decode().strip()
            time.sleep(0.01)
            out.append(response)
            if 'error' in response.lower():
                print('error--------------------------------------------------')
            if 'ok' in response:
                break
            # print(response)
        return response, out
    
    def get_current_position(self):

        command = "? " + "\n"
        response, out = CNCController.send_command(self,command)
        MPos = out[0] # get the idle output
        MPos = MPos.split('|')[1] # get the specific output
        MPos = MPos.split(',')
        MPos[0] = MPos[0].split(':')[1]

        position = dict()

        position['x_pos'] = float(MPos[0])
        position['y_pos'] = float(MPos[1])
        position['z_pos'] = float(MPos[2])

        return position
    
    def move_XY_at_Z_travel(self, position, z_travel_height):

        current_position = CNCController.get_current_position(self)

        if round(float(current_position['z_pos']),1) != float(z_travel_height):
            #### go to z travel height
            # command = "G0 z" + str(z_travel_height) + " " + "\n"
            command = "G1 z" + str(z_travel_height) + " F2500" #+ "\n"
            response, out = CNCController.send_command(self,command)
        
        print('moving to XY')
        # command = 'G0 ' + 'X' + str(position['x_pos']) + ' ' + 'Y' + str(position['y_pos']) 
        command = 'G1 ' + 'X' + str(position['x_pos']) + ' ' + 'Y' + str(position['y_pos']) + ' F2500'
        response, out = CNCController.send_command(self,command)
        ##### move z
        print('moving to Z')
        # command = 'G0 ' + 'Z' + str(position['z_pos']) 
        command = 'G1 ' + 'Z' + str(position['z_pos']) + ' F2500'
        response, out = CNCController.send_command(self,command)

        return CNCController.get_current_position(self)
    
    def move_XYZ(self, position):

        ##### move xyz
        print('moving to XYZ')
        # command = 'G0 ' + 'X' + str(position['x_pos']) + ' ' + 'Y' + str(position['y_pos']) + ' ' + 'Z' + str(position['z_pos']) 
        command = 'G1 ' + 'X' + str(position['x_pos']) + ' ' + 'Y' + str(position['y_pos']) + ' ' + 'Z' + str(position['z_pos']) + ' F2500'
        response, out = CNCController.send_command(self,command)

        return CNCController.get_current_position(self)
    
    def home_grbl(self):
        print("HOMING CNC")
        command = "$H"+ "\n"
        response, out = CNCController.send_command(self,command)
    
    def set_up_grbl(self, home = True):
        # unlock 
        command = "$X"+ "\n"
        response, out = CNCController.send_command(self,command)

        command = "?"+ "\n"
        response, out = CNCController.send_command(self,command)

        if home:
            CNCController.home_grbl(self)

    def close_connection(self):
        self.ser.close()

def jprint(input):
    print(json.dumps(input,indent=4))
    
def run_calib(s_camera_settings,this_plate_parameters,output_dir,s_terasaki_positions, calibration_model, adjust_with_movement = True, final_measurement = False, delete_prev_data = True):
    # take image
    image_filename = camera.camera_control.simple_capture_data_single_image(s_camera_settings, plate_parameters=this_plate_parameters,
                                output_dir=output_dir, image_file_format = 'jpg', testing = delete_prev_data)
    # run yolo model and get the locations of the well and center of the plate
    individual_well_locations,center_location,n_wells = calibration_model.run_yolo_model(img_filename=image_filename, save_results = True, show_results = True, plate_index = this_plate_parameters['plate_index'])

    # Calculate pixels per mm based on well location data
    well_locations_delta = individual_well_locations[-1] - individual_well_locations[0]
    pixels_per_mm = well_locations_delta / [69.7, 42]

    # Calculate center and its delta in mm
    center = [float(s_camera_settings['widefield'][1]) / 2, float(s_camera_settings['widefield'][2]) / 2]
    center_delta = center - center_location
    center_delta_in_mm = center_delta / pixels_per_mm

    # Create calibration coordinates dictionary
    calibration_coordinates = {
        'x_pos': center_delta_in_mm[0],
        'y_pos': center_delta_in_mm[1],
        'z_pos': s_terasaki_positions['calib_z_pos_mm'][0]
    }

    # Adjust position based on calibration
    measured_position = controller.get_current_position()
    adjusted_position = measured_position.copy()
    adjusted_position['x_pos'] += center_delta_in_mm[0]
    adjusted_position['y_pos'] -= center_delta_in_mm[1]

    if adjust_with_movement:
        controller.move_XYZ(position = adjusted_position)
    
    if final_measurement:
        move_down = measured_position.copy()
        move_down['z_pos'] = -106.5
        controller.move_XYZ(position = move_down)
        image_filename = camera.camera_control.simple_capture_data_single_image(s_camera_settings, plate_parameters=this_plate_parameters,
                            output_dir=output_dir, image_file_format = 'jpg', testing = delete_prev_data)
        individual_well_locations,center_location,n_wells = calibration_model.run_yolo_model(img_filename=image_filename, show_results = True, save_results=True, plate_index = this_plate_parameters['plate_index'])
    
    return adjusted_position

import atexit

if __name__ == "__main__":

    # this is a test to try and capture a square of overlapping images
    # then convert them into a single large image(?????)
    
    atexit.register(turn_everything_off_at_exit)
    # print(sys.argv)

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'output')

    # read in settings from files
    s_plate_names_and_opts = settings.get_settings.get_plate_names_and_opts()
    s_plate_positions = settings.get_settings.get_plate_positions()
    s_terasaki_positions = settings.get_settings.get_terasaki_positions()
    s_wm_positions = settings.get_settings.get_wm_positions()
    s_machines = settings.get_settings.get_machine_settings()
    s_camera_settings = settings.get_settings.get_basic_camera_settings()
    s_todays_runs = settings.get_settings.get_todays_runs()

    # calibration_model = yolo_model()

    starting_location_xyz = [-155-170,-35,-89] # center of where you want to measure [-191.4,-300,-86]
    pixels_per_mm = 192

    FOV = 5
  
    autofocus_min_max = [5,-5] # remember that down (towards sample) is negative
    autofocus_delta_z = 0.25 # mm 
    autofocus_steps = int(abs(np.diff(autofocus_min_max) / autofocus_delta_z)) + 1

    z_limit = [-5,-94]

    z_positions = np.linspace(starting_location_xyz[2]+autofocus_min_max[0],starting_location_xyz[2]+autofocus_min_max[1],num = autofocus_steps)

    starting_location = dict()
    starting_location['x_pos'] = round(starting_location_xyz[0],4)
    starting_location['y_pos'] = round(starting_location_xyz[1],4)
    starting_location['z_pos'] = round(starting_location_xyz[2],4)

    settings.get_settings.check_grbl_port(s_machines['grbl'][0], run_as_testing = False)
    controller = CNCController(port=s_machines['grbl'][0], baudrate=s_machines['grbl'][1])
    response, s_grbl_settings = controller.send_command("$$"+ "\n")
    s_grbl_settings_df,s_grbl_settings = settings.get_settings.convert_GRBL_settings(s_grbl_settings)
    z_travel_height = s_machines['grbl'][2]

    coolLED_port = s_machines['coolLed'][0] # test the fluorescent lights (if applicable)
    lights.coolLed_control.turn_everything_off(coolLED_port)

    # run setup test to make sure everything works or throw error
    s_todays_runs = settings.get_settings.update_todays_runs(s_todays_runs, overwrite=True)
    d = lights.labjackU3_control.setup_labjack(verbose=True)    # test the blue and red lights
    lights.labjackU3_control.blink_led(d)

    lights.labjackU3_control.turn_on_red(d)
    controller.set_up_grbl(home = True)
    lights.labjackU3_control.turn_off_everything(d)

    lights.coolLed_control.turn_specified_on(coolLED_port, 
        uv = False, 
        uv_intensity = 100,
        blue = True, 
        blue_intensity = 100,
        green = False, 
        green_intensity = 100,
        red = False, 
        red_intensity = 100)

    controller.move_XY_at_Z_travel(starting_location, z_travel_height)

    images = []
    for counter,z_pos in enumerate(tqdm.tqdm(z_positions)):
        this_location = starting_location.copy()
        this_location['z_pos'] = z_pos
        jprint(this_location)

        if z_pos < z_limit[0] and z_pos > z_limit[1]:

            controller.move_XYZ(position = this_location)
                
            if counter == 0:
                frame, cap = camera.camera_control.capture_fluor_img_return_img(s_camera_settings, return_cap = True)
            else:
                frame, cap = camera.camera_control.capture_fluor_img_return_img(s_camera_settings, cap = cap,return_cap = True)
            images.append(frame)

    lights.labjackU3_control.turn_off_everything(d)
    lights.coolLed_control.turn_everything_off(coolLED_port)

    images = np.asarray(images)
    np.save('autofocus_stack.npy',images)

    a = np.mean(images, axis = 0) # get the average image taken of the stack (for illumination correction)
    binary_img = analysis.fluor_postprocess.largest_blob(a > 3) # get the largest binary blob in the image
    center = [ np.average(indices) for indices in np.where(binary_img) ] # find where the actual center of the frame is 
    center_int = [int(np.round(point)) for point in center]

    norm_array = scipy.ndimage.gaussian_filter(a,100) # get the instensities of the images for the illuminance normalizations
    norm_array_full = 1-(norm_array/np.max(norm_array))
    norm_array = analysis.fluor_postprocess.crop_center_numpy_return(norm_array_full,pixels_per_mm*(FOV), center = center_int)

    focus_score = []
    for this_img in images:
        this_img = this_img*(norm_array_full+1)
        img_data_cropped = analysis.fluor_postprocess.crop_center_numpy_return(this_img,pixels_per_mm*(FOV), center = center_int)
        b = scipy.ndimage.sobel(img_data_cropped.astype(np.float32))
        this_fscore = np.max(np.abs(b))
        focus_score.append(this_fscore)

    assumed_focus = np.argmax(focus_score)

    plt.subplot(1,2,1)
    plt.title('assumed focus:' + str(assumed_focus))
    plt.imshow(images[assumed_focus]*(norm_array_full+1))
    plt.subplot(1,2,2)
    plt.title('plot of focus socre')
    plt.plot(focus_score)
    plt.plot(assumed_focus,focus_score[assumed_focus],'ro')
    plt.show()
            
    # cv2.imwrite('square_test ' + str(int(pixels_per_mm)) + '.bmp', large_img_norm.astype(np.uint8))
    

    lights.labjackU3_control.turn_off_everything(d)
    lights.coolLed_control.turn_everything_off(coolLED_port)


    print('eof') 
