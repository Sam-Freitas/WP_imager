import os,time,glob,sys,time,tqdm,cv2, serial, json, torch
import numpy as np
import lights.labjackU3_control
import lights.coolLed_control
import settings.get_settings
# import movement.simple_stream
import camera.camera_control
import atexit

from Run_yolo_model import yolo_model, sort_rows
from wand.image import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt

def turn_everything_off_at_exit():
    lights.labjackU3_control.turn_off_everything()
    cv2.destroyAllWindows()
    # lights.coolLed_control.turn_everything_off()

def run_calib(s_camera_settings,this_plate_parameters,output_dir,s_terasaki_positions, calibration_model, adjust_with_movement = True, final_measurement = False, delete_prev_data = True):
    # take image
    image_filename = camera.camera_control.simple_capture_data_single_image(s_camera_settings, plate_parameters=this_plate_parameters,
                                output_dir=output_dir, image_file_format = 'jpg', testing = delete_prev_data)
    # run yolo model and get the locations of the well and center of the plate
    individual_well_locations,center_location = calibration_model.run_yolo_model(img_filename=image_filename, save_results = True, show_results = True, plate_index = this_plate_parameters['plate_index'])

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
        individual_well_locations,center_location = calibration_model.run_yolo_model(img_filename=image_filename, plot_results = True, plate_index = this_plate_parameters['plate_index'])
    
    return adjusted_position

def run_calib_terasaki(s_camera_settings,this_plate_parameters,output_dir,s_terasaki_positions, calibration_model, adjust_with_movement = True, final_measurement = False, delete_prev_data = True):

    image_filename = camera.camera_control.simple_capture_data_fluor_single_image(s_camera_settings, plate_parameters=this_plate_parameters,
                                output_dir=output_dir, image_file_format = 'jpg', testing = delete_prev_data)

    img = cv2.imread(image_filename)
    IMAGE_H,IMAGE_W,_ = img.shape
    resize_H,resize_W = 704,704 # get image shape and shape to be transformed into
    small_size = [84,106] 

    base = np.zeros(shape=(resize_H,resize_W,3), dtype='uint8')
    base[int((resize_H/2)-(small_size[1]/2)):int((resize_H/2)+(small_size[1]/2)),int((resize_W/2)-(small_size[0]/2)):int((resize_W/2)+(small_size[0]/2)),:] = cv2.resize(img,(small_size[0],small_size[1]))

    img_in = base.copy()
    img_in = cv2.resize(img_in,(resize_H,resize_W))
    img_in = torch.permute(torch.tensor(img_in),(2,0,1)).to(torch.float32)/255
    img_in = img_in.to(device).unsqueeze(0) # make the image into the correct form

    out = calibration_model.only_model(img_in) ########## x,y,w,h,conf,class0,class1

    conf = out[0][:,4] # get the confidence values of the detected stuff
    temp = out[0][conf>0.2,:].cpu() ############# i think its x,y,w,h,conf,class0,class1
    conf2 = temp[:,4] # get the other confidence intervals

    size_of_detection = (temp[:,2]*temp[:,3]).unsqueeze(1) # get the weighting for the new detections 
    weighted_size_conf = size_of_detection*conf2.unsqueeze(1)
    weighted_size_conf = (weighted_size_conf/weighted_size_conf.max()).squeeze()
    weighted_size_conf = weighted_size_conf*weighted_size_conf # new weighting ------> confidence*((w*h)^2)

    center_of_well = np.array(np.average(temp[:,0:2], axis=0, weights=weighted_size_conf))
    center_of_image = [resize_H/2,resize_W/2]

    # plt.imshow(base)
    # plt.plot(center_of_image[0],center_of_image[1],'gx')
    # plt.plot(center_of_well[0],center_of_well[1],'r+')
    # plt.show()

    # Calculate pixels per mm based on well location data
    center_delta = center_of_image - center_of_well
    pixels_per_mm = [12.5,10.1] # row col
    center_delta_in_mm = center_delta / pixels_per_mm

    # Adjust position based on calibration
    measured_position = controller.get_current_position()
    adjusted_position = measured_position.copy()
    adjusted_position['x_pos'] += center_delta_in_mm[0]
    adjusted_position['y_pos'] -= center_delta_in_mm[1]

    if adjust_with_movement:
        controller.move_XYZ(position = adjusted_position)
        image_filename = camera.camera_control.simple_capture_data_fluor_single_image(s_camera_settings, plate_parameters=this_plate_parameters,
                                output_dir=output_dir, image_file_format = 'jpg', testing = delete_prev_data)

    return adjusted_position, center_delta_in_mm

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

def correct_barrel_distortion(img_path, a = 0.0, b = 0.0, c = 0.0, d = 0.0):

    with Image(filename = img_path) as img:
        args = (a,b,c,d)
        img.distort("barrel",args)
        img_out = np.asarray(img)

    img_name = os.path.basename(img_path)[:-4] + '_modified' + os.path.basename(img_path)[-4:]
    new_img_path = os.path.join(os.path.split(img_path)[0],img_name)
    cv2.imwrite(new_img_path,img_out)
 
    return new_img_path

def jprint(input):
    print(json.dumps(input,indent=4))

if __name__ == "__main__":

    atexit.register(turn_everything_off_at_exit)
    # print(sys.argv)

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'output')

    # read in settings from files
    s_plate_names_and_opts = settings.get_settings.get_plate_names_and_opts()
    s_plate_positions = settings.get_settings.get_plate_positions()
    s_terasaki_positions = settings.get_settings.get_terasaki_positions()
    s_machines = settings.get_settings.get_machine_settings()
    s_camera_settings = settings.get_settings.get_basic_camera_settings()
    s_todays_runs = settings.get_settings.get_todays_runs()

    # read in settings from machines
    run_as_testing = False
    home_setting = True ############################################################################## make sure this is true in production robot

    settings.get_settings.check_grbl_port(s_machines['grbl'][0], run_as_testing = False)
    controller = CNCController(port=s_machines['grbl'][0], baudrate=s_machines['grbl'][1])
    response, s_grbl_settings = controller.send_command("$$"+ "\n")
    s_grbl_settings_df,s_grbl_settings = settings.get_settings.convert_GRBL_settings(s_grbl_settings)
    z_travel_height = s_machines['grbl'][2]

    # run setup test to make sure everything works or throw error
    s_todays_runs = settings.get_settings.update_todays_runs(s_todays_runs, overwrite=True)
    d = lights.labjackU3_control.setup_labjack(verbose=True)    # test the blue and red lights
    lights.labjackU3_control.blink_led(d)
    coolLED_port = s_machines['coolLed'][0] # test the fluorescent lights (if applicable)
    lights.coolLed_control.test_coolLed_connection(coolLED_port, testing= False)

    # set up the calibration model (YOLO)
    calibration_model = yolo_model()

    # get all the experiments that are not defunct
    plate_index = []
    plate_index_fluor = []
    for this_plate_index in s_plate_names_and_opts['plate_index']:
        this_plate_name = s_plate_names_and_opts['plate_name'][this_plate_index]
        if this_plate_name != 'NONE':
            if s_plate_names_and_opts['lifespan'][this_plate_index]:
                plate_index.append(this_plate_index)
            if s_plate_names_and_opts['fluorescence'][this_plate_index]:
                plate_index_fluor.append(this_plate_index)

    lights.labjackU3_control.turn_on_red(d)
    controller.set_up_grbl(home = home_setting)
    # # # run lifespan imaging experiments
    for this_plate_index in plate_index:
        # get the experiment options
        this_plate_parameters,this_plate_position = settings.get_settings.get_indexed_dict_parameters(s_plate_names_and_opts,s_plate_positions,this_plate_index)
        print(this_plate_parameters)
        print(this_plate_position)

        # get the position of the experiment
        position = this_plate_position.copy()
        position['x_pos'],position['y_pos'],position['z_pos'] = round(position['x_pos'],4), round(position['y_pos'],4), round(position['z_pos'],4)

        # move the imaging module to the position
        controller.move_XY_at_Z_travel(position = position,
                                       z_travel_height = z_travel_height)
        
        # # # image the experiment 
        camera.camera_control.simple_capture_data(s_camera_settings, plate_parameters=this_plate_parameters, testing=run_as_testing, output_dir=output_dir)
        # # # turn on blue excitation light and capture a single image
        t = lights.labjackU3_control.turn_on_blue(d, return_time=True)
        camera.camera_control.capture_single_image_wait_N_seconds(s_camera_settings, timestart=t, excitation_amount = s_machines['labjack'][3], 
                                                                  plate_parameters=this_plate_parameters, testing=False, output_dir=output_dir)
        lights.labjackU3_control.turn_off_blue(d)
        # # # image the experiment 
        camera.camera_control.simple_capture_data(s_camera_settings, plate_parameters=this_plate_parameters, testing=False, output_dir=output_dir)
        lights.labjackU3_control.turn_off_blue(d)

        time.sleep(0.05)
        print('')

    # blink the red light as a confirmation of finish
    lights.labjackU3_control.turn_on_red(d)
    time.sleep(0.5)
    lights.labjackU3_control.turn_off_everything(d)
    # reset and home the machine
    if len(plate_index) != 0: # if there are no plate lifespan then no need to home twice
        controller.set_up_grbl(home = home_setting)

    # # # run fluorescent imaging experiments
    for this_plate_index in plate_index_fluor:
        # get the experiment options
        this_plate_parameters,this_plate_position = settings.get_settings.get_indexed_dict_parameters(s_plate_names_and_opts,s_plate_positions,this_plate_index)
        print(this_plate_parameters)
        print(this_plate_position)

        # get the position that the imaging head will be moved to 
        position = this_plate_position.copy()
        position['x_pos'],position['y_pos'],position['z_pos'] = round(position['x_pos'],4), round(position['y_pos'],4), round(position['z_pos'],4)

        # move the imaging head to the experiment
        controller.move_XY_at_Z_travel(position = position,
                                       z_travel_height = z_travel_height)

        # turn on red for the calibration run
        lights.labjackU3_control.turn_on_red(d)
        # this calibrates the imaging head to the center of the plate
        if run_as_testing:
            adjusted_position = controller.get_current_position()
            individual_well_locations,center_location = calibration_model.run_yolo_model(img_filename=None, save_results = True, show_results = True)
        else:
            adjusted_position = run_calib(s_camera_settings,this_plate_parameters,output_dir,s_terasaki_positions,calibration_model)
            adjusted_position = run_calib(s_camera_settings,this_plate_parameters,output_dir,s_terasaki_positions,calibration_model)
            # capture a single image for calibration
            image_filename = camera.camera_control.simple_capture_data_single_image(s_camera_settings, plate_parameters=this_plate_parameters, output_dir=output_dir, image_file_format = 'jpg')
            image_filename = correct_barrel_distortion(image_filename, a = 0.0, b = 0.0, c = -0.03, d = 1.05)
            individual_well_locations,center_location = calibration_model.run_yolo_model(img_filename=image_filename, save_results = True, show_results = True)

        # turn off red
        lights.labjackU3_control.turn_off_red(d)

        ########### calibration attempt
        # get the dx dy of the measured well centers
        well_locations_delta = individual_well_locations[-1]-individual_well_locations[0]
        # get measuring stick
        pixels_per_mm = well_locations_delta/[69.8,41.95]
        # find the realtion between the measured and where it supposed to be currently
        center = [float(s_camera_settings['widefield'][1])/2,float(s_camera_settings['widefield'][2])/2]
        center_delta = center-center_location
        center_delta_in_mm = center_delta/pixels_per_mm

        # calculate the calibration corner coordinates
        calibration_coordinates = dict()
        calibration_coordinates['x_pos'] = center_delta_in_mm[0]
        calibration_coordinates['y_pos'] = center_delta_in_mm[1]
        calibration_coordinates['z_pos'] = s_terasaki_positions['calib_z_pos_mm'][0]

        # move up the imaging head so it doesnt crash into the plate (very important)
        z_pos = controller.get_current_position()
        z_pos['z_pos'] = z_travel_height
        controller.move_XYZ(position = z_pos)

        lights.labjackU3_control.turn_on_red(d)

        try:
            use_adjusted_centers = True
            centers = (sort_rows(individual_well_locations)-center_location)/pixels_per_mm
        except:
            use_adjusted_centers = False
            print('Couldnt find all wells reverting to default')

        # fluorescently image each of the terasaki wells (96)
        for well_index,this_terasaki_well_xy in enumerate(zip(s_terasaki_positions['x_relative_pos_mm'].values(),s_terasaki_positions['y_relative_pos_mm'].values())):
            # get plate parameters
            this_plate_parameters['well_name'] = s_terasaki_positions['name'][well_index]
            terasaki_well_coords = dict()

            if use_adjusted_centers:
                # calculate the specific well location
                terasaki_well_coords['x_pos'] = adjusted_position['x_pos'] + calibration_coordinates['x_pos'] 
                terasaki_well_coords['y_pos'] = adjusted_position['y_pos'] + calibration_coordinates['y_pos'] + s_terasaki_positions['y_offset_to_fluor_mm'][0]

                terasaki_well_coords['x_pos'] += centers[well_index,0] #this_terasaki_well_xy[0]
                terasaki_well_coords['x_pos'] += 0.85
                terasaki_well_coords['y_pos'] += centers[well_index,1] #this_terasaki_well_xy[1]
                terasaki_well_coords['y_pos'] += 1.5
            else:
                terasaki_well_coords['x_pos'] = adjusted_position['x_pos']
                terasaki_well_coords['y_pos'] = adjusted_position['y_pos'] + s_terasaki_positions['y_offset_to_fluor_mm'][0]
                terasaki_well_coords['x_pos'] += this_terasaki_well_xy[0] + -0.85
                terasaki_well_coords['y_pos'] += this_terasaki_well_xy[1] + -2.5

            terasaki_well_coords['z_pos'] = calibration_coordinates['z_pos']
            print(well_index, terasaki_well_coords)
            # move the fluorescent imaging head to that specific well  

            if use_adjusted_centers: # if the first one get a bse measurement for all the rest
                controller.move_XYZ(position = terasaki_well_coords)
                lights.labjackU3_control.turn_on_red(d)
                terasaki_adjusted_position, center_delta_in_mm = run_calib_terasaki(s_camera_settings,this_plate_parameters,output_dir,s_terasaki_positions,calibration_model)
                lights.labjackU3_control.turn_off_everything(d)
            else: # otherswise measure then report finding and then adjust from the inital base
                controller.move_XYZ(position = terasaki_well_coords)
                # lights.labjackU3_control.turn_on_red(d)
                # terasaki_adjusted_position, center_delta_in_mm = run_calib_terasaki(s_camera_settings,this_plate_parameters,output_dir,s_terasaki_positions,calibration_model)
                lights.labjackU3_control.turn_off_everything(d)

            if run_as_testing:
                this_plate_parameters['fluorescence_UV']

            lights.coolLed_control.turn_specified_on(coolLED_port, uv = int(this_plate_parameters['fluorescence_UV']) > 0, uv_intensity = int(this_plate_parameters['fluorescence_UV']),
                                                                    blue = int(this_plate_parameters['fluorescence_BLUE']) > 0, blue_intensity = int(this_plate_parameters['fluorescence_BLUE']),
                                                                    green = int(this_plate_parameters['fluorescence_GREEN']) > 0, green_intensity = int(this_plate_parameters['fluorescence_GREEN']),
                                                                    red = int(this_plate_parameters['fluorescence_RED']) > 0, red_intensity = int(this_plate_parameters['fluorescence_RED']))
            camera.camera_control.simple_capture_data_fluor(s_camera_settings, plate_parameters=this_plate_parameters, testing=False, output_dir=output_dir)
            lights.coolLed_control.turn_everything_off(coolLED_port)

            print('imaging')
            time.sleep(0.1)

    # shut everything down 
    controller.set_up_grbl(home = True)
    # movement.simple_stream.home_GRBL(s_machines['grbl'][0], testing = False,camera=None) # home the machine

    # print('eof')