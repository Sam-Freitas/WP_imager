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

def turn_everything_off_at_exit():
    lights.labjackU3_control.turn_off_everything()
    cv2.destroyAllWindows()
    lights.coolLed_control.turn_everything_off('COM6')

def sq_grad(img,thresh = 50,offset = 10):

    shift = int(0-offset)
    offset = int(offset)

    img1 = img[:,0:shift].astype(np.float32)
    img2 = img[:,offset:].astype(np.float32)

    diff = np.abs(img2-img1)
    mask = diff > thresh
    squared_gradient = diff*diff*mask

    # img3 = img[0:shift,:]
    # img4 = img[offset:,:]

    # diff1 = np.abs(img2-img1)
    # diff2 = np.abs(img4-img3)
    # mask1 = diff1 > thresh
    # mask2 = diff2 > thresh

    # squared_gradient = diff1*diff2*mask1*mask2

    return squared_gradient

def run_calib(s_camera_settings,this_plate_parameters,output_dir, calibration_model, 
    adjust_with_movement = True, final_measurement = False, delete_prev_data = True):
    # take image
    image_filename = camera.camera_control.simple_capture_data_single_image(s_camera_settings, plate_parameters=this_plate_parameters,
                                output_dir=output_dir, image_file_format = 'jpg', testing = delete_prev_data)
    # run yolo model and get the locations of the well and center of the plate
    individual_well_locations,center_location,n_wells = calibration_model.run_yolo_model(img_filename=image_filename, save_results = True, show_results = False, plate_index = this_plate_parameters['plate_index'])

    # Calculate pixels per mm based on well location data
    well_locations_delta = individual_well_locations[-1] - individual_well_locations[0]
    if n_wells == 96:
        pixels_per_mm = well_locations_delta / [69.7, 42]
    if n_wells == 240:
        pixels_per_mm = well_locations_delta / [85.5, 49.5]

    # Calculate center and its delta in mm
    center = [float(s_camera_settings['widefield'][1]) / 2, float(s_camera_settings['widefield'][2]) / 2]
    center_delta = center - center_location
    center_delta_in_mm = center_delta / pixels_per_mm

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
  
def run_autofocus_at_current_position(controller, starting_location, coolLED_port, 
    this_plate_parameters, autofocus_min_max = [1,-1], autofocus_delta_z = 0.25, cap = None, show_results = False, af_area = 2560):

    lights.coolLed_control.turn_everything_off(coolLED_port) # turn everything off

    # set up the variables  
    # z_pos = -5
    # pixels_per_mm = 192
    # FOV = 5
    # autofocus_min_max = [2.5,-6] # remember that down (towards sample) is negative
    # autofocus_delta_z = 0.25 # mm 
    autofocus_steps = int(abs(np.diff(autofocus_min_max) / autofocus_delta_z)) + 1
    z_limit = [-10,-94]
    offset = 25 # this is for the autofocus algorithm how many pixels apart is the focus to be measures
    thresh = 5 # same as above but now ignores all the values under thresh

    # find the z locations for the loop to step through
    z_positions_start = np.linspace(starting_location['z_pos']+autofocus_min_max[0],starting_location['z_pos']+autofocus_min_max[1],num = autofocus_steps)
    z_positions = []

    # turn on the RGB lights to get a white light for focusing 
    lights.coolLed_control.turn_specified_on(coolLED_port, 
        uv = False, 
        uv_intensity = 1,
        blue = True, 
        blue_intensity = 10,
        green = True, 
        green_intensity = 10,
        red = True, 
        red_intensity = 0)
  
    # go though all the z_positions and get the most in focus position
    images = []
    uncalib_fscore = []
    for counter,z_pos in enumerate(z_positions_start):
        this_location = starting_location.copy()
        this_location['z_pos'] = z_pos
        # jprint(this_location)

        if z_pos < z_limit[0] and z_pos > z_limit[1]:
            z_positions.append(z_pos)

            controller.move_XYZ(position = this_location) # move the said location 
                
            if (counter == 0) and (cap is not None): # capture the frame and return the image and camera 'cap' object
                frame, cap = camera.camera_control.capture_fluor_img_return_img(s_camera_settings, cap = cap, return_cap = True, clear_N_images_from_buffer = 5) 
            elif (counter == 0) and (cap == None):
                frame, cap = camera.camera_control.capture_fluor_img_return_img(s_camera_settings, return_cap = True, clear_N_images_from_buffer = 5) 
            else:
                frame, cap = camera.camera_control.capture_fluor_img_return_img(s_camera_settings, cap = cap, return_cap = True, clear_N_images_from_buffer = 1)
            images.append(frame)
            temp = analysis.fluor_postprocess.crop_center_numpy_return(frame, af_area, center = [1440,1252] )
            temp = sq_grad(temp,thresh = thresh,offset = offset)
            uncalib_fscore.append(np.sum(temp))
            camera.camera_control.imshow_resize(frame_name = "stream", frame = frame)
    
    lights.coolLed_control.turn_everything_off(coolLED_port) # turn everything off
    # images = np.asarray(images)
    # np.save('autofocus_stack.npy',images)

    # a = np.mean(images, axis = 0) # get the average image taken of the stack (for illumination correction)
    # # binary_img = analysis.fluor_postprocess.largest_blob(a > 20) # get the largest binary blob in the image
    # # center = [ np.average(indices) for indices in np.where(binary_img) ] # find where the actual center of the frame is (assuming camera sensor is larger than image circle)
    # # center_int = [int(np.round(point)) for point in center]

    # norm_array = scipy.ndimage.gaussian_filter(a,10) # get the instensities of the images for the illuminance normalizations
    # norm_array_full = 1-(norm_array/np.max(norm_array))
    # # norm_array = analysis.fluor_postprocess.crop_center_numpy_return(norm_array_full,pixels_per_mm*(FOV), center = center_int)

    # focus_score = [] # get the focus score for every image that gets stepped through
    # for this_img in images:
    #     this_img = this_img*(norm_array_full+1)
    #     b = sq_grad(this_img,thresh = thresh,offset = offset)
    #     this_fscore = np.sum(b)
    #     focus_score.append(this_fscore)

    assumed_focus_idx = np.argmax(uncalib_fscore)

    if show_results:
        plt.plot(uncalib_fscore)
        plt.plot(assumed_focus_idx,uncalib_fscore[assumed_focus_idx],'go')
        plt.show(block = True)
        plt.pause(5)
        plt.close('all')

    z_pos = z_positions[assumed_focus_idx] # for the final output
    this_location = starting_location.copy()
    this_location['z_pos'] = z_positions[assumed_focus_idx]
    controller.move_XYZ(position = this_location)

    lights.coolLed_control.turn_specified_on(coolLED_port, 
        uv = int(this_plate_parameters['fluorescence_UV']) > 0, 
        uv_intensity = int(this_plate_parameters['fluorescence_UV']),
        blue = int(this_plate_parameters['fluorescence_BLUE']) > 0, 
        blue_intensity = int(this_plate_parameters['fluorescence_BLUE']),
        green = int(this_plate_parameters['fluorescence_GREEN']) > 0, 
        green_intensity = int(this_plate_parameters['fluorescence_GREEN']),
        red = int(this_plate_parameters['fluorescence_RED']) > 0, 
        red_intensity = int(this_plate_parameters['fluorescence_RED']))

    frame, cap = camera.camera_control.capture_fluor_img_return_img(s_camera_settings, cap = cap,return_cap = True, clear_N_images_from_buffer = 1)
    camera.camera_control.imshow_resize(frame_name = "stream", frame = frame)

    return z_pos, cap

class CNCController:
    def __init__(self, port, baudrate):
        import re
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)

    def wait_for_movement_completion(self,cleaned_line):

        # print("waiting on: " + str(cleaned_line))

        if ('$X' not in cleaned_line) and ('$$' not in cleaned_line) and ('?' not in cleaned_line):
            idle_counter = 0
            time.sleep(0.025)
            while True:
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
                time.sleep(0.025)
                command = str.encode("?"+ "\n")
                self.ser.write(command)
                time.sleep(0.025)
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
        time.sleep(0.025)
        self.ser.write(command.encode())
        time.sleep(0.025)

        CNCController.wait_for_movement_completion(self,command)
        out = []
        for i in range(50):
            time.sleep(0.001)
            response = self.ser.readline().decode().strip()
            time.sleep(0.001)
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
        
        # print('moving to XY')
        # command = 'G0 ' + 'X' + str(position['x_pos']) + ' ' + 'Y' + str(position['y_pos']) 
        command = 'G1 ' + 'X' + str(position['x_pos']) + ' ' + 'Y' + str(position['y_pos']) + ' F2500'
        response, out = CNCController.send_command(self,command)
        ##### move z
        # print('moving to Z')
        # command = 'G0 ' + 'Z' + str(position['z_pos']) 
        command = 'G1 ' + 'Z' + str(position['z_pos']) + ' F2500'
        response, out = CNCController.send_command(self,command)

        return CNCController.get_current_position(self)
    
    def move_XYZ(self, position, return_position = False):

        ##### move xyz
        # print('moving to XYZ')
        # command = 'G0 ' + 'X' + str(position['x_pos']) + ' ' + 'Y' + str(position['y_pos']) + ' ' + 'Z' + str(position['z_pos']) 
        command = 'G1 ' + 'X' + str(position['x_pos']) + ' ' + 'Y' + str(position['y_pos']) + ' ' + 'Z' + str(position['z_pos']) + ' F2500'
        response, out = CNCController.send_command(self,command)

        if return_position:
            return CNCController.get_current_position(self)
        else:
            return response
    
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
    s_wm_positions = settings.get_settings.get_wm_positions()
    s_wm_4pair_positions = settings.get_settings.get_wm_4pair_positions()
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
    lights.coolLed_control.turn_everything_off(coolLED_port)
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
            adjusted_position = run_calib(s_camera_settings,this_plate_parameters,output_dir,calibration_model)
            adjusted_position = run_calib(s_camera_settings,this_plate_parameters,output_dir,calibration_model)
            # capture a single image for calibration
            image_filename = camera.camera_control.simple_capture_data_single_image(s_camera_settings, plate_parameters=this_plate_parameters, output_dir=output_dir, image_file_format = 'jpg')
            image_filename = correct_barrel_distortion(image_filename, a = 0.0, b = 0.0, c = -0.03, d = 1.05)
            individual_well_locations,center_location,n_wells = calibration_model.run_yolo_model(img_filename=image_filename, save_results = True, show_results = False)

        # turn off red
        lights.labjackU3_control.turn_off_red(d)

        ########### calibration attempt
        # get the dx dy of the measured well centers
        well_locations_delta = individual_well_locations[-1]-individual_well_locations[0]
        # get measuring stick

        # determine if the plate is a terasaki or wm and use the correct settings
        if n_wells==96:
            pixels_per_mm = well_locations_delta/[69.695,41.845] #[85.5,49.5]
            s_positions = s_terasaki_positions.copy()
            af_area = 650
        elif n_wells==240:
            pixels_per_mm = well_locations_delta/[84.143,49.227]
            s_positions = s_wm_4pair_positions.copy()
            af_area = 1500
        else:
            pixels_per_mm = well_locations_delta/[69.695,41.845] #default to terasaki
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

        lights.labjackU3_control.turn_off_red(d)

        # try: # this is an attempt at finding the wells using the image 
        #     use_adjusted_centers = True
        #     centers = (sort_rows(individual_well_locations)-center_location)/pixels_per_mm
        # except:
        #     use_adjusted_centers = False
        #     print('Couldnt find all wells reverting to d efault')  
        use_adjusted_centers = False

        # lights.coolLed_control.turn_specified_on(coolLED_port, 
        #     uv = int(this_plate_parameters['fluorescence_UV']) > 0, 
        #     uv_intensity = int(this_plate_parameters['fluorescence_UV']),
        #     blue = int(this_plate_parameters['fluorescence_BLUE']) > 0, 
        #     blue_intensity = int(this_plate_parameters['fluorescence_BLUE']),
        #     green = int(this_plate_parameters['fluorescence_GREEN']) > 0, 
        #     green_intensity = int(this_plate_parameters['fluorescence_GREEN']),
        #     red = int(this_plate_parameters['fluorescence_RED']) > 0, 
        #     red_intensity = int(this_plate_parameters['fluorescence_RED']))

        found_autofocus_positions = []

        # fluorescently image each of the wells
        for well_index,this_well_location_xy in enumerate(zip(s_positions['x_relative_pos_mm'].values(),s_positions['y_relative_pos_mm'].values())):
            # get plate parameters
            this_plate_parameters['well_name'] = s_positions['name'][well_index]
            this_well_coords = dict()

            if use_adjusted_centers:
                # calculate the specific well location
                this_well_coords['x_pos'] = adjusted_position['x_pos'] + calibration_coordinates['x_pos'] 
                this_well_coords['y_pos'] = adjusted_position['y_pos'] + calibration_coordinates['y_pos'] + s_positions['y_offset_to_fluor_mm'][0]

                this_well_coords['x_pos'] += centers[well_index,0] #this_well_location_xy[0]
                this_well_coords['x_pos'] += -0.15
                this_well_coords['y_pos'] += centers[well_index,1] #this_well_location_xy[1]
                this_well_coords['y_pos'] += -2.5
            else:  
                this_well_coords['x_pos'] = adjusted_position['x_pos']
                this_well_coords['y_pos'] = adjusted_position['y_pos'] + s_positions['y_offset_to_fluor_mm'][0]
                this_well_coords['x_pos'] += this_well_location_xy[0] + -0.15
                this_well_coords['y_pos'] += this_well_location_xy[1] + -2.5

            if well_index == 0:
                this_well_coords['z_pos'] = calibration_coordinates['z_pos']
            elif well_index == 1:
                this_well_coords['z_pos'] = z_pos_found_autofocus_inital
            else:
                if len(found_autofocus_positions) > 5:
                    this_well_coords['z_pos'] = np.mean(found_autofocus_positions[-5:])
                else:
                    this_well_coords['z_pos'] = np.mean(found_autofocus_positions)
            print(well_index, this_well_coords)
            # move the fluorescent imaging head to that specific well  

            if use_adjusted_centers: # if the fir st one get a bse measurement for all the rest
                controller.move_XYZ(position = this_well_coords)
                lights.labjackU3_control.turn_on_red(d)
                terasaki_adjusted_position, center_delta_in_mm = run_calib_terasaki(s_camera_settings,this_plate_parameters,output_dir,s_terasaki_positions,calibration_model)
                lights.labjackU3_control.turn_off_everything(d)
            else: # otherswise measure then report finding and then adjust from the inital base
                controller.move_XYZ(position = this_well_coords)
                # lights.labjackU3_control.turn_on_red(d)
                # terasaki_adjusted_position, center_delta_in_mm = run_calib_terasaki(s_camera_settings,this_plate_parameters,output_dir,s_terasaki_positions,calibration_model)
            
            if well_index == 0:
                lights.labjackU3_control.turn_off_everything(d)
                # get first autofocus and return the cap
                z_pos_found_autofocus_inital, cap = run_autofocus_at_current_position(controller, 
                    this_well_coords, coolLED_port, this_plate_parameters, autofocus_min_max = [3,-3], 
                    autofocus_delta_z = 0.1, cap = None, af_area=af_area)
                this_well_coords['z_pos'] = z_pos_found_autofocus_inital
                found_autofocus_positions.append(z_pos_found_autofocus_inital)
            else:  
                z_pos_found_autofocus, cap = run_autofocus_at_current_position(controller, 
                    this_well_coords, coolLED_port, this_plate_parameters, autofocus_min_max = [0.75,-0.75], 
                    autofocus_delta_z = 0.25, cap = cap, af_area=af_area)
                this_well_coords['z_pos'] = z_pos_found_autofocus
                found_autofocus_positions.append(z_pos_found_autofocus)

            if well_index == (len(s_positions['x_relative_pos_mm'].values()) - 1): 
                cap = camera.camera_control.capture_data_fluor_multi_exposure(s_camera_settings, plate_parameters=this_plate_parameters, testing=False, output_dir=output_dir, cap = cap, return_cap = False)
            else:  
                cap = camera.camera_control.capture_data_fluor_multi_exposure(s_camera_settings, plate_parameters=this_plate_parameters, testing=False, output_dir=output_dir, cap = cap, return_cap = True)
            lights.coolLed_control.turn_everything_off(coolLED_port)
        lights.coolLed_control.turn_everything_off(coolLED_port)

    # shut everything down 
    controller.set_up_grbl(home = True)
    # movement.simple_stream.home_GRBL(s_machines['grbl'][0], testing = False,camera=None) # home the machine

    # print('eof')