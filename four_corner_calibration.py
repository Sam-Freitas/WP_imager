import os,time,glob,sys,time,tqdm,cv2, serial, json
import numpy as np
import lights.labjackU3_control
# import lights.coolLed_control
import settings.get_settings
import movement.simple_stream
import camera.camera_control

from Run_yolo_model import run_yolo_model

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

        print("waiting on: " + str(cleaned_line))

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
                            print(grbl_response)
                if idle_counter == 1 or idle_counter == 2:
                    print(grbl_response)
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
            command = "g0 z" + str(z_travel_height) + " " + "\n"
            response, out = CNCController.send_command(self,command)
        
        print('moving to XY')
        command = 'G0 ' + 'X' + str(position['x_pos']) + ' ' + 'Y' + str(position['y_pos']) 
        response, out = CNCController.send_command(self,command)
        ##### move z
        print('moving to Z')
        command = 'G0 ' + 'Z' + str(position['z_pos']) 
        response, out = CNCController.send_command(self,command)

        return CNCController.get_current_position(self)
    
    def move_XYZ(self, position):

        ##### move xyz
        print('moving to XYZ')
        command = 'G0 ' + 'X' + str(position['x_pos']) + ' ' + 'Y' + str(position['y_pos']) + ' ' + 'Z' + str(position['z_pos']) 
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
    
def run_calib(s_camera_settings,this_plate_parameters,output_dir,s_terasaki_positions, final_move = True):
    # take image
    image_filename = camera.camera_control.simple_capture_data_single_image(s_camera_settings, plate_parameters=this_plate_parameters,output_dir=output_dir, image_file_format = 'jpg')
    # run yolo model and get the locations of the well and center of the plate
    individual_well_locations,center_location = run_yolo_model(img_filename=image_filename, plot_results = True)

    # find the differnce between what was given and what is measured 
    well_locations_delta = individual_well_locations[-1]-individual_well_locations[0]
    pixels_per_mm = well_locations_delta/[69.7,42] 
    center = [float(s_camera_settings['widefield'][1])/2,float(s_camera_settings['widefield'][2])/2]
    center_delta = center-center_location
    center_delta_in_mm = center_delta/pixels_per_mm
    calibration_coordinates = dict()
    calibration_coordinates['x_pos'] = center_delta_in_mm[0]
    calibration_coordinates['y_pos'] = center_delta_in_mm[1]
    calibration_coordinates['z_pos'] = s_terasaki_positions['calib_z_pos_mm'][0]

    # adjust the position and return where the system should go to correct
    measured_position = controller.get_current_position()
    adjusted_position = measured_position.copy()
    adjusted_position['x_pos'] = measured_position['x_pos'] + center_delta_in_mm[0]
    adjusted_position['y_pos'] = measured_position['y_pos'] - center_delta_in_mm[1]

    if final_move:
        controller.move_XYZ(position = adjusted_position)
    
    return adjusted_position

import atexit

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

    testing = False
    # get all the experiments that are not defunct
    plate_index = [0,7,64,71]
    grid = [9,8] # rows, cols

    if not testing:

        settings.get_settings.check_grbl_port(s_machines['grbl'][0], run_as_testing = False)
        controller = CNCController(port=s_machines['grbl'][0], baudrate=s_machines['grbl'][1])
        response, s_grbl_settings = controller.send_command("$$"+ "\n")
        s_grbl_settings_df,s_grbl_settings = settings.get_settings.convert_GRBL_settings(s_grbl_settings)
        z_travel_height = s_machines['grbl'][2]

        # run setup test to make sure everything works or throw error
        s_todays_runs = settings.get_settings.update_todays_runs(s_todays_runs, overwrite=True)
        d = lights.labjackU3_control.setup_labjack(verbose=True)    # test the blue and red lights
        lights.labjackU3_control.blink_led(d)

        lights.labjackU3_control.turn_on_red(d)
        controller.set_up_grbl(home = True)
        # # # run lifespan imaging experiments

        corner_points = []

        for this_plate_index in plate_index:
            # get the experiment options
            this_plate_parameters,this_plate_position = settings.get_settings.get_indexed_dict_parameters(s_plate_names_and_opts,s_plate_positions,this_plate_index)
            print(this_plate_parameters)
            print(this_plate_position)

            # get the position of the experiment
            # move above the plate
            position = this_plate_position.copy()
            position['x_pos'],position['y_pos'],position['z_pos'] = round(position['x_pos'],4), round(position['y_pos'],4), -90
            controller.move_XY_at_Z_travel(position = position,
                                        z_travel_height = z_travel_height)
            
            # run the adjustment 3 times 
            adjusted_position = run_calib(s_camera_settings,this_plate_parameters,output_dir,s_terasaki_positions)
            adjusted_position = run_calib(s_camera_settings,this_plate_parameters,output_dir,s_terasaki_positions)
            adjusted_position = run_calib(s_camera_settings,this_plate_parameters,output_dir,s_terasaki_positions)
            adjusted_position = run_calib(s_camera_settings,this_plate_parameters,output_dir,s_terasaki_positions, final_move = False)

            print(adjusted_position)
            corner_points.append(adjusted_position)

            time.sleep(0.05)
            print('')

        for i,out in enumerate(corner_points):
            plate_index[i]
            print(i,out)
    else:

        corner_points = dict()
        corner_points = np.asarray([[-42.668,-111.716,-100],
                            [-1111.444,-121.817,-100],
                            [-42.265,-1029.433,-100],
                            [-1110.465,-1038.979,-100]])

    out = []

    starting_x = np.linspace(corner_points[0]['x_pos'],corner_points[2]['x_pos'],9)
    ending_x = np.linspace(corner_points[1]['x_pos'],corner_points[3]['x_pos'],9)

    starting_y = np.linspace(corner_points[0]['y_pos'],corner_points[2]['y_pos'],9)
    ending_y = np.linspace(corner_points[1]['y_pos'],corner_points[3]['y_pos'],9)

    counter = 0
    plt.figure(1)
    for i in range(grid[0]): #rows

        this_staring_x = starting_x[i]
        this_staring_y = starting_y[i]

        this_ending_x = ending_x[i]
        this_ending_y = ending_y[i]

        print(this_staring_x,this_staring_y, ' ----- to ----- ',this_ending_x,this_ending_y)

        this_out_x = np.linspace(this_staring_x, this_ending_x, grid[1])
        this_out_y = np.linspace(this_staring_y, this_ending_y, grid[1])

        print(this_out_x)
        print(this_out_y)

        for j in range(this_out_x.shape[0]):
            temp = [this_out_x[j],this_out_y[j]]
            out.append(temp)
            plt.plot(temp[0],temp[1],'ro')
            plt.text(temp[0]+0.5,temp[1]+0.5, str(temp[0]) + '\n' + str(temp[1]))
            plt.text(temp[0]-10,temp[1]-10, str(counter))
            counter += 1
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed') #works fine on Windows!
    plt.show()

    controller.set_up_grbl(home = True)
    for this_plate_index in plate_index:
    # get the experiment options
        this_plate_parameters,this_plate_position = settings.get_settings.get_indexed_dict_parameters(s_plate_names_and_opts,s_plate_positions,this_plate_index)
        print(this_plate_parameters)
        print(this_plate_position)

        # get the position of the experiment
        # move above the plate
        position = this_plate_position.copy()
        position['x_pos'],position['y_pos'],position['z_pos'] = np.round(out[this_plate_index][0],4), np.round(out[this_plate_index][1],4), -100
        controller.move_XY_at_Z_travel(position = position,
                                z_travel_height = z_travel_height)
        adjusted_position = run_calib(s_camera_settings,this_plate_parameters,output_dir,s_terasaki_positions, final_move = False)


    print('eof')
