import os,time,glob,sys,time,tqdm,cv2, serial
# import lights.labjackU3_control
# import lights.coolLed_control
import settings.get_settings
import movement.simple_stream
# import camera.camera_control

# def turn_everything_off_at_exit():
#     lights.labjackU3_control.turn_off_everything()
#     cv2.destroyAllWindows()
#     # lights.coolLed_control.turn_everything_off()

class CNCController:
    def __init__(self, port, baudrate):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)

    def wait_for_movement_completion(self,cleaned_line):

        print("waiting on: " + str(cleaned_line))

        if ('$X' not in cleaned_line) and ('$$' not in cleaned_line) and ('?' not in cleaned_line):
            idle_counter = 0
            while True:
                time.sleep(0.1)
                # self.ser.flush()
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
                time.sleep(0.1)
                command = str.encode("?"+ "\n")
                self.ser.write(command)
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
        # self.ser.flush()
        self.ser.reset_input_buffer()
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

        position['x_pos'] = MPos[0]
        position['y_pos'] = MPos[1]
        position['z_pos'] = MPos[2]

        return position
    
    def home_grbl(self):
        print("HOMING CNC")
        command = "$H"+ "\n"
        response, out = CNCController.send_command(command)
    
    def set_up_grbl(self, home = True):
        # unlock 
        command = "$X"+ "\n"
        response, out = CNCController.send_command(command)

        command = "?"+ "\n"
        response, out = CNCController.send_command(command)

        if home:
            CNCController.home_grbl(self)

    def close_connection(self):
        self.ser.close()

import atexit

if __name__ == "__main__":

    # atexit.register(turn_everything_off_at_exit)
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

    settings.get_settings.check_grbl_port(s_machines['grbl'][0], run_as_testing = False)
    controller = CNCController(port=s_machines['grbl'][0], baudrate=s_machines['grbl'][1])
    response, s_grbl_settings = controller.send_command("$$"+ "\n")
    s_grbl_settings_df,s_grbl_settings = settings.get_settings.convert_GRBL_settings(s_grbl_settings)

    # run setup test to make sure everything works or throw error
    s_todays_runs = settings.get_settings.update_todays_runs(s_todays_runs, overwrite=True)
    controller.set_up_grbl(home = True)

    plate_index = []
    plate_positions = []
    for this_plate_index in s_plate_names_and_opts['plate_name']:
        this_plate_name = s_plate_names_and_opts['plate_name'][this_plate_index]
        if this_plate_name != 'NONE':
            plate_index.append(this_plate_index)

    # # # run experiment
    for this_plate_index in plate_index:
        this_plate_parameters,this_plate_position = settings.get_settings.get_indexed_dict_parameters(s_plate_names_and_opts,s_plate_positions,this_plate_index)
    
        print(this_plate_parameters)
        print(this_plate_position)

        position = this_plate_position.copy()
        position['x_pos'] = round(position['x_pos'],4)
        position['y_pos'] = round(position['y_pos'],4)
        position['z_pos'] = round(position['z_pos'],4)

        current_position = controller.get_current_position()

        if round(float(current_position['z_pos']),1) != float(z_travel_height):
            #### go to z travel height
            command = "g0 z" + str(z_travel_height) + " " + "\n"
            response, out = controller.send_command(command)

        ##### move xy
        print('moving to XY')
        command = 'G0 ' + 'X' + str(position['x_pos']) + ' ' + 'Y' + str(position['y_pos']) 
        response, out = controller.send_command(command)

        ##### move z
        print('moving to Z')
        command = 'G0 ' + 'Z' + str(position['z_pos']) 
        response, out = controller.send_command(command)

        time.sleep(0.5)
        print('IMAGING TEST RIGHT NOW')

        #### go to z travel height
        z_travel_height = s_machines['grbl'][2]
        command = "g0 z" + str(z_travel_height) + " " + "\n"
        response, out = controller.send_command(command)

        # movement.simple_stream.move_XY_at_Z_travel(this_plate_position,s_machines['grbl'][0],z_travel_height = s_machines['grbl'][2], testing=False, round_decimals = 4, camera = None)

        time.sleep(0.1)
        print('')

    command = "g0 x-5 y-5 z-5\n"
    response, out = controller.send_command(command)

    command = "?" + "\n"
    response, out = controller.send_command(command)
    print(out)

    
    # movement.simple_stream.home_GRBL(s_machines['grbl'][0], testing = False,camera=None) # home the machine

    # for this_plate_index in plate_index:
    #     this_plate_parameters,this_plate_position = settings.get_settings.get_indexed_dict_parameters(s_plate_names_and_opts,s_plate_positions,this_plate_index)
    
    #     print(this_plate_parameters)
    #     print(this_plate_position)

    #     # adjust for the imaging head 7 positions 
    #     this_plate_position['y_pos'] = this_plate_position['y_pos'] + s_terasaki_positions['y_offset_to_fluor_mm'][0]
    #     movement.simple_stream.move_XY_at_Z_travel(this_plate_position,s_machines['grbl'][0],z_travel_height = s_machines['grbl'][2], testing=False, go_back_down = False, round_decimals = 4, camera = None)
    #     # calculate the calibration corner coordinates
    #     calibration_coordinates = dict()
    #     calibration_coordinates['x_pos'] = this_plate_position['x_pos'] + s_terasaki_positions['calib_x_pos_mm'][0]
    #     calibration_coordinates['y_pos'] = this_plate_position['y_pos'] + s_terasaki_positions['calib_y_pos_mm'][0]
    #     calibration_coordinates['z_pos'] = s_terasaki_positions['calib_z_pos_mm'][0]
    #     # move to the calibration side
    #     movement.simple_stream.move_XYZ(calibration_coordinates,s_machines['grbl'][0], testing=False, round_decimals = 4, camera = None)

    #     # run the calibration script 
    #     print('running Z calibration script -------------------------------------------------------------------------------------')
    #     calibration_coordinates['z_pos'] = calibration_coordinates['z_pos'] - 10

    #     for well_index,this_terasaki_well_xy in enumerate(zip(s_terasaki_positions['x_relative_pos_mm'].values(),s_terasaki_positions['y_relative_pos_mm'].values())):
    #         this_plate_parameters['well_name'] = s_terasaki_positions['name'][well_index]
    #         terasaki_well_coords = dict()
    #         terasaki_well_coords['x_pos'] = this_plate_position['x_pos'] + this_terasaki_well_xy[0]
    #         terasaki_well_coords['y_pos'] = this_plate_position['y_pos'] + this_terasaki_well_xy[1]
    #         terasaki_well_coords['z_pos'] = calibration_coordinates['z_pos']
    #         print(well_index, terasaki_well_coords)
    #         movement.simple_stream.move_XYZ(terasaki_well_coords,s_machines['grbl'][0], testing=False, round_decimals = 4, camera = None)
    #         print('imaging')
    #         time.sleep(0.1)

    # # shut everything down 
    # movement.simple_stream.home_GRBL(s_machines['grbl'][0], testing = False,camera=None) # home the machine

    # print('eof')