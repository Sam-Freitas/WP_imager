import tkinter as tk
import pandas as pd
from numpy import asarray, arange, nonzero, zeros,uint8
from PIL import Image, ImageTk
import os
import settings.get_settings
from tkinter import messagebox, PhotoImage
import atexit, json

import os,time,glob,sys,time,tqdm,cv2, serial, json
import numpy as np
import lights.labjackU3_control
import lights.coolLed_control
import settings.get_settings
# import movement.simple_stream
import camera.camera_control

### plate name, experiment name, lifepsan, healthspan, fluor timing, uv, blue, green, red
DEFAULT_VALUES = ['NONE','NONE',1,0,100,0,0,0,0]

path_to_settings_folder = path_to_settings_folder = os.path.join(os.getcwd(), "settings")

BUTTON_W = 10  # Set the size of the buttons
BUTTON_H = 5
POPUP_WIDTH = 300  # Set the width of the popup window
POPUP_HEIGHT = 350  # Set the height of the popup window

s_plate_names_and_opts = settings.get_settings.get_plate_names_and_opts()
s_plate_positions = settings.get_settings.get_plate_positions()
s_machines = settings.get_settings.get_machine_settings()
s_camera_settings = settings.get_settings.get_basic_camera_settings()

output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'output')

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

def get_index_from_row_col(row,col): # get the row and column as the plate index for ease of use

    rows = asarray(list(s_plate_names_and_opts['row'].values())) # get the rows dict as an array
    columns = asarray(list(s_plate_names_and_opts['column'].values())) # get the rows dict as an array
    rows_bool = (rows==row) # mask out the rows and columsn
    columns_bool = (columns==col)

    index_bool = rows_bool*columns_bool # find the overlap
    index = int(nonzero((1*index_bool) * (arange(1,len(index_bool)+1)))[0]) # convert to int

    return index

def get_settings_from_index(index):

    temp = dict()
    for key in s_plate_names_and_opts.keys():
        temp[key] = s_plate_names_and_opts[key][index]

    return temp

def get_position_from_index(index):

    temp = dict()
    for key in s_plate_positions.keys():
        temp[key] = s_plate_positions[key][index]

    return temp

def button_click(row, col):

    index = get_index_from_row_col(row,col)

    position = get_position_from_index(index)
    this_plate_parameters = get_settings_from_index(index)

    controller.move_XY_at_Z_travel(position=position, z_travel_height=z_travel_height)
    image_filename = camera.camera_control.simple_capture_data_single_image(s_camera_settings, plate_parameters=this_plate_parameters, output_dir=output_dir, image_file_format = 'jpg')

if __name__ == "__main__":
    # atexit.register(exit_function)
    # Create the main window

    settings.get_settings.check_grbl_port(s_machines['grbl'][0], run_as_testing = False)
    controller = CNCController(port=s_machines['grbl'][0], baudrate=s_machines['grbl'][1])
    response, s_grbl_settings = controller.send_command("$$"+ "\n")
    s_grbl_settings_df,s_grbl_settings = settings.get_settings.convert_GRBL_settings(s_grbl_settings)
    z_travel_height = s_machines['grbl'][2]

    controller.set_up_grbl(home = False)

    root = tk.Tk()
    root.title("                                                                          WP Imager ------ MOVE TO PLATE")

    # Get the screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate the x and y coordinates to center the grid
    x = (screen_width - (8 * BUTTON_W*10)) // 2
    y = (screen_height - (9 * BUTTON_H*20)) // 2

    # Set the dimensions of the main window
    root.geometry(f"{8 * round(BUTTON_W*8.4)}x{9 * BUTTON_H*18}+{x}+{y}")

    # Create a 9x8 grid of buttons
    buttons = [[None for _ in range(8)] for _ in range(9)]

    index = 0
    for i in range(9):
        for j in range(8):

            text = s_plate_names_and_opts['plate_name'][index] + '\n' + s_plate_names_and_opts['experiment_name'][index] + '\n' + f'{i}-{j}'

            fluor_option = bool(s_plate_names_and_opts['fluorescence'][index])
            bg = zeros(shape= (x,y,3)).astype(uint8)

            if 'NONE' in text:
                bg = 'White'
            else:
                bg = 'Green'

            if 'NONE' not in text and fluor_option:
                bg = 'Purple'

            buttons[i][j] = tk.Button(root, text=text, 
                                    command=lambda i=i, j=j: button_click(i, j), 
                                    width=BUTTON_W, height=BUTTON_H,
                                    bg = bg)
                                    # image= bg)
            buttons[i][j].grid(row=i, column=j, padx=2, pady=2)
            index += 1

    # Start the tkinter event loop
    root.mainloop()
