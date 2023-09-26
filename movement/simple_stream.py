"""
This is a simple script that attempts to connect to the GRBL controller at 
> /dev/tty.usbserial-A906L14X
It then reads the grbl_test.gcode and sends it to the controller

The script waits for the completion of the sent line of gcode before moving onto the next line

tested on
> WINDOWS 10
> Python 3.8 
> Vscode 1.62.3
> Openbuilds BlackBox x32 GRBL controller
> GRBL 1.1g (GRBLhal openbuilds)

"""
import cv2
import serial
import time
from threading import Event
try:
    from camera.camera_control import capture_images_for_time
except:
    print('cant import camera commands')

BAUD_RATE = 115200

def update_camera_for_N_seconds():
    return 0

def remove_comment(string):
    if (string.find(';') == -1):
        return string
    else:
        return string[:string.index(';')]

def remove_eol_chars(string):
    # removed \n or traling spaces
    return string.strip()

def send_wake_up(ser,sleep_amount = 2):
    # Wake up
    # Hit enter a few times to wake the Printrbot
    ser.write(str.encode("\r\n\r\n"))
    time.sleep(sleep_amount)   # Wait for Printrbot to initialize
    ser.reset_input_buffer()  # Flush startup text in serial input # ser.flushInput()

def send_wake_up_update_cam_stream(ser,sleep_amount = 2):

    cap = cv2.VideoCapture(int(0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,int(5472))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,int(3640))
    cap.set(cv2.CAP_PROP_FPS,int(6))
    # Wake up
    # Hit enter a few times to wake the Printrbot
    ser.timeout =2
    ser.write(str.encode("\n"))
    # time.sleep(sleep_amount)   # Wait for Printrbot to initialize
    start_time = time.time()
    capture_images_for_time(cap,sleep_amount, show_images=True,move_to = [1920,520], start_time = start_time)
    ser.reset_input_buffer()  # Flush startup text in serial input # ser.flushInput()

def wait_for_movement_completion(ser,cleaned_line):

    Event().wait(1)

    if (cleaned_line != '$X') and (cleaned_line != '$$'):

        idle_counter = 0
        while True:
            time.sleep(0.2)
            ser.reset_input_buffer()
            command = str.encode('? ' + '\n')
            ser.write(command)
            grbl_out = ser.readline() 
            grbl_response = grbl_out.strip().decode('utf-8')

            if grbl_response != 'ok':
                if grbl_response.find('Idle') > 0:
                    idle_counter += 1
            if idle_counter > 10:
                break
            if 'alarm' in grbl_response.lower():
                raise ValueError(grbl_response)
    return

def wait_for_movement_completion_update_cam_stream(ser,cleaned_line,cam_settings = False):

    cap = cv2.VideoCapture(int(0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,int(5472))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,int(3640))
    cap.set(cv2.CAP_PROP_FPS,int(6))

    # Event().wait(1)
    start_time = time.time()
    capture_images_for_time(cap,N=1, show_images=True,move_to = [1920,520], start_time = start_time)

    if (cleaned_line != '$X') and (cleaned_line != '$$'):

        idle_counter = 0
        while True:
            start_time = time.time()
            capture_images_for_time(cap,0.2, show_images=True,move_to = [1920,520], start_time = start_time)
            ser.reset_input_buffer()
            command = str.encode('? ' + '\n')
            ser.write(command)
            grbl_out = ser.readline() 
            grbl_response = grbl_out.strip().decode('utf-8')

            if grbl_response != 'ok':
                if grbl_response.find('Idle') > 0:
                    idle_counter += 1
            if idle_counter > 10:
                break
            if 'alarm' in grbl_response.lower():
                raise ValueError(grbl_response)

    return

def stream_gcode(GRBL_port_path,gcode_path):
    outputs = []
    # with contect opens file/connection and closes it if function(with) scope is left
    with open(gcode_path, "r") as file, serial.Serial(GRBL_port_path, BAUD_RATE) as ser:
        send_wake_up(ser)
        ser.timeout = 2

        for line in file:
            # cleaning up gcode from file
            cleaned_line = remove_eol_chars(remove_comment(line))
            if cleaned_line:  # checks if string is empty
                print("Sending gcode:" + str(cleaned_line))
                # converts string to byte encoded string and append newline
                command = str.encode(line + '\n')
                ser.write(command)  # Send g-code

                wait_for_movement_completion(ser,cleaned_line)

                grbl_out = ser.readline()  # Wait for response with carriage return
                print(" : " , grbl_out.strip().decode('utf-8'))
                outputs.append(grbl_out.strip().decode('utf-8'))
        
        print('End of gcode')

    return outputs

def send_single_line(GRBL_port_path,gcode_line):
    outputs = []

    with serial.Serial(GRBL_port_path, BAUD_RATE) as ser:
        send_wake_up_update_cam_stream(ser)
        # send_wake_up(ser)
        ser.timeout = 2
        
        line = gcode_line

        # cleaning up gcode from file
        cleaned_line = remove_eol_chars(remove_comment(line))
        if cleaned_line:  # checks if string is empty
            print("Sending gcode:" + str(cleaned_line))
            # converts string to byte encoded string and append newline
            command = str.encode(line + '\n')
            ser.write(command)  # Send g-code

            # if the sent line is a checker then skip right to 
            if cleaned_line != '?' and cleaned_line != '$X' and cleaned_line != '$$':
                wait_for_movement_completion_update_cam_stream(ser,cleaned_line) 
                grbl_out = ser.readline()  # Wait for response with carriage return
            else:
                grbl_out = ser.read_until(expected='ok')

            print(" : " , grbl_out.strip().decode('utf-8'))
            outputs = grbl_out.strip().decode('utf-8')

    return outputs

def get_settings(GRBL_port_path): # gets the settings from the grbl controller 

    print('Getting GRBL controller settings')
    settings = []
 
    with serial.Serial(GRBL_port_path, BAUD_RATE) as ser:
        send_wake_up(ser)
        
        line = '$$'

        # cleaning up gcode from file
        cleaned_line = remove_eol_chars(remove_comment(line))
        if cleaned_line:  # checks if string is empty
            print("Sending gcode:" + str(cleaned_line))
            # converts string to byte encoded string and append newline
            command = str.encode(line + '\n')
            ser.write(command)  # Send g-code
 
            for i in range(1000):
                grbl_out = ser.readline()  # Wait for response with carriage return
                out_decoded = grbl_out.strip().decode('utf-8')
                # print(out_decoded)

                settings.append(out_decoded)

                if 'ok' in out_decoded:
                    break

    return settings

def home_GRBL(GRBL_port_path, testing = False):

    if not testing:
        send_single_line(GRBL_port_path,'$X')
        send_single_line(GRBL_port_path,'$H')
    if testing:
        send_single_line(GRBL_port_path,'$X')
        print('testing --- $H')

def get_current_position(GRBL_port_path, testing = False):

    if not testing:
        send_single_line(GRBL_port_path,'$')
        send_single_line(GRBL_port_path,'$H')
    if testing:
        send_single_line(GRBL_port_path,'$X')
        print('testing --- $H')

    return 0

def move_XYZ(position,GRBL_port_path, testing = False, round_decimals = False):

    if round_decimals:
        position['x_pos'] = round(position['x_pos'],4)
        position['y_pos'] = round(position['y_pos'],4)
        position['z_pos'] = round(position['z_pos'],4)

    if testing == False:
        print('moving to XYZ')
        move_xyz_command = 'G0 ' + 'X' + str(position['x_pos']) + ' ' + 'Y' + str(position['y_pos']) + ' ' + 'Z' + str(position['z_pos'])
        send_single_line(GRBL_port_path,move_xyz_command)
    else:
        print('moving to XYZ')
        move_xyz_command = 'G0 ' + 'X' + str(position['x_pos']) + ' ' + 'Y' + str(position['y_pos']) + ' ' + 'Z' + str(position['z_pos'])
        print(GRBL_port_path,move_xyz_command)

def move_XY_at_Z_travel(position,GRBL_port_path,z_travel_height = 0.5, testing = False, go_back_down = True , round_decimals = False):

    if round_decimals:
        position['x_pos'] = round(position['x_pos'],4)
        position['y_pos'] = round(position['y_pos'],4)
        position['z_pos'] = round(position['z_pos'],4)

    if testing == False:
        print('moving Z to travel height')
        move_z_command = 'G0 ' + 'Z' + str(z_travel_height) 
        send_single_line(GRBL_port_path,move_z_command)

        print('moving to XY')
        move_xy_command = 'G0 ' + 'X' + str(position['x_pos']) + ' ' + 'Y' + str(position['y_pos']) 
        send_single_line(GRBL_port_path,move_xy_command)

        if go_back_down:

            print('move Z to imaging height')
            move_xy_command = 'G0 ' + 'Z' + str(position['z_pos']) 
            send_single_line(GRBL_port_path,move_xy_command)
    else:
        print('moving Z to travel height')
        move_z_command = 'G0 ' + 'Z' + str(z_travel_height) 
        print(GRBL_port_path,move_z_command)

        print('moving to XY')
        move_xy_command = 'G0 ' + 'X' + str(position['x_pos']) + ' ' + 'Y' + str(position['y_pos']) 
        print(GRBL_port_path,move_xy_command)

        if go_back_down:

            print('move Z to imaging height')
            move_xy_command = 'G0 ' + 'Z' + str(position['z_pos']) 
            print(GRBL_port_path,move_xy_command)

if __name__ == "__main__":
    import os

    # GRBL_port_path = '/dev/tty.usbserial-A906L14X'
    GRBL_port_path = 'COM5'
    gcode_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'grbl_test.gcode')
    assert os.path.isfile(gcode_path)

    print("USB Port: ", GRBL_port_path)
    print("Gcode file: ", gcode_path)
    stream_gcode(GRBL_port_path,gcode_path)

    print('EOF')
