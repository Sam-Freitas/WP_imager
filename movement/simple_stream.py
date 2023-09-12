"""
This is a simple script that attempts to connect to the GRBL controller at 
> /dev/tty.usbserial-A906L14X
It then reads the grbl_test.gcode and sends it to the controller

The script waits for the completion of the sent line of gcode before moving onto the next line

tested on
> MacOs Monterey arm64
> Python 3.9.5 | packaged by conda-forge | (default, Jun 19 2021, 00:24:55) 
[Clang 11.1.0 ] on darwin
> Vscode 1.62.3
> Openbuilds BlackBox GRBL controller
> GRBL 1.1

"""

import serial
import time
from threading import Event

BAUD_RATE = 115200

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
    ser.flushInput()  # Flush startup text in serial input

def wait_for_movement_completion(ser,cleaned_line):

    Event().wait(1)

    if cleaned_line != '$X' or '$$':

        idle_counter = 0
        while True:
            # Event().wait(0.01)
            ser.reset_input_buffer()
            command = str.encode('?' + '\n')
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
    # with contect opens file/connection and closes it if function(with) scope is left
    with open(gcode_path, "r") as file, serial.Serial(GRBL_port_path, BAUD_RATE) as ser:
        send_wake_up(ser)
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
        
        print('End of gcode')

def send_single_line(GRBL_port_path,gcode_line):

    with serial.Serial(GRBL_port_path, BAUD_RATE) as ser:
        send_wake_up(ser)
        
        line = gcode_line

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

    # print('End of line')

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


def move_XYZ(position,GRBL_port_path, testing = False):

    if testing == False:
        print('moving to XYZ')
        move_xyz_command = 'G1 ' + 'X' + str(position['x_pos']) + ' ' + 'Y' + str(position['y_pos']) + ' ' + 'Z' + str(position['z_pos'])+ ' F3000'
        send_single_line(GRBL_port_path,move_xyz_command)
    else:
        print('moving to XYZ')
        move_xyz_command = 'G1 ' + 'X' + str(position['x_pos']) + ' ' + 'Y' + str(position['y_pos']) + ' ' + 'Z' + str(position['z_pos'])+ ' F3000'
        print(GRBL_port_path,move_xyz_command)

def move_XY_at_z_travel(plate_position,GRBL_port_path,z_travel_height = 0.5, testing = False, go_back_down = True):

    if testing == False:
        print('moving Z to travel height')
        move_z_command = 'G1 ' + 'Z' + str(z_travel_height) + ' F3000'
        send_single_line(GRBL_port_path,move_z_command)

        print('moving to XY')
        move_xy_command = 'G1 ' + 'X' + str(plate_position['x_pos']) + ' ' + 'Y' + str(plate_position['y_pos']) + ' F3000'
        send_single_line(GRBL_port_path,move_xy_command)

        if go_back_down:

            print('move Z to imaging height')
            move_xy_command = 'G1 ' + 'Z' + str(plate_position['z_pos']) + ' F3000'
            send_single_line(GRBL_port_path,move_xy_command)
    else:
        print('moving Z to travel height')
        move_z_command = 'G1 ' + 'Z' + str(z_travel_height) + ' F3000'
        print(GRBL_port_path,move_z_command)

        print('moving to XY')
        move_xy_command = 'G1 ' + 'X' + str(plate_position['x_pos']) + ' ' + 'Y' + str(plate_position['y_pos']) + ' F3000'
        print(GRBL_port_path,move_xy_command)

        if go_back_down:

            print('move Z to imaging height')
            move_xy_command = 'G1 ' + 'Z' + str(plate_position['z_pos']) + ' F3000'
            print(GRBL_port_path,move_xy_command)

if __name__ == "__main__":

    # GRBL_port_path = '/dev/tty.usbserial-A906L14X'
    GRBL_port_path = 'COM3'
    gcode_path = 'grbl_test.gcode'

    print("USB Port: ", GRBL_port_path)
    print("Gcode file: ", gcode_path)
    stream_gcode(GRBL_port_path,gcode_path)

    print('EOF')
