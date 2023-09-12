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

BAUD_RATE = 9600 # 1200, 2400, 4800, 19200, 38400, 57600, and 115200

def test_coolLed_connection(port, testing = False):

    try:
        with serial.Serial(port, BAUD_RATE) as ser:
            print('Reading coolLed settings:')
            ser.reset_input_buffer()
            ser.write(str.encode("CSS?\r\n"))
            time.sleep(0.5)   # Wait 
            out = ser.readline() 
            response = out.strip().decode('utf-8')
            print(response)
            return_val = 1
    except Exception as err:
        return_val = 0
        print(err)

    return(return_val)

def stream_TX(port):
    # with contect opens file/connection and closes it if function(with) scope is left
    with serial.Serial(port, BAUD_RATE) as ser:
        print('Reading coolLed settings:')
        ser.reset_input_buffer()
        ser.write(str.encode("CSS?\r\n"))
        time.sleep(0.5)   # Wait 
        out = ser.readline() 
        response = out.strip().decode('utf-8')
        print(response)

        time.sleep(0.1)

        print('Resetting coolLed settings:')
        ser.write(str.encode("CSSAXF000BXN000CXN000DXN000\r\n"))
        time.sleep(0.5)   # Wait 
        out = ser.readline() 
        response = out.strip().decode('utf-8')
        print(response)

        time.sleep(0.1)

        print('Writing coolLed settings:')
        ser.write(str.encode("CSSASN005BSN005CSN005DSN005\r\n")) ### CSS (all) A (orBCD channel) S (orX selected) N (orF on) 050 (50 intensity)
        time.sleep(0.5)   # Wait 
        out = ser.readline() 
        response = out.strip().decode('utf-8')
        print(response)

        time.sleep(2)

        print('Resetting coolLed settings:')
        ser.write(str.encode("CSSAXF000BXN000CXN000DXN000\r\n"))
        time.sleep(0.5)   # Wait 
        out = ser.readline() 
        response = out.strip().decode('utf-8')
        print(response)
        
        print('End of commands')

def turn_everything_off():

    import pandas as pd
    s_machines = pd.read_csv('settings\settings_machines.txt', delimiter = '\t',index_col=False).to_dict()
    port = s_machines['coolLed'][0] 
        
    with serial.Serial(port, BAUD_RATE) as ser:
        print('Resetting coolLed settings:')
        ser.write(str.encode("CSSAXF000BXN000CXN000DXN000\r\n"))
        time.sleep(0.5)   # Wait for Printrbot to initialize
        out = ser.readline() 
        response = out.strip().decode('utf-8')
        print(response)

if __name__ == "__main__":

    # GRBL_port_path = '/dev/tty.usbserial-A906L14X'
    GRBL_port_path = 'COM4'

    stream_TX(GRBL_port_path)

    print('EOF')
