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

def stream_gcode(port):
    # with contect opens file/connection and closes it if function(with) scope is left
    with serial.Serial(port, BAUD_RATE) as ser:
        ser.reset_input_buffer()
        ser.write(str.encode("CSS?\r\n"))
        time.sleep(0.5)   # Wait for Printrbot to initialize
        out = ser.readline() 
        response = out.strip().decode('utf-8')
        print(response)

        time.sleep(0.5)

        ser.write(str.encode("CSSASN050BSN050CSN050DSN050\r\n")) ### CSS (all) A (orBCD channel) S (orX selected) N (orF on) 050 (50 intensity)
        time.sleep(0.5)   # Wait for Printrbot to initialize
        out = ser.readline() 
        response = out.strip().decode('utf-8')
        print(response)

        time.sleep(2)

        ser.write(str.encode("CSSAXF050BXN050CXN050DXN050\r\n"))
        time.sleep(0.5)   # Wait for Printrbot to initialize
        out = ser.readline() 
        response = out.strip().decode('utf-8')
        print(response)
        
        print('End of gcode')

if __name__ == "__main__":

    # GRBL_port_path = '/dev/tty.usbserial-A906L14X'
    GRBL_port_path = 'COM4'

    stream_gcode(GRBL_port_path)

    print('EOF')
