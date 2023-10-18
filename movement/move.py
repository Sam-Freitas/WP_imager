import cv2
import serial
import time
from threading import Event
try:
    from camera.camera_control import capture_images_for_time
except:
    print('cant import camera commands')

BAUD_RATE = 115200


def send_wake_up(ser,sleep_amount = 2):
    # Wake up
    # Hit enter a few times to wake the Printrbot
    ser.timeout = 2
    ser.write(str.encode("\n"))
    ser.write(str.encode("\n"))
    time.sleep(sleep_amount)   # Wait for Printrbot to initialize
    ser.reset_input_buffer()  # Flush startup text in serial input # ser.flushInput()
    ser.reset_output_buffer()

def get_serial_connection(GRBL_port_path, ser = None):

    if ser is None:
        ser = serial.Serial(GRBL_port_path,baudrate=BAUD_RATE, timeout = 2)
        send_wake_up(ser)

    print('test')

    return ser