import os,time,glob,sys,time,tqdm
import lights.labjackU3_control
import lights.coolLed_control

if __name__ == "__main__":

    # d = lights.labjackU3_control.setup_labjack()
    # lights.labjackU3_control.blink_led(d)

    coolLED_port = 'COM4'
    lights.coolLed_control.stream_gcode(coolLED_port)

    print('eof')