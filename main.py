import os,time,glob,sys,time,tqdm
import lights.labjackU3_control
import lights.coolLed_control
import settings.get_settings
import movement.simple_stream

if __name__ == "__main__":

    # read in settings
    s_plate_names_and_opts = settings.get_settings.get_plate_names_and_opts()
    s_plate_positions = settings.get_settings.get_plate_positions()
    s_machines = settings.get_settings.get_machine_settings()
    s_grbl_settings = movement.simple_stream.get_settings(s_machines['grbl'][0])
    _,s_grbl_settings = settings.get_settings.convert_GRBL_settings(s_grbl_settings)
    s_camera_settings = []

    # run setup test to make sure everything works or throw error
    d = lights.labjackU3_control.setup_labjack()
    lights.labjackU3_control.blink_led(d)

    coolLED_port = s_machines['coolLed'][0]
    lights.coolLed_control.stream_TX(coolLED_port)

    # experiment set up

    ## find optimal route 

    # run experiment 

    ## for each plate:
    ##   do work


    # shut everything down 

    print('eof')