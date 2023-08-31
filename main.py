import os,time,glob,sys,time,tqdm,cv2
import lights.labjackU3_control
import lights.coolLed_control
import settings.get_settings
import movement.simple_stream
import camera.camera_control

def turn_everything_off_at_exit():
    lights.labjackU3_control.turn_off_everything()
    cv2.destroyAllWindows()
    # lights.coolLed_control.turn_everything_off()

import atexit

if __name__ == "__main__":

    atexit.register(turn_everything_off_at_exit)

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'output')

    # read in settings
    s_plate_names_and_opts = settings.get_settings.get_plate_names_and_opts()
    s_plate_positions = settings.get_settings.get_plate_positions()
    s_machines = settings.get_settings.get_machine_settings()
    s_grbl_settings = movement.simple_stream.get_settings(s_machines['grbl'][0])
    _,s_grbl_settings = settings.get_settings.convert_GRBL_settings(s_grbl_settings)
    s_camera_settings = settings.get_settings.get_basic_camera_settings()

    # run setup test to make sure everything works or throw error

    # # movement.simple_stream.home_GRBL(s_machines['grbl'][0], testing = True) # home the machine
    d = lights.labjackU3_control.setup_labjack()    # test the blue and red lights
    lights.labjackU3_control.blink_led(d)

    # # coolLED_port = s_machines['coolLed'][0] # test the fluorescent lights (if applicable)
    # # lights.coolLed_control.stream_TX(coolLED_port)

    # experiment set up -- find optimal route ()
    lights.labjackU3_control.turn_off_everything(d)

    plate_index = []
    plate_positions = []
    for this_plate_index in s_plate_names_and_opts['plate_name']:
        this_plate_name = s_plate_names_and_opts['plate_name'][this_plate_index]
        if this_plate_name != 'NONE':
            plate_index.append(this_plate_index)

    # run experiment 
    for this_plate_index in plate_index:
        this_plate_parameters,this_plate_position = settings.get_settings.get_indexed_dict_parameters(s_plate_names_and_opts,s_plate_positions,this_plate_index)
    
        print(this_plate_parameters)
        print(this_plate_position)

        # movement.simple_stream.move_XY_at_z_travel(this_plate_position,s_machines['grbl'][0],s_machines['grbl'][2])

        camera.camera_control.simple_capture_data(s_camera_settings,this_plate_parameters, testing=False, output_dir=output_dir)
        lights.labjackU3_control.turn_on_blue(d)
        camera.camera_control.simple_capture_data(s_camera_settings,this_plate_parameters, testing=False, output_dir=output_dir)
        lights.labjackU3_control.turn_off_blue(d)
        print('')

    ## for each plate:
    ##   do work

    # shut everything down 
    lights.labjackU3_control.turn_off_everything(d)

    print('eof')