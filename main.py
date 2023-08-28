import os,time,glob,sys,time,tqdm
import lights.labjackU3_control
import lights.coolLed_control
import settings.get_settings
import movement.simple_stream
import camera.camera_control

if __name__ == "__main__":

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

    # experiment set up

    ## find optimal route ()

    # run experiment 
    for plate_index in s_plate_positions['plate_index']:
        this_plate_parameters,this_plate_position = settings.get_settings.get_indexed_dict_parameters(s_plate_names_and_opts,s_plate_positions,plate_index)

        if this_plate_parameters['plate_name'] != 'NONE':
            print(this_plate_parameters)
            print(this_plate_position)

            # movement.simple_stream.move_XY_at_z_travel(this_plate_position,s_machines['grbl'][0],s_machines['grbl'][2])

            camera.camera_control.simple_capture_data(s_camera_settings,this_plate_parameters, testing=True)
            print('')

    ## for each plate:
    ##   do work


    # shut everything down 

    print('eof')