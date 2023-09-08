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

    print(sys.argv)

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'output')

    # read in settings
    s_plate_names_and_opts = settings.get_settings.get_plate_names_and_opts()
    s_plate_positions = settings.get_settings.get_plate_positions()
    s_machines = settings.get_settings.get_machine_settings()
    s_grbl_settings = movement.simple_stream.get_settings(s_machines['grbl'][0])
    _,s_grbl_settings = settings.get_settings.convert_GRBL_settings(s_grbl_settings)
    s_camera_settings = settings.get_settings.get_basic_camera_settings()
    s_todays_runs = settings.get_settings.get_todays_runs()

    run_as_testing = True

    # run setup test to make sure everything works or throw error

    # # movement.simple_stream.home_GRBL(s_machines['grbl'][0], testing = True) # home the machine
    s_todays_runs = settings.get_settings.update_todays_runs(s_todays_runs, overwrite=run_as_testing)
    d = lights.labjackU3_control.setup_labjack(verbose=run_as_testing)    # test the blue and red lights
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

    lights.labjackU3_control.turn_on_red(d)
    # run experiment 
    for this_plate_index in plate_index:
        this_plate_parameters,this_plate_position = settings.get_settings.get_indexed_dict_parameters(s_plate_names_and_opts,s_plate_positions,this_plate_index)
    
        print(this_plate_parameters)
        print(this_plate_position)

        movement.simple_stream.move_XY_at_z_travel(this_plate_position,s_machines['grbl'][0],s_machines['grbl'][2], testing=run_as_testing)

        camera.camera_control.simple_capture_data(s_camera_settings, plate_parameters=this_plate_parameters, testing=run_as_testing, output_dir=output_dir)
        t = lights.labjackU3_control.turn_on_blue(d, return_time=True)
        camera.camera_control.capture_single_image_wait_N_seconds(s_camera_settings, timestart=t, excitation_amount = s_machines['labjack'][3], 
                                                                  plate_parameters=this_plate_parameters, testing=False, output_dir=output_dir)
        lights.labjackU3_control.turn_off_blue(d)
        camera.camera_control.simple_capture_data(s_camera_settings, plate_parameters=this_plate_parameters, testing=False, output_dir=output_dir)
        lights.labjackU3_control.turn_off_blue(d)
        print('')

    ## for each plate:
    ##   do work

    for this_plate_index in plate_index:
        this_plate_parameters,this_plate_position = settings.get_settings.get_indexed_dict_parameters(s_plate_names_and_opts,s_plate_positions,this_plate_index)
    
        print(this_plate_parameters)
        print(this_plate_position)

        # calculate locations of 
            # autofocus corner 
            # terasaki wells (xy) z to be calculated

        # move to xy base plate position (or just move straight to the autocus corner ????) move_XY_at_z_travel
        # move head to near the autofocus place move_XY_at_z_travel
        # run autofocus algorithm 
        # for each terasaki well
            # move to terasaki well 

        # movement.simple_stream.move_XY_at_z_travel(this_plate_position,s_machines['grbl'][0],s_machines['grbl'][2])

    # shut everything down 
    lights.labjackU3_control.turn_off_everything(d)

    print('eof')