import os,time,glob,sys,time,tqdm,cv2
import lights.labjackU3_control
import lights.coolLed_control
import settings.get_settings
import movement.simple_stream
import camera.camera_control

# def turn_everything_off_at_exit():
#     lights.labjackU3_control.turn_off_everything()
#     cv2.destroyAllWindows()
#     # lights.coolLed_control.turn_everything_off()

import atexit

if __name__ == "__main__":

    # atexit.register(turn_everything_off_at_exit)

    # NEED TO HAVE AT LEAST ONE CAMERA, GRBL, AND LABJACK MACHINE PLUGGED IN

    # print(sys.argv)

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'output')

    # read in settings from files
    s_plate_names_and_opts = settings.get_settings.get_plate_names_and_opts()
    s_plate_positions = settings.get_settings.get_plate_positions()
    s_terasaki_positions = settings.get_settings.get_terasaki_positions()
    s_machines = settings.get_settings.get_machine_settings()
    s_camera_settings = settings.get_settings.get_basic_camera_settings()
    s_todays_runs = settings.get_settings.get_todays_runs()

    # read in settings from machines
    settings.get_settings.check_grbl_port(s_machines['grbl'][0])
    s_grbl_settings = movement.simple_stream.get_settings(s_machines['grbl'][0])
    s_grbl_settings_df,s_grbl_settings = settings.get_settings.convert_GRBL_settings(s_grbl_settings)

    run_as_testing = True

    # run setup test to make sure everything works or throw error

    movement.simple_stream.home_GRBL(s_machines['grbl'][0], testing = run_as_testing, camera = None) # home the machine
    s_todays_runs = settings.get_settings.update_todays_runs(s_todays_runs, overwrite=run_as_testing)
    # d = lights.labjackU3_control.setup_labjack(verbose=run_as_testing)    # test the blue and red lights
    # lights.labjackU3_control.blink_led(d)

    # coolLED_port = s_machines['coolLed'][0] # test the fluorescent lights (if applicable)
    # lights.coolLed_control.test_coolLed_connection(coolLED_port, testing= run_as_testing)

    # # experiment set up -- find optimal route ()
    # lights.labjackU3_control.turn_off_everything(d)

    plate_index = []
    plate_positions = []
    for this_plate_index in s_plate_names_and_opts['plate_name']:
        this_plate_name = s_plate_names_and_opts['plate_name'][this_plate_index]
        if this_plate_name != 'NONE':
            plate_index.append(this_plate_index)

    # lights.labjackU3_control.turn_on_red(d)
    # # # run experiment
    for this_plate_index in plate_index:
        this_plate_parameters,this_plate_position = settings.get_settings.get_indexed_dict_parameters(s_plate_names_and_opts,s_plate_positions,this_plate_index)
    
        print(this_plate_parameters)
        print(this_plate_position)

        movement.simple_stream.move_XY_at_Z_travel(this_plate_position,s_machines['grbl'][0],z_travel_height = s_machines['grbl'][2], testing=False, round_decimals = 4, camera = None)

        # camera.camera_control.simple_capture_data(s_camera_settings, plate_parameters=this_plate_parameters, testing=run_as_testing, output_dir=output_dir)
        # t = lights.labjackU3_control.turn_on_blue(d, return_time=True)
        # camera.camera_control.capture_single_image_wait_N_seconds(s_camera_settings, timestart=t, excitation_amount = s_machines['labjack'][3], 
        #                                                           plate_parameters=this_plate_parameters, testing=False, output_dir=output_dir)
        # lights.labjackU3_control.turn_off_blue(d)
        # camera.camera_control.simple_capture_data(s_camera_settings, plate_parameters=this_plate_parameters, testing=False, output_dir=output_dir)
        # lights.labjackU3_control.turn_off_blue(d)
        time.sleep(1)
        print('')

    movement.simple_stream.home_GRBL(s_machines['grbl'][0], testing = True,camera=None) # home the machine
    # lights.labjackU3_control.turn_off_everything(d) # make sure everything if off

    for this_plate_index in plate_index:
        this_plate_parameters,this_plate_position = settings.get_settings.get_indexed_dict_parameters(s_plate_names_and_opts,s_plate_positions,this_plate_index)
    
        print(this_plate_parameters)
        print(this_plate_position)

        # adjust for the imaging head 7 positions 
        this_plate_position['y_pos'] = this_plate_position['y_pos'] + s_terasaki_positions['y_offset_to_fluor_mm'][0]

        # move to xy base plate position (do not move back down)
        movement.simple_stream.move_XY_at_Z_travel(this_plate_position,s_machines['grbl'][0],z_travel_height = s_machines['grbl'][2], testing=False, go_back_down = False, round_decimals = 4, camera = None)
        # calculate the calibration corner coordinates
        calibration_coordinates = dict()
        calibration_coordinates['x_pos'] = this_plate_position['x_pos'] + s_terasaki_positions['calib_x_pos_mm'][0]
        calibration_coordinates['y_pos'] = this_plate_position['y_pos'] + s_terasaki_positions['calib_y_pos_mm'][0]
        calibration_coordinates['z_pos'] = s_terasaki_positions['calib_z_pos_mm'][0]
        # move to the calibration side
        movement.simple_stream.move_XYZ(calibration_coordinates,s_machines['grbl'][0], testing=False, round_decimals = 4, camera = None)

        # run the calibration script 
        print('running Z calibration script -------------------------------------------------------------------------------------')
        calibration_coordinates['z_pos'] = calibration_coordinates['z_pos'] - 10

        for well_index,this_terasaki_well_xy in enumerate(zip(s_terasaki_positions['x_relative_pos_mm'].values(),s_terasaki_positions['y_relative_pos_mm'].values())):
            this_plate_parameters['well_name'] = s_terasaki_positions['name'][well_index]
            terasaki_well_coords = dict()
            terasaki_well_coords['x_pos'] = this_plate_position['x_pos'] + this_terasaki_well_xy[0]
            terasaki_well_coords['y_pos'] = this_plate_position['y_pos'] + this_terasaki_well_xy[1]
            terasaki_well_coords['z_pos'] = calibration_coordinates['z_pos']
            print(well_index, terasaki_well_coords)
            movement.simple_stream.move_XYZ(terasaki_well_coords,s_machines['grbl'][0], testing=False, round_decimals = 4, camera = None)
            print('imaging')
            # camera.camera_control.simple_capture_data_fluor(s_camera_settings, plate_parameters=this_plate_parameters, testing=False, output_dir=output_dir)
            time.sleep(1)

    # shut everything down 
    # lights.labjackU3_control.turn_off_everything(d)

    print('eof')