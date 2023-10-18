import os,time,glob,sys,time,tqdm,cv2
# import lights.labjackU3_control
# import lights.coolLed_control
import settings.get_settings
import movement.simple_stream
import movement.move
# import camera.camera_control

# def turn_everything_off_at_exit():
#     lights.labjackU3_control.turn_off_everything()
#     cv2.destroyAllWindows()
#     # lights.coolLed_control.turn_everything_off()

import atexit

if __name__ == "__main__":

    # atexit.register(turn_everything_off_at_exit)
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
    run_as_testing = False

    settings.get_settings.check_grbl_port(s_machines['grbl'][0], run_as_testing = True)
    s_grbl_settings = movement.simple_stream.get_settings(s_machines['grbl'][0])
    s_grbl_settings_df,s_grbl_settings = settings.get_settings.convert_GRBL_settings(s_grbl_settings)

    # run setup test to make sure everything works or throw error

    # s_todays_runs = settings.get_settings.update_todays_runs(s_todays_runs, overwrite=True)
    # movement.simple_stream.home_GRBL(s_machines['grbl'][0], testing = run_as_testing, camera = None) # home the machine

    coords = dict()
    coords['x_pos'] = -100.0
    coords['y_pos'] = -100.0
    coords['z_pos'] = -10.0
    serial_connection = movement.move.get_serial_connection(s_machines['grbl'][0])

    serial_connection.flush()
    command = str.encode('?')
    serial_connection.write(command)
    a = serial_connection.read_until('ok')
    print(a)
    time.sleep(0.1)

    serial_connection.flush()
    command = str.encode('$X')
    serial_connection.write(command)
    a = serial_connection.read_until('ok')
    print(a)
    time.sleep(0.1)

    serial_connection.flush()
    command = str.encode('$H')
    serial_connection.write(command)
    a = serial_connection.read_until('ok')
    print(a)
    time.sleep(0.1)

    serial_connection.flush()
    command = str.encode('g0 x-100 y-100 z-10')
    serial_connection.write(command)
    a = serial_connection.read_until('ok')
    print(a)
    time.sleep(0.1)

    serial_connection.flush()
    command = str.encode('g0 x-100 y-100 z-10')
    serial_connection.write(command)
    a = serial_connection.read_until('ok')
    print(a)
    time.sleep(0.1)
    # time.sleep(1)

    # movement.simple_stream.move_XYZ(coords,s_machines['grbl'][0], testing=False, round_decimals = 4, camera = None)

    # time.sleep(1)

    # movement.simple_stream.move_XYZ(coords,s_machines['grbl'][0], testing=False, round_decimals = 4, camera = None)

    # print('eof')