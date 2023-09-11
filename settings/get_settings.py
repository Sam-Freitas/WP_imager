import pandas as pd
import numpy as np
import datetime
import os

def get_base_path():

    dir_path = os.path.dirname(os.path.realpath(__file__))

    return dir_path

def get_todays_runs():

    path = os.path.join(get_base_path(),'todays_runs.txt')

    current_date_for_file = datetime.datetime.now().strftime("%Y-%m-%d")

    if os.path.isfile(path):
        with open(path, 'r') as f:
            out = f.read()
            f.close()
    else:
        with open(path, 'w') as f:
            f.write(current_date_for_file)
            f.write('\n0 0 0')
            f.close()
        with open(path, 'r') as f:
            out = f.read()
            f.close()

    assert '\n' in out
    out = out.split()

    if out[0] != current_date_for_file: # if the dates do not match up anymore then delete the file and reset
        os.remove(path)
        out = get_todays_runs()

    return out

def write_todays_runs(data):

    path = os.path.join(get_base_path(),'todays_runs.txt')

    with open(path, 'w') as f:
        f.write(data[0])
        f.write('\n' + data[1] + ' ' + data[2] + ' ' + data[3])
        f.close()

def update_todays_runs(s_todays_runs = None, overwrite = False, overwrite_val = ['1','0','0']):

    if s_todays_runs == None:
        s_todays_runs = get_todays_runs()

    path = os.path.join(get_base_path(),'todays_runs.txt')
    runs = np.asarray(s_todays_runs[1:])

    if overwrite == False:
        idx = np.where(runs == '0')
        next_idx = np.min(idx)
        runs[next_idx] = '1'
        s_todays_runs[1:] = runs

    if overwrite == True:
        runs = overwrite_val
        s_todays_runs[1:] = runs

    print('Run information: ')
    print('date - run1 - run2 - run3')
    print(s_todays_runs)

    write_todays_runs(s_todays_runs)

    return s_todays_runs

def get_plate_names_and_opts(): # this gets the plate parameters (this SHOULD change )

    print('Getting plate names and options')
    path = os.path.join(get_base_path(),'settings_plate_names_and_opts.csv')
    df = pd.read_csv(path, delimiter = ',',index_col=False)
    df = df.to_dict()

    return df

def get_plate_positions(): # this gets the plate positions (this shouldnt change )

    print('Getting plate positions')
    path = os.path.join(get_base_path(),'settings_plate_positions.csv')
    df = pd.read_csv(path, delimiter = ',',index_col=False)
    df = df.to_dict()

    return df

def get_machine_settings(): # this reads the machine settings file

    print('Getting machines settings')
    path = os.path.join(get_base_path(),'settings_machines.csv')
    df = pd.read_csv(path, delimiter = ',',index_col=False)
    df = df.to_dict()

    return df

def get_basic_camera_settings():

    print('Getting camera(s) settings')
    path = os.path.join(get_base_path(),'settings_cameras.csv')
    df = pd.read_csv(path, delimiter = ',',index_col=False)
    df = df.to_dict()

    return df

def convert_GRBL_settings(settings): # this converts grblHAL settings to readable dictionary 

    path = os.path.join(get_base_path(),'settings_grbl_base.txt')
    df_descriptor = df = pd.read_csv(path, delimiter = ';',index_col=False,header = None)

    new_settings = dict()

    for i,this_setting in enumerate(settings): # parse then covert the settings into readable format
        if 'ok' not in this_setting:
            setting_number = this_setting[1:this_setting.index('=')]
            setting_set = this_setting[(this_setting.index('=') + 1):]
            setting_descriptor = df_descriptor.values[i,1]
            new_settings[setting_number] = [setting_set, setting_descriptor]

    df = pd.DataFrame(new_settings)

    return df, new_settings

def get_indexed_dict_parameters(s_plate_names_and_opts,s_plate_positions,plate_index): # this returns the plate parameters for that specified index

    plate_parameters,plate_position = dict(), dict()    # create the blank dicts

    for parameter in list(s_plate_names_and_opts.keys()): # get all the parameters at INDEX
        plate_parameters[parameter] = s_plate_names_and_opts[parameter][plate_index]

    for parameter in list(s_plate_positions.keys()):  # get all the positions at INDEX
        plate_position[parameter] = s_plate_positions[parameter][plate_index]

    return plate_parameters,plate_position