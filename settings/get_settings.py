import pandas as pd
import os

def get_base_path():

    dir_path = os.path.dirname(os.path.realpath(__file__))

    return dir_path

def get_plate_names_and_opts(): # this gets the plate parameters (this SHOULD change )

    print('Getting plate names and options')
    path = os.path.join(get_base_path(),'settings_plate_names_and_opts.txt')
    df = pd.read_csv(path, delimiter = '\t',index_col=False)
    df = df.to_dict()

    return df

def get_plate_positions(): # this gets the plate positions (this shouldnt change )

    print('Getting plate positions')
    path = os.path.join(get_base_path(),'settings_plate_positions.txt')
    df = pd.read_csv(path, delimiter = '\t',index_col=False)
    df = df.to_dict()

    return df

def get_machine_settings(): # this reads the machine settings file

    print('Getting machines settings')
    path = os.path.join(get_base_path(),'settings_machines.txt')
    df = pd.read_csv(path, delimiter = '\t',index_col=False)
    df = df.to_dict()

    return df

def get_basic_camera_settings():

    print('Getting camera(s) settings')
    path = os.path.join(get_base_path(),'settings_cameras.txt')
    df = pd.read_csv(path, delimiter = '\t',index_col=False)
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