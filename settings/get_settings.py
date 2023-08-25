import pandas as pd


def get_plate_names_and_opts():

    print('Getting plate names and options')
    df = pd.read_csv('settings\settings_plate_names_and_opts.txt', delimiter = '\t',index_col=False)

    return df

def get_plate_positions():

    print('Getting plate positions')
    df = pd.read_csv('settings\settings_plate_positions.txt', delimiter = '\t',index_col=False)

    return df

def get_machine_settings():

    print('Getting machines settings')
    df = pd.read_csv('settings\settings_machines.txt', delimiter = '\t',index_col=False)

    return df


def convert_GRBL_settings(settings):

    new_settings = []

    for this_setting in settings:
        if 'ok' not in this_setting:
            new_settings.append(this_setting)

    return new_settings