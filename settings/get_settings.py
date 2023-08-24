import pandas as pd


def get_plate_names_and_opts():

    df = pd.read_csv('settings\settings_plate_names_and_opts.txt', delimiter = '\t',index_col=False)

    return df

def get_plate_positions():

    df = pd.read_csv('settings\settings_plate_positions.txt', delimiter = '\t',index_col=False)

    return df

def get_machine_settings():

    df = pd.read_csv('settings\settings_machines.txt', delimiter = '\t',index_col=False)

    return df