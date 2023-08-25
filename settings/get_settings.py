import pandas as pd


def get_plate_names_and_opts():

    print('Getting plate names and options')
    df = pd.read_csv('settings\settings_plate_names_and_opts.txt', delimiter = '\t',index_col=False)
    df = df.to_dict()

    return df

def get_plate_positions():

    print('Getting plate positions')
    df = pd.read_csv('settings\settings_plate_positions.txt', delimiter = '\t',index_col=False)
    df = df.to_dict()


    return df

def get_machine_settings():

    print('Getting machines settings')
    df = pd.read_csv('settings\settings_machines.txt', delimiter = '\t',index_col=False)
    df = df.to_dict()

    return df


def convert_GRBL_settings(settings):

    df_descriptor = df = pd.read_csv('settings\settings_grbl_base.txt', delimiter = ';',index_col=False,header = None)

    new_settings = dict()

    for i,this_setting in enumerate(settings):
        if 'ok' not in this_setting:
            setting_number = this_setting[1:this_setting.index('=')]
            setting_set = this_setting[(this_setting.index('=') + 1):]
            setting_descriptor = df_descriptor.values[i,1]
            new_settings[setting_number] = [setting_set, setting_descriptor]
            # df[setting_number] = setting_descriptor
            # new_settings.append(this_setting)

    df = pd.DataFrame(new_settings)

    return df, new_settings