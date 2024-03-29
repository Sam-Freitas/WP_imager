import tkinter as tk
import pandas as pd
from numpy import asarray, arange, nonzero, zeros,uint8
from PIL import Image, ImageTk
import os
import settings.get_settings
from tkinter import messagebox, PhotoImage
import atexit, json
import pathlib

### Default values for the experiments 
### plate name, experiment name, lifepsan, healthspan, fluor timing, uv, blue, green, red
DEFAULT_VALUES = ['NONE','NONE',1,0,'100',0,0,0,0]
# Excitation amount from the coolLED system --- must be integers and total must not go above ~20
UV_EX_MAGNITUDE = 1
BLUE_EX_MAGNITUDE = 10
GREEN_EX_MAGNITUDE = 10
RED_EX_MAGNITUDE = 5

# get basic pathings for the system
path_to_file = pathlib.Path(__file__).parent.resolve() 
path_to_settings_folder = path_to_settings_folder = os.path.join(path_to_file, "settings")

BUTTON_W = 10  # Set the size of the buttons
BUTTON_H = 5
POPUP_WIDTH = 300  # Set the width of the popup window
POPUP_HEIGHT = 350  # Set the height of the popup window

s_plate_names_and_opts = settings.get_settings.get_plate_names_and_opts()

def save_function():
    df = pd.DataFrame(s_plate_names_and_opts)
    print(os.path.join(path_to_settings_folder,'settings_plate_names_and_opts.csv'))
    df.to_csv(os.path.join(path_to_settings_folder,'settings_plate_names_and_opts.csv'),index= False)

# save whatever was just changed before quitting
def exit_function():
    save_function()

def update_s_plate_names_and_opts(index,options):

    # this (badly) loops through the options and changes the boolean values to what they should be

    if options is not None:
        for i,val in enumerate(options):
            if i < 4: #plate name, experiment name, lifepsan, healthspan,
                if options[i] == True:
                    options[i] = 1
                if options[i] == False:
                    options[i] = 0
            if i == 4: # fluor timing
                options[i] = str(options[i])
                # if options[i] == True:
                #     options[i] = 100
                # if options[i] == False:
                #     options[i] = 0
            if i == 5: # UV excitation 
                if options[i] == True:
                    options[i] = UV_EX_MAGNITUDE
                if options[i] == False:
                    options[i] = 0
            if i == 6: # Blue excitation 
                if options[i] == True:
                    options[i] = BLUE_EX_MAGNITUDE
                if options[i] == False:
                    options[i] = 0
            if i == 7: # Green excitation 
                if options[i] == True:
                    options[i] = GREEN_EX_MAGNITUDE
                if options[i] == False:
                    options[i] = 0
            if i == 8: # RED excitation 
                if options[i] == True:
                    options[i] = RED_EX_MAGNITUDE
                if options[i] == False:
                    options[i] = 0
            
    else:
        options = DEFAULT_VALUES

    s_plate_names_and_opts['plate_name'][index] = options[0]
    s_plate_names_and_opts['experiment_name'][index] = options[1]
    s_plate_names_and_opts['lifespan'][index] = options[2]
    s_plate_names_and_opts['fluorescence'][index] = options[3]
    s_plate_names_and_opts['fluorescence_times'][index] = options[4]
    s_plate_names_and_opts['fluorescence_UV'][index] = options[5]
    s_plate_names_and_opts['fluorescence_BLUE'][index] = options[6]
    s_plate_names_and_opts['fluorescence_GREEN'][index] = options[7]
    s_plate_names_and_opts['fluorescence_RED'][index] = options[8]
    save_function()

def update_button_with_new_preferences(row,col,options):

    index = get_index_from_row_col(row,col)

    update_s_plate_names_and_opts(index,options)

    if options is not None:
        buttons[row][col]['text'] = options[0] + '\n' + options[1] + '\n' + f'{row}-{col}'
        buttons[row][col]['bg'] = 'Green'
        if bool(options[3]):
            buttons[row][col]['bg'] = 'Purple'
    if options == None:
        buttons[row][col]['text'] = DEFAULT_VALUES[0] + '\n' + DEFAULT_VALUES[1] + '\n' + f'{row}-{col}'
        buttons[row][col]['bg'] = 'White'

def get_index_from_row_col(row,col): # get the row and column as the plate index for ease of use

    rows = asarray(list(s_plate_names_and_opts['row'].values())) # get the rows dict as an array
    columns = asarray(list(s_plate_names_and_opts['column'].values())) # get the rows dict as an array
    rows_bool = (rows==row) # mask out the rows and columsn
    columns_bool = (columns==col)

    index_bool = rows_bool*columns_bool # find the overlap
    index = int(nonzero((1*index_bool) * (arange(1,len(index_bool)+1)))[0]) # convert to int

    return index

def get_settings_from_index(index):

    temp = dict()
    for key in s_plate_names_and_opts.keys():
        temp[key] = s_plate_names_and_opts[key][index]

    return temp

def convert_string_to_bool_list(in_str):

    boolean_list = [False] * len(in_str)

    for i in range(3):
        if in_str[i] == '1':
            boolean_list[i] = True

    return boolean_list

def convert_bool_list_to_integer_string(boolean_list):

    integer_amount = '000'
    for i,b in enumerate(boolean_list):
        if b:
            integer_amount = integer_amount[:i] + '1' + integer_amount[i + 1:]

    return integer_amount

def button_click(row, col):
    # Create a new tkinter window
    popup = tk.Toplevel(root)
    popup.title(f"          {row}-{col}")
    popup.geometry(f"{POPUP_WIDTH}x{POPUP_HEIGHT}")

    index = get_index_from_row_col(row,col)

    # Create labels and entries for the first two options with default value 'NONE'
    label1 = tk.Label(popup, text="Plate name -- Plate ID:")
    label1.pack()
    entry1 = tk.Entry(popup)
    entry1.insert(0, s_plate_names_and_opts['plate_name'][index])  # Set default value
    entry1.pack()

    label2 = tk.Label(popup, text="Experiment name:")
    label2.pack()
    entry2 = tk.Entry(popup)
    entry2.insert(0, s_plate_names_and_opts['experiment_name'][index])  # Set default value
    entry2.pack()

    button_settings = get_settings_from_index(index)
    print(json.dumps(button_settings, indent = 4))

    fluorescent_times_bool = convert_string_to_bool_list(str(button_settings['fluorescence_times']))

    # Create checkbuttons for the remaining seven options with default value True
    check_var = [tk.BooleanVar(value=bool(button_settings['lifespan'])),
                 tk.BooleanVar(value=bool(button_settings['fluorescence'])),
                 tk.BooleanVar(value=bool(fluorescent_times_bool[0])),
                 tk.BooleanVar(value=bool(fluorescent_times_bool[1])),
                #  tk.BooleanVar(value=bool(button_settings['fluorescence_times'])),
                 tk.BooleanVar(value=bool(button_settings['fluorescence_UV'])),
                 tk.BooleanVar(value=bool(button_settings['fluorescence_BLUE'])),
                 tk.BooleanVar(value=bool(button_settings['fluorescence_GREEN'])),
                 tk.BooleanVar(value=bool(button_settings['fluorescence_RED']))]
    checkbutton_names = ["Lifespan", "Fluorescence", "Fluorescence Time(s) 1", "Fluorescence Time(s) 2", "UV excitation", "BLUE excitation", "GREEN excitation", "RED excitation"]
    counter = 0
    # populate the buttons
    for i in range(len(check_var)):

        checkbutton = tk.Checkbutton(popup, text=checkbutton_names[i], variable=check_var[i])
        checkbutton.pack()
        counter =+ 1

    def return_values():

        # get experiment and plate names
        option1 = entry1.get()
        option2 = entry2.get()
        # checkbutton_names = ["Lifespan", "Fluorescence", "Fluorescence Time(s) 1", "Fluorescence Time(s) 2", "UV excitation", "BLUE excitation", "GREEN excitation", "RED excitation"]
        # get plate options or known as checkbutton names
        option3_to_end = [var.get() for var in check_var]

        fluor_times_list = option3_to_end[2:4] # get the two fluor times
        fluor_times_int_str = convert_bool_list_to_integer_string(fluor_times_list)
        option3_to_end[2:4] = [fluor_times_int_str] # update the list 

        options = [option1, option2] + option3_to_end

        # messagebox.showinfo("Button Clicked", f"Button ({row}, {col}) clicked with options: {options}")
        print("Button Clicked", f"Button ({row}, {col}) clicked with options: {options}")

        update_button_with_new_preferences(row,col,options)

        popup.destroy()

    def reset_values():
        option1 = entry1.get()
        option2 = entry2.get()
        option3_to_9 = [var.get() for var in check_var]

        options = [option1, option2] + option3_to_9

        confirm_reset = messagebox.askyesno("Confirmation", "Are you sure you want to reset this value?")
        # messagebox.showinfo("Button Clicked", f"Button ({row}, {col}) clicked with options: {options}")
        # print("Button Clicked", f"Button ({row}, {col}) clicked with options: {options}")

        if confirm_reset:
            update_button_with_new_preferences(row,col,None)

        popup.destroy()

    ok_button = tk.Button(popup, text="OK", command=return_values)
    ok_button.pack()

    delete_button = tk.Button(popup, text="DELETE", command=reset_values)
    delete_button.pack()

def numpy_to_photoimage(numpy_array):
    # Convert the NumPy array to a PIL image
    pil_image = Image.fromarray(numpy_array)
    # Convert the PIL image to a PhotoImage object
    photo_image = ImageTk.PhotoImage(pil_image)
    return photo_image

if __name__ == "__main__":
    atexit.register(exit_function)
    # Create the main window
    root = tk.Tk()
    root.title("                                                                          WP Imager ------ configurable settings")

    # Get the screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate the x and y coordinates to center the grid
    x = (screen_width - (8 * BUTTON_W*10)) // 2
    y = (screen_height - (9 * BUTTON_H*20)) // 2

    # Set the dimensions of the main window
    root.geometry(f"{8 * round(BUTTON_W*8.4)}x{9 * BUTTON_H*18}+{x}+{y}")

    # Create a 9x8 grid of buttons
    buttons = [[None for _ in range(8)] for _ in range(9)]

    index = 0
    for i in range(9):
        for j in range(8):

            text = s_plate_names_and_opts['plate_name'][index] + '\n' + s_plate_names_and_opts['experiment_name'][index] + '\n' + f'{i}-{j}'

            fluor_option = bool(s_plate_names_and_opts['fluorescence'][index])
            bg = zeros(shape= (x,y,3)).astype(uint8)

            if 'NONE' in text:
                bg = 'White'
            else:
                bg = 'Green'

            if 'NONE' not in text and fluor_option:
                bg = 'Purple'

            buttons[i][j] = tk.Button(root, text=text, 
                                    command=lambda i=i, j=j: button_click(i, j), 
                                    width=BUTTON_W, height=BUTTON_H,
                                    bg = bg)
                                    # image= bg)
            buttons[i][j].grid(row=i, column=j, padx=2, pady=2)
            index += 1

    # Start the tkinter event loop
    root.mainloop()
