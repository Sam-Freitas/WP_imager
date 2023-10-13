import tkinter as tk
import pandas as pd
from numpy import asarray, arange, nonzero
import os
import settings.get_settings
from tkinter import messagebox
import atexit

### plate name, experiment name, lifepsan, healthspan, fluor timing, uv, blue, green, red
DEFAULT_VALUES = ['NONE','NONE',1,0,100,0,0,0,0]

path_to_settings_folder = path_to_settings_folder = os.path.join(os.getcwd(), "settings")

BUTTON_W = 10  # Set the size of the buttons
BUTTON_H = 5
POPUP_WIDTH = 300  # Set the width of the popup window
POPUP_HEIGHT = 350  # Set the height of the popup window

s_plate_names_and_opts = settings.get_settings.get_plate_names_and_opts()

def exit_function():
    df = pd.DataFrame(s_plate_names_and_opts)
    print(os.path.join(path_to_settings_folder,'settings_plate_names_and_opts.csv'))
    df.to_csv(os.path.join(path_to_settings_folder,'settings_plate_names_and_opts.csv'),index= False)

def update_s_plate_names_and_opts(index,options):

    if options is not None:
        for i,val in enumerate(options):
            if i < 4:
                if options[i] == True:
                    options[i] = 1
                if options[i] == False:
                    options[i] = 0
            if i > 3:
                if options[i] == True:
                    options[i] = 100
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


def update_button_with_new_preferences(row,col,options):

    index = get_index_from_row_col(row,col)

    update_s_plate_names_and_opts(index,options)

    if options is not None:
        buttons[row][col]['text'] = options[0] + '\n' + options[1] + '\n' + f'{row}-{col}'
        buttons[row][col]['bg'] = 'Green'
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

def button_click(row, col):
    # Create a new tkinter window
    popup = tk.Toplevel(root)
    popup.title(f"          {row}-{col}")
    popup.geometry(f"{POPUP_WIDTH}x{POPUP_HEIGHT}")

    index = get_index_from_row_col(row,col)

    # Create labels and entries for the first two options with default value 'NONE'
    label1 = tk.Label(popup, text="Option 1:")
    label1.pack()
    entry1 = tk.Entry(popup)
    entry1.insert(0, s_plate_names_and_opts['plate_name'][index])  # Set default value
    entry1.pack()

    label2 = tk.Label(popup, text="Option 2:")
    label2.pack()
    entry2 = tk.Entry(popup)
    entry2.insert(0, s_plate_names_and_opts['experiment_name'][index])  # Set default value
    entry2.pack()

    # Create checkbuttons for the remaining seven options with default value True
    check_var = [tk.BooleanVar(value=True),tk.BooleanVar(value=False),tk.BooleanVar(value=True),
                 tk.BooleanVar(value=False),tk.BooleanVar(value=False),tk.BooleanVar(value=False),
                 tk.BooleanVar(value=False)]
    checkbutton_names = ["Lifespan", "Fluorescence", "Fluorescence Time(s)", "UV excitation", "BLUE excitation", "GREEN excitation", "RED excitation"]
    for i in range(7):
        if i == 2:  # Move the first check button to a different row
            checkbutton = tk.Checkbutton(popup, text=checkbutton_names[i], variable=check_var[i])
            checkbutton.pack()
        else:
            checkbutton = tk.Checkbutton(popup, text=checkbutton_names[i], variable=check_var[i])
            checkbutton.pack()

    def return_values():
        option1 = entry1.get()
        option2 = entry2.get()
        option3_to_9 = [var.get() for var in check_var]

        options = [option1, option2] + option3_to_9

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

            if 'NONE' in text:
                bg = 'White'
            else:
                bg = 'Green'

            buttons[i][j] = tk.Button(root, text=text, 
                                    command=lambda i=i, j=j: button_click(i, j), width=BUTTON_W, height=BUTTON_H,
                                    bg = bg)
            buttons[i][j].grid(row=i, column=j, padx=2, pady=2)
            index += 1

    # Start the tkinter event loop
    root.mainloop()
