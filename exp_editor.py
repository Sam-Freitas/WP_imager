import tkinter as tk
import settings.get_settings

s_plate_names_and_opts = settings.get_settings.get_plate_names_and_opts()
s_plate_positions = settings.get_settings.get_plate_positions()
s_terasaki_positions = settings.get_settings.get_terasaki_positions()


def button_click(row, col):
    print(f"Button clicked at ({row}, {col})")
    buttons[row][col]["text"] = "Clicked!"

def create_grid(root, rows, columns):
    index = 0
    for i in range(rows):
        row_buttons = []
        for j in range(columns):
            button = tk.Button(root, text= s_plate_names_and_opts['plate_name'][index] + '\n' + s_plate_names_and_opts['experiment_name'][index] + '\n' + f'{i}-{j}', 
                               command=lambda i=i, j=j: button_click(i, j), width=10, height=5)
            button.grid(row=i, column=j, padx=2, pady=2)  # Added padx and pady for spacing
            row_buttons.append(button)
            index +=1
        buttons.append(row_buttons)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Clickable Grid")

    rows = 9
    columns = 8
    buttons = []

    create_grid(root, rows, columns)

    root.mainloop()
