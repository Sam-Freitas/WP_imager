import tkinter as tk
from tkinter import ttk
import serial
import time

class CNCController:
    def __init__(self, port, baudrate):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)

    def send_command(self, command):
        self.ser.write(command.encode())
        response = self.ser.readline().decode().strip()
        return response

    def close_connection(self):
        self.ser.close()

class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("bCNC-like G-code Sender")

        self.label = ttk.Label(self, text="Enter G-code Command:")
        self.label.pack()

        self.entry = ttk.Entry(self)
        self.entry.pack()

        self.send_button = ttk.Button(self, text="Send G-code", command=self.send_gcode)
        self.send_button.pack()

        self.controller = CNCController(port='COM6', baudrate=115200)  # Replace with your port and baudrate

    def send_gcode(self):
        command = self.entry.get() + "\n"
        response = self.controller.send_command(command)
        print(f"Sent command: {command}")
        print(f"Received response: {response}")

    def on_close(self):
        self.controller.close_connection()
        self.destroy()

if __name__ == "__main__":
    app = Application()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
