import numpy as np
import pandas as pd
import os

# this is a script that will overwrite settings and replace them with cvs's
# uncomment to run a specific section
# this WILL REWRITE ALL SETTINGS

path_to_settings_folder = os.path.join(os.getcwd(), "settings")

rows = 9  # rows and columns
cols = 8

xo = -50.375  #  approximate intial starting points
yo = -212.035  #  approximate intial starting points
zo = -90  #  approximate intial starting points

dx = -153.2  # distace between plates
dy = -115  # used to be 116

# #############################################
# ## this is for the settings plate postions
# header = ['plate_index','row','column','x_pos','y_pos','z_pos']
# df = pd.DataFrame(columns = header)

# i = 0
# j = 0
# counter = 0

# for r in range(rows):
#     i = 0
#     for c in range(cols):
#         df.loc[counter] = [counter,r,c,xo+(dx*i),yo+(dy*j),zo] #just xyz
#         counter = counter + 1
#         i = i +1
#     j = j+1

# df.to_csv(os.path.join(path_to_settings_folder,'settings_plate_positions.csv'),index= False)

# #############################################
# ## this is for the settings plate names and opts
# header = ['plate_index','row','column','plate_name','experiment_name','lifespan','fluorescence','fluorescence_times']
# df = pd.DataFrame(columns = header)
# i = 0
# j = 0
# counter = 0

# for r in range(rows):
#     i = 0
#     for c in range(cols):
#         df.loc[counter] = [counter ,r , c ,'NONE','NONE','1','1','100']
#         counter = counter + 1
#         i = i +1
#     j = j+1

# df.to_csv(os.path.join(path_to_settings_folder,'settings_plate_names_and_opts.csv'),index= False)

# #############################################
# ## this is for the setting machines

# header = ['Explanation','labjack','grbl','coolLed']
# df = pd.DataFrame(columns = header)

# df.loc[0] = ['COM_port','COM1','COM3','COM4']
# df.loc[1] = ['baud_rate','NONE','115200','9600']
# df.loc[2] = ['travel_height','NONE','-20','NONE']
# df.loc[3] = ['excitation_amount','9','NONE','NONE']

# df.to_csv(os.path.join(path_to_settings_folder,'settings_machines.csv'),index= False)

# #############################################
# ## this is for the setting machines

# header = ['Explanation','widefield','fluorescence']
# df = pd.DataFrame(columns = header)

# df.loc[0] = ['camera_port','0','1']
# df.loc[1] = ['pixel_width','5472','1080']
# df.loc[2] = ['pixel_height','3640','1080']
# df.loc[3] = ['max_framerate','6','14']
# df.loc[4] = ['time_between_images_seconds','5','0.1']
# df.loc[5] = ['time_of_single_burst_seconds','60','0']
# df.loc[6] = ['number_of_images_per_burst','12','3']
# df.loc[7] = ['img_file_format','png','png']
# df.loc[8] = ['img_pixel_depth','8','8']
# df.loc[9] = ['img_color','0','1']

# df.to_csv(os.path.join(path_to_settings_folder,'settings_cameras.csv'),index= False)

# #############################################
# ## this id for the grbl base

# # just open a nopepad++ and paste this in without the #, pandas is annoying

# # $0=10.0  ;  Step pulse time, microseconds
# # $1=255  ;  Step idle delay, milliseconds
# # $2=0  ;  Step pulse invert, mask
# # $3=0  ;  Step direction invert, mask
# # $4=0  ;  Invert step enable pin, boolean
# # $5=7  ;  Invert limit pins, boolean/mask
# # $6=1  ;  Invert probe pin, boolean
# # $8=0  ;  Ganged axes direction invert as bitfield
# # $9=1  ;  PWM Spindle as bitfield where setting bit 0 enables the rest
# # $10=511  ;  Status report options, mask
# # $11=0.010  ;  Junction deviation, millimeters
# # $12=0.002  ;  Arc tolerance, millimeters
# # $13=0  ;  Report in inches, boolean
# # $14=0  ;  Limit pins invert, mask
# # $15=0  ;  Coolant pins invert, mask
# # $16=0  ;  Spindle pins invert, mask
# # $17=0  ;  Control pins pullup disable, mask
# # $18=0  ;  Limit pins pullup disable, mask
# # $19=0  ;  Probe pin pullup disable, boolean
# # $20=0  ;  Soft limits enable, boolean
# # $21=1  ;  Hard limits enable, boolean
# # $22=1  ;  Homing cycle enable, boolean (Grbl) / mask (GrblHAL)
# # $23=3  ;  Homing direction invert, mask
# # $24=100.0  ;  Homing locate feed rate, mm/min
# # $25=2000.0  ;  Homing search seek rate, mm/min
# # $26=250  ;  Homing switch debounce delay, milliseconds
# # $27=5.000  ;  Homing switch pull-off distance, millimeters
# # $28=0.100  ;  G73 retract distance, in mm
# # $29=5.0  ;  Step pulse delay (ms)
# # $30=1000.000  ;  Maximum spindle speed, RPM
# # $31=0.000  ;  Minimum spindle speed, RPM
# # $32=0  ;  Laser-mode enable, boolean
# # $33=5000.0  ;  Spindle PWM frequency
# # $34=0.0  ;  Spindle off Value
# # $35=0.0  ;  Spindle min value
# # $36=100.0  ;  Spindle max value
# # $37=0  ;  Stepper deenergize mask
# # $39=1  ;  Enable printable realtime command characters, boolean
# # $40=0  ;  Apply soft limits for jog commands, boolean
# # $43=1  ;  Homing passes
# # $44=4  ;  Homing cycle 1
# # $45=3  ;  Homing cycle 2
# # $46=0  ;  Homing cycle 3
# # $62=0  ;  Sleep Enable
# # $63=2  ;  Feed Hold Actions
# # $64=0  ;  Force Init Alarm
# # $65=0  ;  Require homing sequence to be executed at startup
# # $70=7  ;  Network Services
# # $73=1  ;  Wifi Mode
# # $74=  ;  Wifi network SSID
# # $75=  ;  Wifi network PSK
# # $100=198.109  ;  X-axis steps per millimeter
# # $101=198.109  ;  Y-axis steps per millimeter
# # $102=199.100  ;  Z-axis steps per millimeter
# # $110=3500.000  ;  X-axis maximum rate, mm/min
# # $111=3500.000  ;  Y-axis maximum rate, mm/min
# # $112=1000.000  ;  Z-axis maximum rate, mm/min
# # $120=350.000  ;  X-axis acceleration, mm/sec^2
# # $121=350.000  ;  Y-axis acceleration, mm/sec^2
# # $122=350.000  ;  Z-axis acceleration, mm/sec^2
# # $130=1180.000  ;  X-axis maximum travel, millimeters
# # $131=1270.000  ;  Y-axis maximum travel, millimeters
# # $132=90.000  ;  Z-axis maximum travel, millimeters
# # $300=Grbl  ;  Hostname
# # $302=192.168.5.1  ;  IP Address
# # $303=192.168.5.1  ;  Gateway
# # $304=255.255.255.0  ;  Netmask
# # $305=23  ;  Telnet Port
# # $306=80  ;  HTTP Port
# # $307=81  ;  Websocket Port
# # $341=0  ;  Tool Change Mode
# # $342=30.0  ;  Tool Change probing distance
# # $343=25.0  ;  Tool Change Locate Feed rate
# # $344=200.0  ;  Tool Change Search Seek rate
# # $345=200.0  ;  Tool Change Probe Pull Off rate
# # $346=1  ;  Restore position after M6 as boolean
# # $370=0  ;  Invert I/O Port Inputs (mask)
# # $384=0  ;  Disable G92 Persistence
# # $396=30  ;  WebUI timeout in minutes
# # $397=0  ;  WebUI auto report interval in milliseconds
# # $398=35  ;  Planner buffer blocks
# # $481=0  ;  Autoreport interval in ms
# # $I=leadmachine1515
