import numpy as np
import pandas as pd
import os

# this is a script that will overwrite settings and replace them with cvs's
# uncomment to run a specific section
# this WILL REWRITE ALL SETTINGS

path_to_settings_folder = os.path.join(os.getcwd(), "settings")

rows = 9  # rows and columns of the robot
cols = 8

xo = -108.85 + 63.1 + 5.0 - 5.0 #  approximate intial starting points -------------- xo to center of plate + xo from center of plate to homing swtich + homing switch offset 
yo = -269.66 + 148.16 + 5. + 3. + 1.9#  approximate intial starting points ------- same as above with an added 3 for calibration(???)
zo = -107  #  approximate intial starting points --------------------------- measured height for widefield imaging

xo = round(xo,4)
yo = round(yo,4)
zo = round(zo,4)

print('START:','x',xo,'y',yo,'z',zo)

dx = -153.2  # distace between plates
dy = -115  # used to be 116


# # ##############################################
## this is for the settings Wormotel positions 4 at a time measurements

# header = ['CentroidsX_mm','CentroidsY_mm','x_relative_pos_mm','y_relative_pos_mm','calib_x_pos_mm','calib_y_pos_mm','calib_z_pos_mm','y_offset_to_fluor_mm','name']
# df = pd.DataFrame(columns = header)
# df2 = df.copy(deep=True)

# calib_x_pos_mm = -43.75
# calib_y_pos_mm = -28.5
# calib_z_pos_mm = -89.0
# y_offset_to_fluor_mm = 87.75

# names = []
# for WM_row_names in range(1,21):
#     for WM_col_names in ['A','B','C','D','E','F','G','H','I','J','K','L']:
#         names.append(str(WM_col_names)+str(WM_row_names))
         
# i = 0
# j = 0
# counter = 0

# # WM_xo = 0.0
# # WM_yo = -42
# # WM_zo = -83.5

# WM_xextent = -84.143
# WM_yextent = -49.227

# WM_rows = 12
# WM_cols = 20

# x_linspace = np.linspace(0,WM_xextent,WM_cols)
# y_linspace = np.linspace(WM_yextent,0,WM_rows)

# xv,yv = np.meshgrid(x_linspace,y_linspace)

# for c in range(WM_cols):
#     i = 0
#     for r in range(WM_rows):

#         BLx = round(xv[r][c],4)
#         BLy = round(yv[r][c],4)

#         centered_x = BLx - (WM_xextent/2)
#         centered_y = BLy - (WM_yextent/2)

#         df.loc[counter] = [BLx,BLy,centered_x,centered_y,0,0,0,0,names[counter]] #just xyz
#         counter = counter + 1
#         i = i +1
#     j = j+1

# df.iloc[0,4] = calib_x_pos_mm
# df.iloc[0,5] = calib_y_pos_mm
# df.iloc[0,6] = calib_z_pos_mm
# df.iloc[0,7] = y_offset_to_fluor_mm

# row_names = ['A','B','C','D','E','F','G','H','I','J','K','L']
# col_names = list(range(1,21))

# names_of_wells = list(df['name'].values)
# x_positions = list(df['x_relative_pos_mm'].values)
# y_positions = list(df['y_relative_pos_mm'].values)
# x_centroids = list(df['CentroidsX_mm'].values)
# y_centroids = list(df['CentroidsY_mm'].values)
# names = []
# counter = 0
# for i in range(0,len(row_names),2):
#     for j in range(0,len(col_names),2):

#         first_well = str(row_names[i]) + str(col_names[j]) 
#         fourth_well = str(row_names[i+1]) + str(col_names[j+1]) 

#         export_name = str(row_names[i]) + str(col_names[j]) + '_' + str(row_names[i]) + str(col_names[j+1]) + '_' + str(row_names[i+1]) + str(col_names[j]) + '_' + str(row_names[i+1]) + str(col_names[j+1])
#         names.append(export_name)

#         idx_1 = names_of_wells.index(first_well)
#         idx_4 = names_of_wells.index(fourth_well)

#         x_pos = (x_positions[idx_1] + x_positions[idx_4])/2
#         y_pos = (y_positions[idx_1] + y_positions[idx_4])/2
#         x_cen = (x_centroids[idx_1] + x_centroids[idx_4])/2
#         y_cen = (y_centroids[idx_1] + y_centroids[idx_4])/2

#         # df2.iloc[counter,0:4] = []
#         df2.loc[counter] = [x_cen,y_cen,x_pos,y_pos,0,0,0,0,export_name] #just xyz
#         # print(export_name, x_pos,y_pos)
#         counter = counter + 1

# df2.iloc[0,4] = calib_x_pos_mm
# df2.iloc[0,5] = calib_y_pos_mm
# df2.iloc[0,6] = calib_z_pos_mm
# df2.iloc[0,7] = y_offset_to_fluor_mm
# df2.to_csv(os.path.join(path_to_settings_folder,'settings_WM_4pair_positions.csv'),index= False)


# # # ##############################################
# ## this is for the settings Wormotel positions 

# header = ['CentroidsX_mm','CentroidsY_mm','x_relative_pos_mm','y_relative_pos_mm','calib_x_pos_mm','calib_y_pos_mm','calib_z_pos_mm','y_offset_to_fluor_mm','name']
# df = pd.DataFrame(columns = header)

# calib_x_pos_mm = -43.75
# calib_y_pos_mm = -28.5
# calib_z_pos_mm = -89.0
# y_offset_to_fluor_mm = 87.75

# names = []
# for WM_row_names in range(1,21):
#     for WM_col_names in ['A','B','C','D','E','F','G','H','I','J','K','L']:
#         names.append(str(WM_col_names)+str(WM_row_names))
         
# i = 0
# j = 0
# counter = 0

# # WM_xo = 0.0
# # WM_yo = -42
# # WM_zo = -83.5

# WM_xextent = -84.143
# WM_yextent = -49.227

# WM_rows = 12
# WM_cols = 20

# x_linspace = np.linspace(0,WM_xextent,WM_cols)
# y_linspace = np.linspace(WM_yextent,0,WM_rows)

# xv,yv = np.meshgrid(x_linspace,y_linspace)

# for c in range(WM_cols):
#     i = 0
#     for r in range(WM_rows):

#         BLx = round(xv[r][c],4)
#         BLy = round(yv[r][c],4)

#         centered_x = BLx - (WM_xextent/2)
#         centered_y = BLy - (WM_yextent/2)

#         df.loc[counter] = [BLx,BLy,centered_x,centered_y,0,0,0,0,names[counter]] #just xyz
#         counter = counter + 1
#         i = i +1
#     j = j+1

# df.iloc[0,4] = calib_x_pos_mm
# df.iloc[0,5] = calib_y_pos_mm
# df.iloc[0,6] = calib_z_pos_mm
# df.iloc[0,7] = y_offset_to_fluor_mm
# df.to_csv(os.path.join(path_to_settings_folder,'settings_WM_positions.csv'),index= False)

####### this is just a little check script to make sure that the output is correct (and flipped that axis')
# # # import matplotlib.pyplot as plt
# # # x,y = df.x_relative_pos_mm,df.y_relative_pos_mm
# # # plt.scatter(-x,-y)
# # # for i,txt in enumerate(names):
# # #     plt.annotate(txt,(-x[int(i)],-y[int(i)]))
# # # plt.show()




# # # ##############################################
# ## this is for the settings terasaki positions 
# header = ['CentroidsX_mm','CentroidsY_mm','x_relative_pos_mm','y_relative_pos_mm','calib_x_pos_mm','calib_y_pos_mm','calib_z_pos_mm','y_offset_to_fluor_mm','name']
# df = pd.DataFrame(columns = header)

# calib_x_pos_mm = -43.75
# calib_y_pos_mm = -28.5
# calib_z_pos_mm = -89.0
# y_offset_to_fluor_mm = 87.75

# names = ['h1','g1','f1','e1','d1','c1','b1','a1',
#          'h2','g2','f2','e2','d2','c2','b2','a2',
#          'h3','g3','f3','e3','d3','c3','b3','a3',
#          'h4','g4','f4','e4','d4','c4','b4','a4',
#          'h5','g5','f5','e5','d5','c5','b5','a5',
#          'h6','g6','f6','e6','d6','c6','b6','a6',
#          'h7','g7','f7','e7','d7','c7','b7','a7',
#          'h8','g8','f8','e8','d8','c8','b8','a8',
#          'h9','g9','f9','e9','d9','c9','b9','a9',
#          'h10','g10','f10','e10','d10','c10','b10','a10',
#          'h11','g11','f11','e11','d11','c11','b11','a11',
#          'h12','g12','f12','e12','d12','c12','b12','a12']

# i = 0
# j = 0
# counter = 0

# terasaki_xextent = -69.695
# terasaki_yextent = -41.896 #[69.695,41.845] 

# terasaki_rows = 8
# terasaki_cols = 12

# x_linspace = np.linspace(0,terasaki_xextent,terasaki_cols)
# y_linspace = np.linspace(terasaki_yextent,0,terasaki_rows)

# xv,yv = np.meshgrid(x_linspace,y_linspace)

# for c in range(terasaki_cols):
#     i = 0
#     for r in range(terasaki_rows):

#         BLx = round(xv[r][c],4)
#         BLy = round(yv[r][c],4)

#         centered_x = BLx - (terasaki_xextent/2)
#         centered_y = BLy - (terasaki_yextent/2)

#         centered_x = round(centered_x,5)
#         centered_y = round(centered_y,5)

#         df.loc[counter] = [BLx,BLy,centered_x,centered_y,0,0,0,0,names[counter]] #just xyz
#         counter = counter + 1
#         i = i +1
#     j = j+1

# df.iloc[0,4] = calib_x_pos_mm
# df.iloc[0,5] = calib_y_pos_mm
# df.iloc[0,6] = calib_z_pos_mm
# df.iloc[0,7] = y_offset_to_fluor_mm
# df.to_csv(os.path.join(path_to_settings_folder,'settings_terasaki_positions.csv'),index= False)


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
#         df.loc[counter] = [counter,r,c,round(xo+(dx*i),4),round(yo+(dy*j),4),zo] #just xyz
#         counter = counter + 1
#         i = i +1
#     j = j+1

# df.to_csv(os.path.join(path_to_settings_folder,'settings_plate_positions.csv'),index= False)




# # #############################################
# # ## this is for the settings plate names and opts
# header = ['plate_index','row','column','plate_name','experiment_name','lifespan','fluorescence','fluorescence_times','fluorescence_UV','fluorescence_BLUE','fluorescence_GREEN','fluorescence_RED']
# df = pd.DataFrame(columns = header)
# i = 0
# j = 0
# counter = 0

# for r in range(rows):
#     i = 0
#     for c in range(cols):
#         df.loc[counter] = [counter ,r , c ,'NONE','NONE','1','1','100','1','1','5','5']
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

#############################################
## this is for the cameras setting machines

header = ['Explanation','widefield','fluorescence']
df = pd.DataFrame(columns = header)

df.loc[0] = ['camera_port','0','1']
df.loc[1] = ['pixel_width','5472','2560']
df.loc[2] = ['pixel_height','3640','2560']
df.loc[3] = ['max_framerate','6','10']
df.loc[4] = ['time_between_images_seconds','5','0.01']
df.loc[5] = ['time_of_single_burst_seconds','60','0']
df.loc[6] = ['number_of_images_per_burst','12','3']
df.loc[7] = ['img_file_format','png','png']
df.loc[8] = ['img_pixel_depth','8','8']
df.loc[9] = ['img_color','0','1']
df.loc[10] = ['pixel_binning','1','1']
df.loc[11] = ['HDR','0','1']

df.to_csv(os.path.join(path_to_settings_folder,'settings_cameras.csv'),index= False)

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
