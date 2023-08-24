% WW_format_rectangular_script.m
% Anthony Fouad
% modified by Samuel Freitas
% Fang-Yen Lab and Sutphin Lab
% December 11, 2018
% 
% Formats a script to cycle through each plate on a WormWatcher tray,
% performing an imaging experiment on each.
%
% NOTE: Wells that do not contain any plate will automatically be skipped.
%
% NOTE: The functionality in this script will eventually be replaced by a
%       later version of WWConfig.exe

%% OPTION 1: Manually specify rows and columns
rows = 9;
cols = 9*ones(1,rows);

%% OPTION 2: Automatically detect the rows and columns in the WW User Parameters

% % File location
% pww = 'C:\WormWatcher';
% fuser = 'params_PlateTray_User.txt';
% fname = fullfile(pww,fuser);
% 
% % verify file exists
% if ~exist(fname,'file')
%     beep;
%     errordlg('Failed to detect PlateTray User Parameters file. Please re-install WormWatcher in the default directory','Failed to find file');
%     return;
% end
% 
% % Load the user parameters file
% data = dlmread(fname);
% rows = data(1,1);
% cols = data(2,:);

%% Format script - copy output to params_PlateTray_ActionScript.txt
% 
% clc;
% for r = 0:rows - 1
%     for c = 0:cols(r+1) - 1
%         fprintf('%s\t%d\t%d\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n','action',r,c,'WWEXP',0,0,0,0,0,0);
%     end
% end
% 
% fprintf('\n\n');

% copy output to params_PlateTray_ActionScript.txt
%% Format script for PlateNames

% clc;
% for r = 0:rows - 1
%     for c = 0:cols(r+1) - 1
%         fprintf('%d\t%d\t%s%d%d\t%s\t\n',r,c,'BoxCalibration',r,c,'C:\WormWatcher\BoxCalibration');
%     end
% end
% 
% fprintf('\n\n');

%% Format script for PlateTray Coordinates 

clc;

% fprintf('plate_index\trow\tcolumn\tx_pos\ty_pos\tz_pos\n') %%% default
fprintf('plate_index\trow\tcolumn\tplate_name\texperiment_name\tlifespan\tfluorescence\n') %name and opts
xo = -31;
yo = -29;
zo = -24;

dx = -153.2;
dy = -116;

i = 0;
j = 0;

counter = 0;

for r = 0:rows - 1
    i = 0;
    for c = 0:cols(r+1) - 1
%         fprintf('%d\t%d\t%d\t%5.2f\t%5.2f\t%5.2f\t\n',counter,r,c,xo+(dx*i),yo+(dy*j),zo); %%%default 
%         fprintf('%5.2f\t%5.2f\t%5.2f\t\n',xo+(dx*i),yo+(dy*j),zo); %%%%%%%%%%%just xyz 
        fprintf('%d\t%d\t%d\tNONE\tNONE\t%d\t%d\n',counter,r,c,1,1); %%% name and opts 
        counter = counter + 1;
        i = i +1;
    end
    j = j+1;
end

fprintf('\n\n');
