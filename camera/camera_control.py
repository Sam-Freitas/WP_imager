import cv2, os, tqdm, glob, time, datetime
import matplotlib.pyplot as plt
from numpy import zeros, logical_and, logical_or, logical_xor

# cv2 imshow but resized to a size that can fit in a normal monitor size
def imshow_resize(frame_name = "img", frame = 0, resize_size = [640,480], default_ratio = 1.3333, 
                  always_on_top = True, use_waitkey = True, move_to = [1920,1]):

    frame = cv2.resize(frame, dsize=resize_size)
    cv2.imshow(frame_name, frame)
    if always_on_top:
        cv2.setWindowProperty(frame_name, cv2.WND_PROP_TOPMOST, 1)
    if use_waitkey:
        cv2.waitKey(1)
    cv2.moveWindow(frame_name,move_to[0]-resize_size[0],move_to[1])
    return True

# this deletes all the dir contents, recursive or not 
def del_dir_contents(path_to_dir, recursive = False): 
    if recursive:
        files = glob.glob(os.path.join(path_to_dir,'**/*'), recursive=recursive)
        for f in files:
            if not os.path.isdir(f):
                os.remove(f)
    else: # default
        files = glob.glob(os.path.join(path_to_dir,'*'))
        for f in files:
            os.remove(f)

# this is a buffer that continues to capture images until the the specified time has elapsed 
# this is because if you use time.sleep() the image taken is buffered and not at the actual elapsed time
def capture_images_for_time(cap,N, show_images = False, move_to = [100,100], resize_size = [640,480], start_time = None):
    if start_time == None:
        start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break
        if show_images:
            imshow_resize("stream", frame, resize_size=resize_size, move_to=move_to)
        current_time = time.time()
        if start_time < current_time - N + 0.1: # dont know why the 0.1 is necessary but it works much better with it
            break 

# this captures the first N images to clear the pipeline (sometime just black images)
def clear_camera_image_buffer(cap,N=2):
    for i in range(N):
        ret, frame = cap.read()

def capture_single_image_wait_N_seconds(camera_settings,timestart = None, excitation_amount = 9, plate_parameters = None, testing = False, output_dir = None):

    todays_date = datetime.date.today().strftime("%Y-%m-%d")

    if timestart == None:
        timestart = time.time()

    if output_dir == None:
        output_dir = os.path.join(os.getcwd(),'output',plate_parameters['experiment_name'],plate_parameters['plate_name'],todays_date)
    else:
        output_dir = os.path.join(output_dir,plate_parameters['experiment_name'],plate_parameters['plate_name'],todays_date)
    
    os.makedirs(output_dir,exist_ok=True)
    if testing:
        del_dir_contents(output_dir)

    camera_id = camera_settings['widefield'][0]
    cam_width = float(camera_settings['widefield'][1])
    cam_height = float(camera_settings['widefield'][2])
    cam_framerate = camera_settings['widefield'][3]
    time_between_images_seconds = float(excitation_amount)
    time_of_single_burst_seconds = camera_settings['widefield'][5]
    number_of_images_per_burst = 1
    img_file_format = camera_settings['widefield'][7]
    img_pixel_depth = camera_settings['widefield'][8]

    # Define the text and font settings
    text = plate_parameters['experiment_name'] + '--' + plate_parameters['plate_name'] + '--' + todays_date
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 10.0
    font_color1 = (255, 255, 255)  # white color
    thickness1 = 15
    font_color2 = (0, 0, 0)  # black color for outline
    thickness2 = 50

    # Calculate the position for placing the text
    text_size = cv2.getTextSize(text, font, font_scale, thickness1)[0]
    text_x = (cam_width - text_size[0]) // 2  # Center horizontally
    text_y = 250  # 250 pixels from the top
    text_x2 = text_x-200
    text_y2 = 500

    # time_between_images_seconds = 0 # this is just for testing 
    img_file_format = 'png' # slow and lossless but smaller 

    # Open the camera0
    cap = cv2.VideoCapture(int(camera_id))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,int(cam_width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,int(cam_height))
    cap.set(cv2.CAP_PROP_FPS,int(cam_framerate))

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        exit()

    clear_camera_image_buffer(cap)

    num_images = int(number_of_images_per_burst)
    # Capture a series of images
    for i in tqdm.tqdm(range(num_images)):
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break
        
        current_time_for_filename = datetime.datetime.now().strftime("%Y-%m-%d (%H-%M-%S-%f)")
        image_name = current_time_for_filename + '.' + img_file_format
        image_filename = os.path.join(output_dir, image_name)

        cv2.imwrite(image_filename, frame[:,:,-1])
        # print(f"\nCaptured image {i+1}/{num_images}")

        # Put the text on the image
        cv2.putText(frame, text, (int(text_x), int(text_y)), font, font_scale, font_color2, thickness2) # black 
        cv2.putText(frame, text, (int(text_x), int(text_y)), font, font_scale, font_color1, thickness1) # white
        cv2.putText(frame, current_time_for_filename, (int(text_x2), int(text_y2)), font, font_scale, font_color2, thickness2) # black 
        cv2.putText(frame, current_time_for_filename, (int(text_x2), int(text_y2)), font, font_scale, font_color1, thickness1) # white

        imshow_resize("img", frame, resize_size=[640,480])

        capture_images_for_time(cap,time_between_images_seconds, show_images=True,move_to = [1920,520], start_time = timestart)        # time.sleep(1)

    # Release the camera
    # cv2.destroyAllWindows()
    cap.release()

def simple_capture_data(camera_settings, plate_parameters = None, testing = False, output_dir = None):

    todays_date = datetime.date.today().strftime("%Y-%m-%d")

    if output_dir == None:
        output_dir = os.path.join(os.getcwd(),'output',plate_parameters['experiment_name'],plate_parameters['plate_name'],todays_date)
    else:
        output_dir = os.path.join(output_dir,plate_parameters['experiment_name'],plate_parameters['plate_name'],todays_date)
    
    os.makedirs(output_dir,exist_ok=True)
    if testing:
        del_dir_contents(output_dir, recursive=testing)

    camera_id = camera_settings['widefield'][0]
    cam_width = float(camera_settings['widefield'][1])
    cam_height = float(camera_settings['widefield'][2])
    cam_framerate = camera_settings['widefield'][3]
    time_between_images_seconds = float(camera_settings['widefield'][4])
    time_of_single_burst_seconds = camera_settings['widefield'][5]
    number_of_images_per_burst = float(camera_settings['widefield'][6])
    img_file_format = camera_settings['widefield'][7]
    img_pixel_depth = camera_settings['widefield'][8]

    # Define the text and font settings
    text = plate_parameters['experiment_name'] + '--' + plate_parameters['plate_name'] + '--' + todays_date
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 10.0
    font_color1 = (255, 255, 255)  # white color
    thickness1 = 15
    font_color2 = (0, 0, 0)  # black color for outline
    thickness2 = 50

    # Calculate the position for placing the text
    text_size = cv2.getTextSize(text, font, font_scale, thickness1)[0]
    text_x = (cam_width - text_size[0]) // 2  # Center horizontally
    text_y = 250  # 250 pixels from the top
    text_x2 = text_x-200
    text_y2 = 500

    # time_between_images_seconds = 2 # this is just for testing 
    img_file_format = 'png' # slow and lossless but smaller 
    # # img_file_format = 'jpg' # fast but lossy small files
    # # img_file_format = 'bmp' # fastest and lossess huge files

    # Open the camera0
    cap = cv2.VideoCapture(int(camera_id))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,int(cam_width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,int(cam_height))
    cap.set(cv2.CAP_PROP_FPS,int(cam_framerate))

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        exit()

    clear_camera_image_buffer(cap)

    num_images = int(number_of_images_per_burst)
    # Capture a series of images
    for i in tqdm.tqdm(range(num_images)):
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break
        
        current_time_for_filename = datetime.datetime.now().strftime("%Y-%m-%d (%H-%M-%S-%f)")
        image_name = current_time_for_filename + '.' + img_file_format
        image_filename = os.path.join(output_dir, image_name)

        cv2.imwrite(image_filename, frame[:,:,-1])#, [int(cv2.IMWRITE_PNG_COMPRESSION), 5])
        # print(f"\nCaptured image {i+1}/{num_images}")

        # Put the text on the image white with a black background
        cv2.putText(frame, text, (int(text_x), int(text_y)), font, font_scale, font_color2, thickness2) # black 
        cv2.putText(frame, text, (int(text_x), int(text_y)), font, font_scale, font_color1, thickness1) # white
        cv2.putText(frame, current_time_for_filename, (int(text_x2), int(text_y2)), font, font_scale, font_color2, thickness2) # black 
        cv2.putText(frame, current_time_for_filename, (int(text_x2), int(text_y2)), font, font_scale, font_color1, thickness1) # white

        imshow_resize("img", frame, resize_size=[640,480])

        capture_images_for_time(cap,time_between_images_seconds, show_images=True,move_to = [1920,520], start_time = start_time)
        # time.sleep(1)

    # Release the camera
    # cv2.destroyAllWindows()
    cap.release()

def simple_capture_data_single_image(camera_settings, plate_parameters = None, testing = False, output_dir = None, image_file_format = 'png'):

    todays_date = datetime.date.today().strftime("%Y-%m-%d")

    if output_dir == None:
        output_dir = os.path.join(os.getcwd(),'output','calibration')
    else:
        output_dir = os.path.join(output_dir,'calibration')
    
    os.makedirs(output_dir,exist_ok=True)
    if testing:
        del_dir_contents(output_dir, recursive=testing)

    camera_id = camera_settings['widefield'][0]
    cam_width = float(camera_settings['widefield'][1])
    cam_height = float(camera_settings['widefield'][2])
    cam_framerate = camera_settings['widefield'][3]
    time_between_images_seconds = float(camera_settings['widefield'][4])
    time_of_single_burst_seconds = camera_settings['widefield'][5]
    number_of_images_per_burst = float(camera_settings['widefield'][6])
    img_file_format = camera_settings['widefield'][7]
    img_pixel_depth = camera_settings['widefield'][8]

    # Define the text and font settings
    text = plate_parameters['experiment_name'] + '--' + plate_parameters['plate_name'] + '--' + todays_date
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 10.0
    font_color1 = (255, 255, 255)  # white color
    thickness1 = 15
    font_color2 = (0, 0, 0)  # black color for outline
    thickness2 = 50

    # Calculate the position for placing the text
    text_size = cv2.getTextSize(text, font, font_scale, thickness1)[0]
    text_x = (cam_width - text_size[0]) // 2  # Center horizontally
    text_y = 250  # 250 pixels from the top
    text_x2 = text_x-200
    text_y2 = 500

    # time_between_images_seconds = 2 # this is just for testing 
    if img_file_format == 'png':
        img_file_format = 'png' # slow and lossless but smaller 
    else:
        img_file_format = image_file_format # fast but lossy small files
    # # img_file_format = 'bmp' # fastest and lossess huge files

    # Open the camera0
    cap = cv2.VideoCapture(int(camera_id))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,int(cam_width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,int(cam_height))
    cap.set(cv2.CAP_PROP_FPS,int(cam_framerate))

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        exit()

    clear_camera_image_buffer(cap)

    num_images = 1
    # Capture a series of images
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
    
    current_time_for_filename = datetime.datetime.now().strftime("%Y-%m-%d (%H-%M-%S-%f)")
    image_name = current_time_for_filename + '.' + img_file_format
    image_filename = os.path.join(output_dir, image_name)

    cv2.imwrite(image_filename, frame[:,:,-1])#, [int(cv2.IMWRITE_PNG_COMPRESSION), 5])
    # print(f"\nCaptured image {i+1}/{num_images}")

    # Put the text on the image white with a black background
    cv2.putText(frame, text, (int(text_x), int(text_y)), font, font_scale, font_color2, thickness2) # black 
    cv2.putText(frame, text, (int(text_x), int(text_y)), font, font_scale, font_color1, thickness1) # white
    cv2.putText(frame, current_time_for_filename, (int(text_x2), int(text_y2)), font, font_scale, font_color2, thickness2) # black 
    cv2.putText(frame, current_time_for_filename, (int(text_x2), int(text_y2)), font, font_scale, font_color1, thickness1) # white

    imshow_resize("img", frame, resize_size=[640,480])

    # capture_images_for_time(cap,time_between_images_seconds, show_images=True,move_to = [1920,520], start_time = start_time)
    # time.sleep(1)

    # Release the camera
    # cv2.destroyAllWindows()
    cap.release()

    return image_filename

def simple_capture_data_fluor(camera_settings, plate_parameters = None, testing = False, output_dir = None):

    todays_date = datetime.date.today().strftime("%Y-%m-%d")

    if output_dir == None:
        output_dir = os.path.join(os.getcwd(),'output',plate_parameters['experiment_name'],plate_parameters['plate_name'],todays_date)
    else:
        output_dir = os.path.join(output_dir,plate_parameters['experiment_name'],plate_parameters['plate_name'],todays_date,'fluorescent_data')
    
    os.makedirs(output_dir,exist_ok=True)
    if testing:
        del_dir_contents(output_dir)

    camera_id = camera_settings['fluorescence'][0]
    camera_id = 0 #####################################################################################S
    cam_width = float(camera_settings['fluorescence'][1])
    cam_height = float(camera_settings['fluorescence'][2])
    cam_framerate = camera_settings['fluorescence'][3]
    time_between_images_seconds = float(camera_settings['fluorescence'][4])
    time_of_single_burst_seconds = camera_settings['fluorescence'][5]
    number_of_images_per_burst = float(camera_settings['fluorescence'][6])
    img_file_format = camera_settings['fluorescence'][7]
    img_pixel_depth = camera_settings['fluorescence'][8]
    img_color = camera_settings['fluorescence'][9]

    # Define the text and font settings
    text = plate_parameters['experiment_name'] + '--' + plate_parameters['plate_name'] + '--' + todays_date
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4
    font_color1 = (255, 255, 255)  # white color
    thickness1 = 15
    font_color2 = (0, 0, 0)  # black color for outline
    thickness2 = 50

    # Calculate the position for placing the text
    text_size = cv2.getTextSize(text, font, font_scale, thickness1)[0]
    text_x = (cam_width - text_size[0]) // 2  # Center horizontally
    text_y = 250  # 250 pixels from the top
    text_x2 = text_x-200
    text_y2 = 500

    time_between_images_seconds = 1 # this is just for testing 
    img_file_format = 'png' # slow and lossless but smaller 
    # # img_file_format = 'jpg' # fast but lossy small files
    # # img_file_format = 'bmp' # fastest and lossess huge files

    # Open the camera0
    cap = cv2.VideoCapture(int(camera_id))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,int(cam_width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,int(cam_height))
    cap.set(cv2.CAP_PROP_FPS,int(cam_framerate))

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        exit()

    clear_camera_image_buffer(cap, N = 3)

    num_images = int(number_of_images_per_burst)
    # Capture a series of images
    for i in tqdm.tqdm(range(num_images)):
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break
        
        # current_time_for_filename = datetime.datetime.now().strftime("%Y-%m-%d (%H-%M-%S-%f)")
        image_subtype = plate_parameters['well_name'] + '_00' + str(i+1)
        image_name = plate_parameters['well_name'] + '_00' + str(i+1) + '_' + '.' + img_file_format#current_time_for_filename + '.' + img_file_format
        image_filename = os.path.join(output_dir, image_name)

        cv2.imwrite(image_filename, frame[:,:,-1])#, [int(cv2.IMWRITE_PNG_COMPRESSION), 5])
        # print(f"\nCaptured image {i+1}/{num_images}")

        # Put the text on the image white with a black background
        cv2.putText(frame, text, (int(text_x), int(text_y)), font, font_scale, font_color2, thickness2) # black 
        cv2.putText(frame, text, (int(text_x), int(text_y)), font, font_scale, font_color1, thickness1) # white
        cv2.putText(frame, image_subtype, (int(text_x2), int(text_y2)), font, font_scale, font_color2, thickness2) # black 
        cv2.putText(frame, image_subtype, (int(text_x2), int(text_y2)), font, font_scale, font_color1, thickness1) # white

        imshow_resize("img", frame, resize_size=[640,480])

        capture_images_for_time(cap,time_between_images_seconds, show_images=True,move_to = [1920,520], start_time = start_time)
        # time.sleep(1)

    # Release the camera
    # cv2.destroyAllWindows()
    cap.release()

def simple_capture_data_fluor_single_image(camera_settings, plate_parameters = None, testing = False, output_dir = None, image_file_format = 'png'):

    todays_date = datetime.date.today().strftime("%Y-%m-%d")

    if output_dir == None:
        output_dir = os.path.join(os.getcwd(),'output','calibration')
    else:
        output_dir = os.path.join(output_dir,'calibration')
    
    os.makedirs(output_dir,exist_ok=True)
    if testing:
        del_dir_contents(output_dir, recursive=testing)

    camera_id = camera_settings['fluorescence'][0]
    camera_id = 0 #####################################################################################S
    cam_width = float(camera_settings['fluorescence'][1])
    cam_height = float(camera_settings['fluorescence'][2])
    cam_framerate = camera_settings['fluorescence'][3]
    time_between_images_seconds = float(camera_settings['fluorescence'][4])
    time_of_single_burst_seconds = camera_settings['fluorescence'][5]
    number_of_images_per_burst = float(camera_settings['fluorescence'][6])
    img_file_format = camera_settings['fluorescence'][7]
    img_pixel_depth = camera_settings['fluorescence'][8]
    img_color = camera_settings['fluorescence'][9]

    # Define the text and font settings
    text = plate_parameters['experiment_name'] + '--' + plate_parameters['plate_name'] + '--' + todays_date
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4
    font_color1 = (255, 255, 255)  # white color
    thickness1 = 15
    font_color2 = (0, 0, 0)  # black color for outline
    thickness2 = 50

    # Calculate the position for placing the text
    text_size = cv2.getTextSize(text, font, font_scale, thickness1)[0]
    text_x = (cam_width - text_size[0]) // 2  # Center horizontally
    text_y = 250  # 250 pixels from the top
    text_x2 = text_x-200
    text_y2 = 500

    # time_between_images_seconds = 2 # this is just for testing 
    img_file_format = 'png' # slow and lossless but smaller 
    # # img_file_format = 'jpg' # fast but lossy small files
    # # img_file_format = 'bmp' # fastest and lossess huge files

    # Open the camera0
    cap = cv2.VideoCapture(int(camera_id))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,int(cam_width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,int(cam_height))
    cap.set(cv2.CAP_PROP_FPS,int(cam_framerate))

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        exit()

    clear_camera_image_buffer(cap)

    num_images = 1
    # Capture a series of images
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
    
    current_time_for_filename = datetime.datetime.now().strftime("%Y-%m-%d (%H-%M-%S-%f)")
    image_name = current_time_for_filename + '.' + img_file_format
    image_filename = os.path.join(output_dir, image_name)

    cv2.imwrite(image_filename, frame[:,:,-1])#, [int(cv2.IMWRITE_PNG_COMPRESSION), 5])
    # print(f"\nCaptured image {i+1}/{num_images}")

    # Put the text on the image white with a black background
    cv2.putText(frame, text, (int(text_x), int(text_y)), font, font_scale, font_color2, thickness2) # black 
    cv2.putText(frame, text, (int(text_x), int(text_y)), font, font_scale, font_color1, thickness1) # white
    cv2.putText(frame, current_time_for_filename, (int(text_x2), int(text_y2)), font, font_scale, font_color2, thickness2) # black 
    cv2.putText(frame, current_time_for_filename, (int(text_x2), int(text_y2)), font, font_scale, font_color1, thickness1) # white

    imshow_resize("img", frame, resize_size=[640,480])

    # capture_images_for_time(cap,time_between_images_seconds, show_images=True,move_to = [1920,520], start_time = start_time)
    # time.sleep(1)

    # Release the camera
    # cv2.destroyAllWindows()
    cap.release()

    return image_filename


def capture_fluor_img_return_img(camera_settings, cap = None, return_cap = False):

    if cap is None:
        cap_release = True

        camera_id = camera_settings['fluorescence'][0]
        camera_id = 0 #####################################################################################S
        cam_width = float(camera_settings['fluorescence'][1])
        cam_height = float(camera_settings['fluorescence'][2])
        cam_framerate = camera_settings['fluorescence'][3]
        time_between_images_seconds = float(camera_settings['fluorescence'][4])
        time_of_single_burst_seconds = camera_settings['fluorescence'][5]
        number_of_images_per_burst = float(camera_settings['fluorescence'][6])
        img_file_format = camera_settings['fluorescence'][7]
        img_pixel_depth = camera_settings['fluorescence'][8]
        img_color = camera_settings['fluorescence'][9]

        # Open the camera0
        cap = cv2.VideoCapture(int(camera_id))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,int(cam_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,int(cam_height))
        cap.set(cv2.CAP_PROP_FPS,int(cam_framerate))

        if not cap.isOpened():
            print("Error: Unable to open camera.")
            exit()

        if return_cap:
            cap_release = False
    else:
        cap_release = False

    clear_camera_image_buffer(cap, N = 10)
    num_images = 1
    # Capture a series of images
    ret, frame = cap.read()
    frame = frame[:,:,-1]
    if not ret:
        print("Error: Unable to capture frame.")

    if cap_release:
        cap.release()

    if return_cap:
        return frame, cap
    else:
        return frame 

def crop_center_numpy_return(img_array, n, center=None):

    # Get the dimensions of the image
    height, width = img_array.shape

    # If center is provided, calculate cropping coordinates
    if center is not None:
        # center = np.array(center)
        left = center[1] - n // 2
        top = center[0] - n // 2
        right = center[1] + n // 2
        bottom = center[0] + n // 2

    else:
        # Calculate the cropping coordinates for geometric center
        left = (width - n) // 2
        top = (height - n) // 2
        right = (width + n) // 2
        bottom = (height + n) // 2

    left, top, right, bottom = int(left), int(top), int(right), int(bottom)

    # Crop the image using NumPy array slicing
    cropped_array = img_array[top:bottom, left:right]

    return cropped_array

def put_frame_in_large_img(extent_y, extent_x, pixels_per_mm, FOV, delta_x, delta_y, i, img_data, row, col):
    row_start = int(row * pixels_per_mm * abs(delta_x))
    row_end = row_start + int(pixels_per_mm * FOV)
    col_start = int(col * pixels_per_mm * abs(delta_y))
    col_end = col_start + int(pixels_per_mm * FOV)

    temp_large_img = zeros((int(extent_y * pixels_per_mm), int(extent_x * pixels_per_mm)))
    temp_large_img[row_start:row_end, col_start:col_end] = img_data

    return temp_large_img

def average_arrays_ignore_zeros(out_array, array2):
    # still has an error where the values are halved each time because its averaged between zero
    # Create masks for zero values in each array (only works for float values)
    mask1 = (out_array != 0)
    mask2 = (array2 != 0)

    # Combine masks to find non-zero values in either arrays
    mask_or = logical_or(mask1, mask2)
    mask_and = logical_and(mask1, mask2)

    nonoverlapping_mask = logical_xor(mask_or,mask_and)
    overlapping_mask = logical_xor(mask_or,nonoverlapping_mask)

    # Calculate the average for non-zero values
    out_array[nonoverlapping_mask] = out_array[nonoverlapping_mask] + array2[nonoverlapping_mask]
    out_array[overlapping_mask] = (out_array[overlapping_mask] + array2[overlapping_mask]) / 2 

    return out_array

if __name__ == "__main__":

    print('pass')