import cv2, os, tqdm, glob, time, datetime
import matplotlib.pyplot as plt

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
def capture_images_for_time(cap,N, show_images = False, move_to = [100,100], resize_size = [640,480]):
    start_time = datetime.datetime.now()
    while True:
        current_time = datetime.datetime.now()
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break
        if show_images:
            imshow_resize("stream", frame, resize_size=resize_size, move_to=move_to)
        if start_time + datetime.timedelta(0,N + 0.1) < current_time:
            break 

# this captures the first N images to clear the pipeline (sometime just black images)
def clear_camera_image_buffer(cap,N=2):
    for i in range(N):
        ret, frame = cap.read()

def simple_capture_data(camera_settings, plate_parameters = None, testing = False, output_dir = None):

    todays_date = datetime.date.today().strftime("%Y-%m-%d")

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

    # time_between_images_seconds = 0 # this is just for testing 
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
        start_time = time.process_time()
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d (%H-%M-%S-%f)")
        image_name = current_time + '.' + img_file_format
        image_filename = os.path.join(output_dir, image_name)

        cv2.imwrite(image_filename, frame)
        # print(f"\nCaptured image {i+1}/{num_images}")

        # Put the text on the image
        cv2.putText(frame, text, (int(text_x), int(text_y)), font, font_scale, font_color2, thickness2) # black 
        cv2.putText(frame, text, (int(text_x), int(text_y)), font, font_scale, font_color1, thickness1) # white

        imshow_resize("img", frame, resize_size=[640,480])

        end_time = time.process_time() - start_time
        delay_time = time_between_images_seconds-end_time
        if i != num_images-1:
            if delay_time<0:
                pass
            else:
                capture_images_for_time(cap,delay_time, show_images=True,move_to = [1920,520])
        # time.sleep(1)

    # Release the camera
    cv2.destroyWindow("img")
    cap.release()

if __name__ == "__main__":

    print('pass')