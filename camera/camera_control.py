import cv2, os, tqdm, glob, time, datetime
import matplotlib.pyplot as plt

def imshow_resize(frame_name = "img", frame = 0, resize_size = [640,480], default_ratio = 1.3333, always_on_top = True, use_waitkey = True):

    frame = cv2.resize(frame, dsize=resize_size)
    cv2.imshow(frame_name, frame)
    if always_on_top:
        cv2.setWindowProperty(frame_name, cv2.WND_PROP_TOPMOST, 1)
    if use_waitkey:
        cv2.waitKey(1)
    cv2.moveWindow(frame_name,1920-resize_size[0],1)
    return True

def del_dir_contents(path_to_dir):
    files = glob.glob(os.path.join(path_to_dir,'*'))
    for f in files:
        os.remove(f)

def capture_images_for_time(cap,N):

    start_time = datetime.datetime.now()

    while True:
        current_time = datetime.datetime.now()
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break
        # imshow_resize("img", frame, resize_size=[640,480])
        if start_time + datetime.timedelta(0,N + 0.1) < current_time:
            break 

def clear_camera_image_buffer(cap,N=2):
    for i in range(N):
        ret, frame = cap.read()

def simple_capture_data(camera_settings, plate_parameters = None, testing = False):

    todays_date = datetime.date.today().strftime("%Y-%m-%d")

    output_dir = os.path.join(os.getcwd(),'output',plate_parameters['experiment_name'],plate_parameters['plate_name'],todays_date)
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

    time_between_images_seconds = 0.1

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

        # cv2.imwrite(image_filename, frame)
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
                capture_images_for_time(cap,delay_time)
        # time.sleep(1)

    # Release the camera
    cv2.destroyWindow("img")
    cap.release()

if __name__ == "__main__":

    print('pass')