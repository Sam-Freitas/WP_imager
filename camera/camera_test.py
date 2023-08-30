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

def generator():
    while True:
        yield

if __name__ == "__main__":

    import pandas as pd

    camera_settings = pd.read_csv('settings\settings_cameras.txt', delimiter = '\t',index_col=False).to_dict()

    camera_id = camera_settings['widefield'][0]
    cam_width = camera_settings['widefield'][1]
    cam_height = camera_settings['widefield'][2]
    cam_framerate = camera_settings['widefield'][3]
    time_between_images_seconds = float(camera_settings['widefield'][4])
    time_of_single_burst_seconds = camera_settings['widefield'][5]
    number_of_images_per_burst = camera_settings['widefield'][6]
    img_file_format = camera_settings['widefield'][7]
    img_pixel_depth = camera_settings['widefield'][8]

    # time_between_images_seconds = 0.1

    # Open the camera0
    cap = cv2.VideoCapture(int(camera_id))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,int(cam_width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,int(cam_height))
    cap.set(cv2.CAP_PROP_FPS,int(cam_framerate))

    cap.set(cv2.CAP_PROP_PVAPI_BINNINGX,4)
    cap.set(cv2.CAP_PROP_PVAPI_BINNINGY,4)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,int(1824))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,int(1824))
    cap.set(cv2.CAP_PROP_FPS,int(100))

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        exit()

    window_name = "Press ESC on window to end"

    # Capture a series of images
    for _ in tqdm.tqdm(generator()):
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        imshow_resize(window_name, frame, resize_size=[640,480],always_on_top = True, use_waitkey = False)
        # time.sleep(1)
        c = cv2.waitKey(1)
        if c == 27 or c == 10:
            break

    # Release the camera
    cv2.destroyWindow(window_name)
    cap.release()