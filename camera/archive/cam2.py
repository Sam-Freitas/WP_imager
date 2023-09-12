import cv2, os, tqdm
import matplotlib.pyplot as plt


def imshow_resize(frame_name="img", frame=0, resize_size=[640, 480]):
    frame = cv2.resize(frame, dsize=resize_size)

    cv2.imshow(frame_name, frame)
    return True


# Configure camera settings
camera_id0 = 0  # Adjust this based on your system's camera index
camera_id1 = 1
num_images = 1000
output_folder = "captured_images"

capture_or_show = "show"  #'show' # 'capture' # 'both'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

cam_height0 = 1824
cam_width0 = 2736

cam_height1 = 1824
cam_width1 = 1824

resize_size0 = [640, 480]
resize_size1 = [480, 480]


# Open the camera1
cap0 = cv2.VideoCapture(camera_id0)
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width0)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height0)
cap0.set(cv2.CAP_PROP_FPS, 24)

# Open the camera2
cap1 = cv2.VideoCapture(camera_id1)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width1)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height1)
cap1.set(cv2.CAP_PROP_FPS, 24)

if not cap0.isOpened():
    print("Error: Unable to open camera0.")
    exit()
if not cap1.isOpened():
    print("Error: Unable to open camera1.")
    exit()

# Capture a series of images
for i in tqdm.tqdm(range(num_images)):
    ret0, frame0 = cap0.read()

    if not ret0:
        print("Error: Unable to capture frame0.")
        break

    imshow_resize("img0", frame0, resize_size=resize_size0)

    c = cv2.waitKey(int(0.01 * 1000))
    if c == 27 or c == 10:
        break

# Release the camera
cv2.destroyAllWindows()

# Capture a series of images
for i in tqdm.tqdm(range(num_images)):
    ret1, frame1 = cap1.read()

    if not ret1:
        print("Error: Unable to capture frame0.")
        break

    imshow_resize("img1", frame1, resize_size=resize_size1)

    c = cv2.waitKey(int(0.01 * 1000))
    if c == 27 or c == 10:
        break

# Release the camera
cv2.destroyAllWindows()

# Capture a series of images
for i in tqdm.tqdm(range(num_images)):
    ret0, frame0 = cap0.read()
    if not ret0:
        print("Error: Unable to capture frame0.")
        break

    ret1, frame1 = cap1.read()
    if not ret1:
        print("Error: Unable to capture frame0.")
        break

    imshow_resize("img0", frame0, resize_size=resize_size0)
    imshow_resize("img1", frame1, resize_size=resize_size1)

    c = cv2.waitKey(int(0.01 * 1000))
    if c == 27 or c == 10:
        break

# Release the camera
cap0.release()
cap1.release()
cv2.destroyAllWindows()

print("Image capture complete.")
