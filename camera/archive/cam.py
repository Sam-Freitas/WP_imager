import cv2, os, tqdm, glob
import matplotlib.pyplot as plt


def imshow_resize(
    frame_name="img", frame=0, resize_size=[640, 480], default_ratio=1.3333
):
    frame = cv2.resize(frame, dsize=resize_size)
    cv2.imshow(frame_name, frame)
    return True


def del_dir_contents(path_to_dir):
    files = glob.glob(os.path.join(path_to_dir, "*"))
    for f in files:
        os.remove(f)


# Configure camera settings
camera_id = 0  # Adjust this based on your system's camera index
num_images = 100
output_folder = "camera/captured_images2"

capture_or_show = "both"  #'show' # 'capture' # 'both'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
del_dir_contents(output_folder)

cam_height = 1824
cam_width = 1824

# Open the camera0
cap = cv2.VideoCapture(camera_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
cap.set(cv2.CAP_PROP_FPS, 14)

if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

# cv2.namedWindow("img", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)

# Capture a series of images
for i in tqdm.tqdm(range(num_images)):
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    image_filename = os.path.join(output_folder, f"image_{i+1}.jpg")
    # image_filename = os.path.join(output_folder, "cap.jpg")
    if capture_or_show == "capture":
        cv2.imwrite(image_filename, frame)
        print(f"Captured image {i+1}/{num_images}")
    if capture_or_show == "show":
        imshow_resize("img", frame, resize_size=[640, 640])
    if capture_or_show == "both":
        cv2.imwrite(image_filename, frame)
        print(f"Captured image {i+1}/{num_images}")
        imshow_resize("img", frame, resize_size=[640, 640])

    c = cv2.waitKey(1)
    if c == 27 or c == 10:
        break

# Release the camera
cv2.destroyWindow("img")
cap.release()

print("Image capture complete.")
