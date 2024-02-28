import cv2, os, tqdm, glob, time, datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy
import camera.camera_control
import analysis.fluor_postprocess
from natsort import natsorted

def check_same_pixel_value(image_stack):
    # Get the shape of the image stack
    num_images, height, width = image_stack.shape
    
    # Reshape the image stack to have the same number of dimensions
    reshaped_stack = image_stack.reshape(num_images, -1)
    
    # Compare all pixel values across all images
    # Check if all values along the first axis are equal (same pixel location, same value)
    same_values = np.all(reshaped_stack == reshaped_stack[0, :], axis=0)
    
    # Reshape the boolean array back to the original image shape
    same_values = same_values.reshape(height, width)
    
    return same_values

def find_maximal_value_with_more_than_10_pixels(array):
    # Get unique values and their counts
    unique_values, counts = np.unique(array, return_counts=True)
    
    # Find values with more than 10 pixels
    values_with_more_than_10_pixels = unique_values[counts > 10]
    
    # If there are no values with more than 10 pixels, return None
    if len(values_with_more_than_10_pixels) == 0:
        return None
    
    # Find the maximal value among those
    maximal_value = np.max(values_with_more_than_10_pixels)
    
    return maximal_value

## this is an attempt for a first pass in the fluorescent analysis of a WM4 plate (red)
data_dir = r"Y:\Users\Sam Freitas\WP_imager\output\Ftest\wm_3R\2024-02-26\fluorescent_data"
img_paths = natsorted(glob.glob(os.path.join(data_dir,'*.png')))

print('reading imgs')
imgs = []
for i,img in enumerate(img_paths):
    img = cv2.imread(img,-1)
    img = scipy.ndimage.gaussian_filter(img,10)
    imgs.append(img)

print('filtering')
imgs = np.asarray(imgs).astype(np.float32)
data_to_ignore = check_same_pixel_value(imgs)
imgs[:,data_to_ignore] = 0

print('mean imgs')
# imgs = scipy.ndimage.gaussian_filter(imgs,2)
a = np.mean(imgs,axis = 0)
actual_max = find_maximal_value_with_more_than_10_pixels(a)
b = np.clip(a/actual_max,0,1)

plt.imshow(b)
plt.show()
print(imgs)