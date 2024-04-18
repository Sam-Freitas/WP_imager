import cv2, os, tqdm, glob, time, datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy, skimage
import camera.camera_control
import analysis.fluor_postprocess
from natsort import natsorted
import h5py


def save_images_to_h5(images, file_path):
    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset('images', data=images)

def load_images_from_h5(file_path):
    with h5py.File(file_path, 'r') as hf:
        images = hf['images'][:]
    return images

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

def open_blur(input_img, iterations):

    temp_img = input_img.copy()
    sigma = 2.5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    for i in range(iterations):
        temp_img = cv2.morphologyEx(temp_img,cv2.MORPH_OPEN, kernel, iterations = iterations)
        # temp_img = cv2.morphologyEx(temp_img, cv2.MORPH_TOPHAT, kernel, iterations= iterations)
        temp_img = cv2.GaussianBlur(temp_img, (0, 0), sigma, sigma)

    return temp_img


## this is an attempt for a first pass in the fluorescent analysis of a WM4 plate (red)
data_dir = r"Y:\Users\Sam Freitas\WP_imager\output\Ftest\wm_3R\2024-02-26\fluorescent_data"
img_paths = natsorted(glob.glob(os.path.join(data_dir,'*.png')))

out_dir = 'output/Ftest/analysis'
os.makedirs(out_dir, exist_ok=True)

print('reading imgs')
imgs = []
for i,img in enumerate(img_paths):
    img = cv2.imread(img,-1)
    # img = scipy.ndimage.gaussian_filter(img,10)
    imgs.append(img)
imgs = np.asarray(imgs).astype(np.float32)
# save_images_to_h5(imgs, 'img_stack.h5')

# # Load the images from the HDF5 file
# imgs = load_images_from_h5('img_stack.h5')
print('filtering')
data_to_ignore = check_same_pixel_value(imgs)
imgs[:,data_to_ignore] = 0

print('mean imgs')
a = np.mean(imgs,axis = 0)
# bl = open_blur(a,10)
# bl = np.max(bl)-bl

actual_max = find_maximal_value_with_more_than_10_pixels(a)
b = np.clip(a/actual_max,0,1)
bl = open_blur(b,10)

for i,img in enumerate(imgs):
    if i % 3 == 0:

        # img_stack = np.moveaxis(imgs[i:i+3],0,-1)
        # bl2 = open_blur(img_stack,10)

        c = img - (1+np.mean(img))*bl
        c = analysis.fluor_postprocess.norm(np.clip(c,0,c.max()))*255

        bw = c > (np.mean(c[~data_to_ignore]) + (2*np.std(c[~data_to_ignore])))
        cleaned = skimage.morphology.remove_small_objects(bw, min_size=64, connectivity=3)

        d = np.concatenate((img,c,cleaned*255),axis = -1)

        # cv2.imwrite(os.path.join(out_dir,str(i) + '.png'), d.astype(np.uint8))
        cv2.imwrite(os.path.join(out_dir,str(i) + '_c.png'), c.astype(np.uint8))

plt.imshow(bl)
plt.show()

print('eof')