import os,time,glob,sys,time,tqdm,cv2, serial, json, torch, scipy, math
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import label, find_objects, measurements
from skimage.measure import find_contours
from skimage.registration import phase_cross_correlation

# this is a test to try and segment out the 4 corners of the WM4 
# by finding the "center blob" of the image and then trying to find what a single well size is
# then using that to basically cleave out 4 different sections and to ignore the rest

def largest_blob(binary_image):
    # Label connected components in the binary image
    labeled_array, num_features = label(binary_image)

    # Find bounding boxes of labeled regions
    slices = find_objects(labeled_array)

    # Get the largest region based on area
    largest_area = 0
    largest_slice = None

    for i, slice_ in enumerate(slices):
        region = labeled_array[slice_]
        area = np.sum(region == i + 1)  # labeled_array starts from 1

        if area > largest_area:
            largest_area = area
            largest_slice = slice_
            largest_slice_idx = i

    largest_blob_image = labeled_array==(largest_slice_idx+1)

    return largest_blob_image

def normalize_to_smallest_nonzero(input_array):

    meansured_min = np.min(input_array[np.nonzero(input_array)])
    meansured_max = np.max(input_array[np.nonzero(input_array)])

    out = (input_array - meansured_min) / (meansured_max-meansured_min)
    out = np.clip(out,0,1)

    return out

def clip_to_not_singleton(input_array, n = 10):

    unique_vals, counts= np.unique(input_array,return_counts=True)

    idx = counts > n
    new_max = np.max(unique_vals[idx])

    out = np.clip(input_array/new_max,0,1)

    return out

def norm(input_array):

    input_array = input_array.astype(np.float32)

    input_array = input_array - np.min(input_array)

    return input_array/np.max(input_array)

def plot_images(images):
    num_images = len(images)
    num_cols = math.ceil(math.sqrt(num_images))
    num_rows = math.ceil(num_images / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 6))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i])
            ax.axis('off')
        else:
            ax.axis('off')

    plt.show()

def read_imgs(paths):
    imgs = []
    for path in paths:
        imgs.append(cv2.imread(path,-1))
    return imgs

img_paths = natsorted(glob.glob(os.path.join(r"Y:\Users\Sam Freitas\WP_imager\output\Ftest\wm_3R\2024-03-06\fluorescent_data",'*.png')))
output = 'output/HDR'
os.makedirs(output,exist_ok=True)

exposures = [-2,-4,-6,-8]

maxes = []
max_max = 0
for i,this_path in enumerate(img_paths):

    if (i % len(exposures))== 0:
        these_paths = img_paths[i:i+len(exposures)]
        this_imgs = read_imgs(these_paths)
        a = np.asarray(this_imgs).astype(np.float32)
        b = np.sum(a,axis = np.argmin(a.shape))

        if i == 0:
            totals = np.zeros_like(b)
        totals = totals + b

        this_max = b.max()
        maxes.append(b.max())
        if this_max > max_max:
            print(this_max)
            max_max = this_max

# get blob and moments
blob = largest_blob(totals>0)
blob = scipy.ndimage.binary_dilation(blob, iterations = 20)
blob = scipy.ndimage.binary_fill_holes(blob)
blob = blob.astype(np.uint8)

M = cv2.moments(blob) 
# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
# get indecies of nonzeros
x = np.nonzero(blob)

bbox = []
bbox.append(x[0].min()) # x min
bbox.append(x[0].max()) # x max
bbox.append(x[1].min()) # y min
bbox.append(x[1].max()) # y max

backtorgb = cv2.cvtColor(blob,cv2.COLOR_GRAY2RGB)

n = (normalize_to_smallest_nonzero(totals)*255).astype(np.uint8)
n = (clip_to_not_singleton(n)*255).astype(np.uint8)

backtorgb[:,:,0] = n
backtorgb[:,:,1] = n
backtorgb[:,:,2] = n

start_point = (int(bbox[2]), int(bbox[0]))
end_point = (int(bbox[3]), int(bbox[1]))
cv2.rectangle(backtorgb, start_point, end_point, color=(0,255,0), thickness=2)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(backtorgb)
plt.plot(cX,cY,'co')
plt.subplot(1,2,2)
plt.imshow(blob - norm(n))
plt.plot(cX,cY,'ro')
plt.show()

print('eof')