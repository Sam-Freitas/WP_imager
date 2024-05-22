import os,time,glob,sys,time,tqdm,cv2#, serial, json, torch, scipy
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt

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

def norm(input_array):

    input_array = input_array - np.min(input_array)

    return input_array/np.max(input_array)

def display_images(images):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')  # Assuming images are grayscale
        ax.axis('off')
    plt.tight_layout()
    plt.show(block = False)

def display_nonzero_histograms(images):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        non_zero_values = images[i][images[i] != 0].ravel()
        ax.hist(non_zero_values, bins=256, range=(0, 255),log=True)  # Assuming images are in the range [0, 1]
        ax.set_title(f'Image {i+1}')
    plt.tight_layout()
    plt.show(block = False)

def display_images_and_histograms(images, title = "Images and Histograms"):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(title)  # Set title for the overarching figure
    img_min = np.min(images)
    img_max = np.max(images)
    for i, ax in enumerate(axes.flat):
        if i < 4:
            # Display image
            ax.imshow(images[i], cmap='gray')#, vmin=0, vmax=255)  # Assuming images are grayscale
            ax.axis('off')
            ax.set_title(f'Image {i+1}')
        else:
            # Display histogram
            hist_ax = ax.twinx()
            image_index = i - 4
            non_zero_values = images[image_index][images[image_index] != 0].ravel()
            hist_ax.hist(non_zero_values, bins=256, range=(img_min, img_max), log=True, color='orange', alpha=0.7)  # Assuming images are in the range [0, 255]
            # hist_ax.set_yticks([])
            # ax.set_xticks([])
    # plt.tight_layout()
    plt.show(block = True)

path_to_testing_imgs = r"Y:\Users\Sam Freitas\WP_imager\output\TJ356\TJ356-1F3\2024-05-03\fluorescent_data"
img_paths = natsorted(glob.glob(os.path.join(path_to_testing_imgs,'*.png')))

cv2_exposures = [-8,-6,-4,-2]#,-2]#[-2,-4,-6,-8]

N = len(img_paths)  # You can replace 20 with any number you want
for i in range(N):
    if i % 4 == 0:
        idxs = [j for j in range(i, i+4)]

        these_img_paths = [img_paths[j] for j in idxs]

        these_imgs = [cv2.imread(p,-1) for p in these_img_paths]#).astype(np.float32)

        these_imgs = np.asarray([crop_center_numpy_return(p,1000,center = [1440,1252]) for p in these_imgs])

        # these_imgs_norms = np.asarray([norm(img) for img in these_imgs])

        # mask = scipy.ndimage.gaussian_filter(these_imgs[0], sigma = 0.75)>0
        # mask_nd = np.asarray([mask,mask,mask,mask])

        display_images_and_histograms(these_imgs, title = os.path.basename(these_img_paths[-1]))#*mask_nd)

        pass
        plt.close('all')



print(img_paths)

