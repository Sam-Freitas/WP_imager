import os,time,glob,sys,time,tqdm,cv2#, serial, json, torch, scipy
import numpy as np
from natsort import natsorted
from fnmatch import fnmatch
import matplotlib.pyplot as plt
from pystackreg import StackReg
from skimage import transform, io, exposure
import pyGPUreg as reg

def composite_images(imgs, equalize=False, aggregator=np.mean):

    if equalize:
        imgs = [exposure.equalize_hist(img) for img in imgs]

    imgs = [img / img.max() for img in imgs]

    if len(imgs) < 3:
        imgs += [np.zeros(shape=imgs[0].shape)] * (3-len(imgs))

    imgs = np.dstack(imgs)

    return imgs

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

def display_images(images, title = '',blocking = False):
    if 'list' in str(type(images)):
        fig, axes = plt.subplots(1, len(images), figsize=(8, 8))
        vmin = 0
        vmax = 255
    else:
        fig, axes = plt.subplots(1, images.shape[0], figsize=(8, 8))
        vmin = 0
        vmax = np.max(images)
    fig.suptitle(title)  # Set title for the overarching figure

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray', vmin=vmin, vmax=vmax) # Assuming images are grayscale
        ax.axis('off')
    plt.tight_layout()
    plt.show(block = blocking)

def display_images2(images, title = '',blocking = False):
    if 'list' in str(type(images)):
        fig, axes = plt.subplots(2, len(images)-1, figsize=(8, 8))
        vmin = 0
        vmax = 255
    else:
        fig, axes = plt.subplots(2, images.shape[0]-1, figsize=(8, 8))
        vmin = 0
        vmax = np.max(images)
    fig.suptitle(title)  # Set title for the overarching figure

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray', vmin=vmin, vmax=vmax) # Assuming images are grayscale
        ax.axis('off')
    plt.tight_layout()
    plt.show(block = blocking)

def display_images_and_histograms(images, title = "Images and Histograms"):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(title)  # Set title for the overarching figure
    img_min = 0
    img_max = np.max(images)
    for i, ax in enumerate(axes.flat):
        if i < 4:
            # Display image
            ax.imshow(images[i], cmap='gray', vmin=img_min, vmax=img_max)  # Assuming images are grayscale
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

path_to_testing_imgs = r"Y:\Users\Sam Freitas\WP_imager\output\TJ356\TJ356-1F3"
pattern = '*.png'
img_paths = []
# find all the image files and then natsort them
for path, subdirs, files in os.walk(path_to_testing_imgs):
    for name in files:
        if fnmatch(name, pattern):
            if ('fluorescent_data' in path) and ('-2_' in name):
                img_paths.append(os.path.join(path, name))
img_paths = natsorted(img_paths)

all_names = [os.path.basename(p) for p in img_paths]
unique_names = natsorted(list(np.unique(all_names)))


reg.init()

for i,name_to_check in enumerate(unique_names):
# name_to_check = "a4_004_-2_.png"

    idx = []
    check_these_paths = []
    for j,p in enumerate(img_paths):
        if name_to_check in p:
            idx.append(i)
            check_these_paths.append(p)

    # cv2_exposures = [-8,-6,-4,-2]#,-2]#[-2,-4,-6,-8]

    these_imgs = [cv2.imread(p,-1) for p in check_these_paths]
    before_reg = composite_images([these_imgs[0],these_imgs[-1]])
    # these_imgs_array = np.asarray(these_imgs)

    these_imgs_array = np.asarray([cv2.resize(crop_center_numpy_return(norm(p.astype(np.float64)),1600),dsize = (512,512)) for p in these_imgs])

    # reg.set_template(these_imgs_array[0])

    out = []
    out.append(these_imgs_array[0])
    for j,img in enumerate(these_imgs_array[1:]):
        registered_image, shift = reg.register(these_imgs_array[0], img, apply_shift=True, subpixel_mode=reg.SUBPIXEL_MODE_NONE, edge_mode=reg.EDGE_MODE_REFLECT, interpolation_mode=reg.INTERPOLATION_MODE_LINEAR)
        out.append(registered_image)
    out = np.asarray(out)

    after_reg =  composite_images([these_imgs_array[0], out[-1]])

    med_img = np.median(out,axis = 0)

    subtraction_img = np.sum(out,axis = 0) - ((j+2)*med_img)
    out_disp = composite_images([out[0], out[-1]])

    display_images([before_reg,after_reg,out_disp], title=name_to_check,blocking = True)

    print(check_these_paths)