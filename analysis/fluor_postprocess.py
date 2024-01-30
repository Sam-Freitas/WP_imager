import cv2, os, tqdm, glob, time, datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.ndimage import label, find_objects, measurements
from skimage.registration import phase_cross_correlation
import camera.camera_control

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

def put_frame_in_large_img(extent_y, extent_x, pixels_per_mm, FOV, delta_x, delta_y, i, img_data, row, col):
    row_start = int(row * pixels_per_mm * abs(delta_x))
    row_end = row_start + int(pixels_per_mm * FOV)
    col_start = int(col * pixels_per_mm * abs(delta_y))
    col_end = col_start + int(pixels_per_mm * FOV)

    temp_large_img = np.zeros((int(extent_y * pixels_per_mm), int(extent_x * pixels_per_mm)))
    temp_large_img[row_start:row_end, col_start:col_end] = img_data

    return temp_large_img

def put_frame_in_large_img2(extent_y, extent_x, pixels_per_mm, FOV, delta_x, delta_y, i, img_data, row, col):
    row_start = int(row * pixels_per_mm * abs(delta_x))
    row_end = row_start + int(pixels_per_mm * FOV)
    col_start = int(col * pixels_per_mm * abs(delta_y))
    col_end = col_start + int(pixels_per_mm * FOV)

    temp_large_img = np.zeros((int(extent_y * pixels_per_mm), int(extent_x * pixels_per_mm)))

    # Calculate the cropped region for img_data
    img_data_row_start = max(0, -row_start)
    img_data_row_end = min(img_data.shape[0], temp_large_img.shape[0] - row_start)
    img_data_col_start = max(0, -col_start)
    img_data_col_end = min(img_data.shape[1], temp_large_img.shape[1] - col_start)

    # Crop img_data and place it in temp_large_img
    temp_large_img[row_start + img_data_row_start: row_start + img_data_row_end,
                   col_start + img_data_col_start: col_start + img_data_col_end] = img_data[img_data_row_start:img_data_row_end, img_data_col_start:img_data_col_end]

    return temp_large_img

def imshow_n_imgs(imgs):

    for i in range(len(imgs)):
        plt.subplot(1,len(imgs),i+1)
        plt.imshow(imgs[i])

    plt.show()

def average_arrays_ignore_zeros(out_array, array2, register_to_out_array=False):
    # Create masks for zero values in each array (only works for float values)
    mask1 = (out_array != 0)
    mask2 = (array2 != 0)

    # Combine masks to find non-zero values in either arrays
    mask_or = np.logical_or(mask1, mask2)
    mask_and = np.logical_and(mask1, mask2)

    nonoverlapping_mask = np.logical_xor(mask_or, mask_and)
    overlapping_mask = np.logical_xor(mask_or, nonoverlapping_mask)

    if register_to_out_array:
        bw_1 = out_array>(np.mean(out_array[np.nonzero(out_array)]))#   + np.std(out_array[np.nonzero(out_array)])    )
        bw_2 =    array2>(np.mean(   array2[np.nonzero(array2)])   )#+      np.std(   array2[np.nonzero(array2)]))
        # Register array2 to out_array using skimage's register_translation
        shift, error, diffphase = phase_cross_correlation(out_array*bw_1*(1*overlapping_mask),array2*bw_2*(1*overlapping_mask))
        shifted_array2 = scipy.ndimage.shift(array2,shift)  # Replace out_array with registered array2

        mask2 = (shifted_array2 > 1)
        shifted_array2 = (1*mask2)*shifted_array2

        # Combine masks to find non-zero values in either arrays
        mask_or = np.logical_or(mask1, mask2)
        mask_and = np.logical_and(mask1, mask2)

        nonoverlapping_mask = np.logical_xor(mask_or, mask_and)
        overlapping_mask = np.logical_xor(mask_or, nonoverlapping_mask)
    
        out = np.zeros_like(out_array)
    
        out[nonoverlapping_mask] = out_array[nonoverlapping_mask] + shifted_array2[nonoverlapping_mask]
        out[overlapping_mask] = shifted_array2[overlapping_mask] #(out_array[overlapping_mask] + shifted_array2[overlapping_mask]) / 2 

        return out
    else:
        # Calculate the average for non-zero values
        out_array[nonoverlapping_mask] = out_array[nonoverlapping_mask] + array2[nonoverlapping_mask]
        out_array[overlapping_mask] = (out_array[overlapping_mask] + array2[overlapping_mask]) / 2 

        return out_array

def match_intensities(img1,img2):

    a = scipy.ndimage.gaussian_filter(img1,2) # large img
    b = scipy.ndimage.gaussian_filter(img2,2) # temp large img

    overlap = np.logical_and(a,b)

    img1_idx = np.logical_and(img1,overlap)
    img2_idx = np.logical_and(img2,overlap)

    img1_values = img1[img1_idx]
    img2_values = img2[img2_idx]

    img1_mean = np.mean(img1_values)
    img2_mean = np.mean(img2_values)
    print(img1_mean)
    print(img2_mean)

    scaler = img1_mean/img2_mean
    img2_sclaed = img2*scaler

    return img2_sclaed

def align_frames(frames,pixels_per_mm,FOV,extent_x,extent_y,delta_x,delta_y):

    images_loaded = np.asarray(frames)

    a = np.mean(images_loaded, axis = 0) # get the average image taken of the stack (for illumination correction)

    binary_img = largest_blob(a > 0) # get the largest binary blob in the image
    # (min_row, min_col, size), cropped_blob = crop_smallest_square(binary_img) # get the square crop of the blob
    center = [ np.average(indices) for indices in np.where(binary_img) ] # find where the actual center of the frame is 
    center_int = [int(np.round(point)) for point in center]

    norm_array = scipy.ndimage.gaussian_filter(a,100) # get the instensities of the images for the illuminance normalizations
    norm_array_full = 1-(norm_array/np.max(norm_array))
    norm_array = crop_center_numpy_return(norm_array_full,pixels_per_mm*(FOV), center = center_int)
    
    y_images = x_images = int(np.sqrt(images_loaded.shape[0]))

    # the term row and col are not exact as the system might have overlapping images, this is just nomeclature if there was zero overlap
    images = []
    images_cropped = []
    counter = 0
    for row in range(y_images): # rows
        if row % 2:
            cols = range(x_images-1,-1,-1) # odd flag
        else:
            cols = range(0,x_images) # even flag (includes zero)

        for col in cols:
            print(row,col)

            frame = images_loaded[counter] 
                
            norm_frame = frame*(norm_array_full+1)
            # temp_large_img = put_uncropped_img_in_large_img(extent_y,extent_x,pixels_per_mm,FOV,delta_x,delta_y, counter, norm_frame, row, col,center)

            images.append(frame)
            img_data_cropped = crop_center_numpy_return(images_loaded[counter],pixels_per_mm*(FOV), center = center_int)
            img_data_cropped = img_data_cropped*(norm_array+1)
            images_cropped.append(img_data_cropped)
            temp_large_img = put_frame_in_large_img2(extent_y,extent_x,pixels_per_mm,FOV,delta_x,delta_y, counter, img_data_cropped, row, col)

            # for overlapping images (still needs work)
            if (row == 0) and (col == 0):
                large_img = temp_large_img
            else:
                temp_large_img = match_intensities(large_img,temp_large_img)
                large_img = average_arrays_ignore_zeros(large_img, temp_large_img, register_to_out_array = False) # , register_to_out_array = F)
            
            imshow_resize('img',large_img.astype(np.uint8), resize_size = [640,640])
            counter += 1

    return images, img_data_cropped, large_img
    
def align_frames_register(frames,pixels_per_mm,FOV,extent_x,extent_y,delta_x,delta_y,overlap = 1):

    images_loaded = np.asarray(frames)

    a = np.mean(images_loaded, axis = 0) # get the average image taken of the stack (for illumination correction)

    binary_img = largest_blob(a > 0) # get the largest binary blob in the image
    # (min_row, min_col, size), cropped_blob = crop_smallest_square(binary_img) # get the square crop of the blob
    center = [ np.average(indices) for indices in np.where(binary_img) ] # find where the actual center of the frame is 
    center_int = [int(np.round(point)) for point in center]

    norm_array = scipy.ndimage.gaussian_filter(a,100) # get the instensities of the images for the illuminance normalizations
    norm_array_full = 1-(norm_array/np.max(norm_array))
    norm_array = crop_center_numpy_return(norm_array_full,pixels_per_mm*(FOV+overlap), center = center_int)
    
    y_images = x_images = int(np.sqrt(images_loaded.shape[0]))

    # the term row and col are not exact as the system might have overlapping images, this is just nomeclature if there was zero overlap
    images = []
    images_cropped = []
    counter = 0
    for row in range(y_images): # rows
        if row % 2:
            cols = range(x_images-1,-1,-1) # odd flag
        else:
            cols = range(0,x_images) # even flag (includes zero)

        for col in cols:
            print(row,col)

            frame = images_loaded[counter] 
                
            norm_frame = frame*(norm_array_full+1)
            # temp_large_img = put_uncropped_img_in_large_img(extent_y,extent_x,pixels_per_mm,FOV,delta_x,delta_y, counter, norm_frame, row, col,center)

            images.append(frame)
            img_data_cropped = crop_center_numpy_return(images_loaded[counter],pixels_per_mm*(FOV+overlap), center = center_int)
            img_data_cropped = img_data_cropped*(norm_array+1)
            images_cropped.append(img_data_cropped)
            temp_large_img = put_frame_in_large_img2(extent_y,extent_x,pixels_per_mm,FOV,delta_x,delta_y, counter, img_data_cropped, row, col)

            # for overlapping images (still needs work)
            if (row == 0) and (col == 0):
                large_img = temp_large_img
            else:
                temp_large_img = match_intensities(large_img,temp_large_img)
                large_img = average_arrays_ignore_zeros(large_img, temp_large_img, register_to_out_array = True) # , register_to_out_array = F)
            
            cv2.imwrite('test.bmp',np.clip(large_img,0,255).astype(np.uint8))
            camera.camera_control.imshow_resize('img',large_img.astype(np.uint8), resize_size = [640,640])
            counter += 1

    return images, img_data_cropped, large_img
    

if __name__ == "__main__":

    print('pass')