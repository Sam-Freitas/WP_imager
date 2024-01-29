import cv2, os, tqdm, glob, time, datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.ndimage import label, find_objects, measurements
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

    # # Create an empty array for the largest blob
    # largest_blob_image = np.zeros_like(binary_image)

    # if largest_slice is not None:
    #     largest_blob_image[largest_slice] = 1

    largest_blob_image = labeled_array==(largest_slice_idx+1)

    return largest_blob_image

# def crop_smallest_square(binary_blob):
#     # Find non-zero indices (coordinates) of the blob
#     non_zero_indices = np.nonzero(binary_blob)

#     # Find the minimum and maximum coordinates
#     min_row, min_col = np.min(non_zero_indices, axis=1)
#     max_row, max_col = np.max(non_zero_indices, axis=1)

#     # Calculate the size of the smallest square
#     size = max(max_row - min_row, max_col - min_col)

#     # Crop the smallest square from the binary blob
#     cropped_blob = binary_blob[min_row:min_row + size, min_col:min_col + size]

#     # Return the coordinates and the cropped blob
#     return (min_row, min_col, size), cropped_blob
    
# def place_image_np(bg_array, img_array, center_x, center_y):
#     # Get dimensions
#     bg_height, bg_width = bg_array.shape
#     img_height, img_width = img_array.shape

#     # Calculate the top-left corner coordinates based on the center
#     x = center_x - img_width // 2
#     y = center_y - img_height // 2

#     # Check if placement is out of bounds
#     if x < 0:
#         img_array = img_array[:, -x:]
#         x = 0
#     if y < 0:
#         img_array = img_array[-y:, :]
#         y = 0
#     if x + img_width > bg_width:
#         img_array = img_array[:, :bg_width - x]
#         x = bg_width - img_width
#     if y + img_height > bg_height:
#         img_array = img_array[:bg_height - y, :]
#         y = bg_height - img_height

#     # Paste the image onto the background
#     bg_array[y:y+img_height, x:x+img_width, :] = img_array

# def put_uncropped_img_in_large_img(extent_y,extent_x,pixels_per_mm,FOV,delta_x,delta_y, counter, img_data, row, col, center):
#     row_start = int(row * pixels_per_mm * abs(delta_x))
#     row_end = row_start + int(pixels_per_mm * FOV)
#     col_start = int(col * pixels_per_mm * abs(delta_y))
#     col_end = col_start + int(pixels_per_mm * FOV)

#     temp_large_img = np.zeros((int(extent_y * pixels_per_mm), int(extent_x * pixels_per_mm)))

#     placing_center = [np.mean([row_start,row_end]),np.mean([col_start,col_end])]
#     center_int = [int(np.round(point)) for point in center]

#     temp_large_img = place_image_np(temp_large_img, img_data, placing_center[0], placing_center[1])

#     temp_large_img[row_start:row_end, col_start:col_end] = img_data #camera.camera_control.crop_center_numpy_return(img_data,pixels_per_mm*FOV, center = center_int)

#     return temp_large_img

def align_slivers(slice1,slice2):
    
    slice2_aligned = []
    return slice2

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
    norm_array = camera.camera_control.crop_center_numpy_return(norm_array_full,pixels_per_mm*FOV, center = center_int)
    
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
            img_data_cropped = camera.camera_control.crop_center_numpy_return(images_loaded[counter],pixels_per_mm*FOV, center = center_int)
            img_data_cropped = img_data_cropped*(norm_array+1)
            images_cropped.append(img_data_cropped)
            temp_large_img = camera.camera_control.put_frame_in_large_img(extent_y,extent_x,pixels_per_mm,FOV,delta_x,delta_y, counter, img_data_cropped, row, col)

            # for overlapping images (still needs work)
            if (row == 0) and (col == 0):
                large_img = temp_large_img
            else:
                temp_large_img = match_intensities(large_img,temp_large_img)
                large_img = camera.camera_control.average_arrays_ignore_zeros(large_img, temp_large_img)
            
            camera.camera_control.imshow_resize('img',large_img.astype(np.uint8), resize_size = [640,640])
            counter += 1

    return images, img_data_cropped, large_img
    
    


if __name__ == "__main__":

    print('pass')