import cv2, os, tqdm, glob, time, datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy
import camera.camera_control

def align_slivers(slice1,slice2):
    
    slice2_aligned = []
    return slice2

def match_intensities(img1,img2):

    a = scipy.ndimage.gaussian_filter(img1,10) # large img
    b = scipy.ndimage.gaussian_filter(img2,10) # temp large img

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

    # binary_img = a > 0
    center = [ np.average(indices) for indices in np.where(a > 0) ] # find where the actual center of the frame is 
    center_int = [int(np.round(point)) for point in center]

    out = scipy.ndimage.gaussian_filter(a,100) # get the instensities of the images for the illuminance normalizations
    out = camera.camera_control.crop_center_numpy_return(out,pixels_per_mm*FOV, center = center_int)
    out = 1-(out/np.max(out))
    
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

            images.append(frame)
            img_data_cropped = camera.camera_control.crop_center_numpy_return(images_loaded[counter],pixels_per_mm*FOV, center = center_int)
            img_data_cropped = img_data_cropped*(out+1)
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
            # cv2.imwrite('square_test.bmp', large_img.astype(np.uint8))
            
    # cv2.imwrite('square_test.bmp', large_img.astype(np.uint8))
    # np.save('test.npy',np.asarray(images))

    return images, img_data_cropped, large_img
    
    


if __name__ == "__main__":

    print('pass')