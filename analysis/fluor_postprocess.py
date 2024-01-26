import cv2, os, tqdm, glob, time, datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy
import camera.camera_control

def align_frames(frames):

    images_loaded = np.asarray(frames)
        
    a = np.mean(images_loaded, axis = 0) # get the average image taken of the stack (for illumination correction)

    # binary_img = a > 0
    center = [ np.average(indices) for indices in np.where(a > 0) ] # find where the actual center of the frame is 
    center_int = [int(np.round(point)) for point in center]

    out = scipy.ndimage.gaussian_filter(a,100)
    out = 1-(out/np.max(out))
    out = camera.camera_control.crop_center_numpy_return(out,pixels_per_mm*FOV, center = center_int)

    # counter = 1
    # for i in range(img_num):
    #     for j in range(img_num):
    #         plt.subplot(img_num,img_num,counter)
    #         plt.imshow(camera.camera_control.crop_center_numpy_return(images_loaded[counter-1],pixels_per_mm*FOV)*(out+1))
    #         counter += 1

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
                large_img = camera.camera_control.average_arrays_ignore_zeros(large_img, temp_large_img)
            
            camera.camera_control.imshow_resize('img',large_img.astype(np.uint8), resize_size = [640,640])
            counter += 1
            cv2.imwrite('square_test.bmp', large_img.astype(np.uint8))
            
    cv2.imwrite('square_test.bmp', large_img.astype(np.uint8))
    # np.save('test.npy',np.asarray(images))

    return images, img_data_cropped, large_img
    
    


if __name__ == "__main__":

    print('pass')