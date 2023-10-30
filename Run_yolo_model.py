from ultralytics_yolov5_master.models.experimental import attempt_load
import os,glob,torch,cv2
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    if torch.backends.mps.is_available():
        device = torch.device('mps')

def normalize_img(input_img):
    input_img = input_img - input_img.min()
    input_img = input_img/input_img.max()
    return input_img

def s(img, title = None):
    matplotlib.use('TkAgg')
    if 'torch' in str(img.dtype):
        img = img.squeeze()
        if len(img.shape) > 2: # check RGB
            if np.argmin(torch.tensor(img.shape)) == 0: # check if CHW 
                img = img.permute((1, 2, 0)) # change to HWC
        img = normalize_img(img)*255
        img = img.to('cpu').to(torch.uint8)
    else:
        img_shape = img.shape
        if np.argmin(img_shape) == 0:
            img = np.moveaxis(img,0,-1)
    plt.figure()
    plt.imshow(img)
    plt.title(title)

def run_yolo_model(img_filename = None, plot_results = False, i = 0, pause_block = False):
    model_path = 'ultralytics_yolov5_master\WPdata_weightsWMterasaki.pt'
    yolomodel = attempt_load(model_path, map_location='cuda') # load model

    if img_filename is None:
        img_file_paths = glob.glob('*.png') + glob.glob('*.jpg') # get the images
    else:
        img_file_paths = [img_filename]

    img = cv2.imread(img_file_paths[0])
    IMAGE_H,IMAGE_W,_ = img.shape
    resize_H,resize_W = 704,704 # get image shape and shape to be transformed into

    a = cv2.resize(img,(resize_H,resize_W))
    a = torch.permute(torch.tensor(a),(2,0,1)).to(torch.float32)/255
    a = a.to(device).unsqueeze(0) # make the image into the correct form

    out = yolomodel(a)[0] # run the model
    conf = out[0][:,4] # get the confidence values of the detected stuff
    temp = out[0][conf>0.6,:].cpu() ############# i think its x,y,w,h,conf,class0,class1
    conf2 = temp[:,4] # get the other confidence intervals

    print('converting output')

    points = temp[:,0:2]
    kmeans = KMeans(n_clusters=96, random_state=0, n_init="auto").fit(points)
    cluster_centers = kmeans.cluster_centers_

    labels = kmeans.labels_
    centers = np.zeros((len(np.unique(labels)),2))

    # get the average of the clustered centers 
    for label in np.unique(labels):
        idx = labels==label
        these_points = points[idx,:]
        these_conf = conf2[idx]
        this_center = np.average(these_points,axis = 0,weights = these_conf)
        centers[label,:] = this_center

    centers_sum = np.sum(centers,axis=1)
    sort_idx = np.argsort(centers_sum)

    # sort the centers by sortrows (????)
    sorted_centers = centers[sort_idx] 
    center_of_plate = np.average(sorted_centers,axis=0)

    normalized_centers = sorted_centers/resize_H
    normalized_center_of_plate = center_of_plate/resize_H

    input_sized_centers = normalized_centers
    input_sized_centers[:,0] = input_sized_centers[:,0]*IMAGE_W
    input_sized_centers[:,1] = input_sized_centers[:,1]*IMAGE_H
    input_sized_center_of_plate = normalized_center_of_plate
    input_sized_center_of_plate[0] = input_sized_center_of_plate[0]*IMAGE_W
    input_sized_center_of_plate[1] = input_sized_center_of_plate[1]*IMAGE_H

    if plot_results:

        print("SHOWING:", sorted_centers.shape[0], " ITEMS")
        cmap = matplotlib.colormaps['jet']
        s(img)
        for i,this_center in enumerate(sorted_centers):
            if i == 0 or i == len(sorted_centers)-1:
                plt.plot(this_center[0]*(IMAGE_W/resize_W),this_center[1]*(IMAGE_H/resize_H),marker = 'X',markersize = 20,color = cmap(i/sorted_centers.shape[0]))
            else:        
                plt.plot(this_center[0]*(IMAGE_W/resize_W),this_center[1]*(IMAGE_H/resize_H),marker = 'o',color = cmap(i/sorted_centers.shape[0]))
            # plt.pause(0.05)
        plt.plot(center_of_plate[0]*(IMAGE_W/resize_W),center_of_plate[1]*(IMAGE_H/resize_H),marker = 'P',color = 'green')
        plt.plot((IMAGE_W/2),(IMAGE_H/2),marker = 'P',color = 'cyan')
        plt.savefig('output\calibration\calib_out' + str(i) + '.jpg',dpi = 500)

        if pause_block == True:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(3)
            plt.close()

    return input_sized_centers,input_sized_center_of_plate

if __name__ == "__main__":

    run_yolo_model(plot_results = True)
    print('eof')
# for i in range(points.shape[0]):
#     plt.plot(points[i][0]*(IMAGE_W/resize_W),points[i][1]*(IMAGE_H/resize_H),'bo')
# plt.show()

# print(yolomodel)


# imgs = []
# img_names = []
# for filename in img_file_paths: #assuming gif
#     # print('Opening image: ', filename)
#     base = os.path.basename(filename)
#     img_names.append(os.path.splitext(base)[0])
#     im=Image.open(filename)
#     imgs.append(im)
