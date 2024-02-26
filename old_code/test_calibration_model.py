import os,time,glob,sys,time,tqdm,cv2, serial, json, torch
import settings.get_settings
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from natsort import natsorted
from Run_yolo_model import yolo_model

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

# set up the calibration model (YOLO)
calibration_model = yolo_model()
imgs = natsorted(glob.glob(os.path.join(r'C:\Users\LabPC2\Documents\GitHub\WP_imager\output\test\t_2\2023-11-02\fluorescent_data', '*.png')))
img_names = [os.path.split(name)[-1] for name in imgs]
s_terasaki_positions = settings.get_settings.get_terasaki_positions()
terasaki_names = list(s_terasaki_positions['name'].values())

dxdy = np.zeros(shape=(len(imgs),2))

for i,name in enumerate(terasaki_names):

    index = []
    for this_img_name in img_names:
        if name + '_' in this_img_name:
            index.append(True)
        else:
            index.append(False)

    print("index of image", np.argmax(index))
    img_path = str(np.asarray(imgs)[index][0])
    # img_path = r'output\calibration\2023-11-02 (12-20-57-328753).png'

    img = cv2.imread(img_path)
    IMAGE_H,IMAGE_W,_ = img.shape
    resize_H,resize_W = 704,704 # get image shape and shape to be transformed into
    small_size = [84,106] 

    base = np.zeros(shape=(resize_H,resize_W,3), dtype='uint8')
    base[int((resize_H/2)-(small_size[1]/2)):int((resize_H/2)+(small_size[1]/2)),int((resize_W/2)-(small_size[0]/2)):int((resize_W/2)+(small_size[0]/2)),:] = cv2.resize(img,(small_size[0],small_size[1]))

    img_in = base.copy()
    img_in = cv2.resize(img_in,(resize_H,resize_W))
    img_in = torch.permute(torch.tensor(img_in),(2,0,1)).to(torch.float32)/255
    img_in = img_in.to(device).unsqueeze(0) # make the image into the correct form

    out = calibration_model.only_model(img_in) ########## x,y,w,h,conf,class0,class1

    conf = out[0][:,4] # get the confidence values of the detected stuff
    temp = out[0][conf>0.2,:].cpu() ############# i think its x,y,w,h,conf,class0,class1
    conf2 = temp[:,4] # get the other confidence intervals

    center = np.array(np.average(temp[:,0:2], axis=0, weights=conf2))
    center_of_img = [resize_H/2,resize_W/2]

    dxdy[i,:] = center_of_img-center

    print('loop')
print(np.mean(dxdy,0))
print(np.mean(dxdy,0)/[69.7, 42])

#     print('converting output')

#     points = temp[:,0:2]
#     kmeans = KMeans(n_clusters=96, random_state=0, n_init="auto").fit(points)
#     cluster_centers = kmeans.cluster_centers_

#     labels = kmeans.labels_
#     centers = np.zeros((len(np.unique(labels)),2))

#     confidences = temp[:, 4]
#     centers = np.array([np.average(points[labels == i], axis=0, weights=confidences[labels == i]) for i in np.unique(labels)])

#     centers_sum = np.sum(centers,axis=1)
#     sort_idx = np.argsort(centers_sum)

#     # sort the centers by sortrows (????)
#     sorted_centers = centers[sort_idx] 
#     center_of_plate = np.average(sorted_centers,axis=0)

#     normalized_centers = sorted_centers/resize_H
#     normalized_center_of_plate = center_of_plate/resize_H

#     input_sized_centers = normalized_centers
#     input_sized_centers[:,0] = input_sized_centers[:,0]*IMAGE_W
#     input_sized_centers[:,1] = input_sized_centers[:,1]*IMAGE_H
#     input_sized_center_of_plate = normalized_center_of_plate
#     input_sized_center_of_plate[0] = input_sized_center_of_plate[0]*IMAGE_W
#     input_sized_center_of_plate[1] = input_sized_center_of_plate[1]*IMAGE_H


#     matplotlib.use('TkAgg')

#     print("SHOWING:", sorted_centers.shape[0], " ITEMS")
#     cmap = plt.get_cmap('jet', sorted_centers.shape[0])
#     plt.imshow(img)
#     # Plot points
#     plt.scatter(sorted_centers[:, 0] * (IMAGE_W / resize_W),
#                 sorted_centers[:, 1] * (IMAGE_H / resize_H),
#                 marker='o', c=range(sorted_centers.shape[0]), cmap=cmap)
#     # Highlight first and last points
#     plt.scatter([sorted_centers[0, 0] * (IMAGE_W / resize_W), sorted_centers[-1, 0] * (IMAGE_W / resize_W)],
#                 [sorted_centers[0, 1] * (IMAGE_H / resize_H), sorted_centers[-1, 1] * (IMAGE_H / resize_H)],
#                 marker='X', s=400, c='black')#, cmap=cmap)
#     # Plot center of plate
#     plt.scatter(center_of_plate[0] * (IMAGE_W / resize_W), center_of_plate[1] * (IMAGE_H / resize_H),
#                 marker='P', color='green')

#     # Plot center of image
#     plt.scatter(IMAGE_W / 2, IMAGE_H / 2, marker='P', color='cyan')

#     plt.show(block = True)

#     print('eof')
# print(imgs)