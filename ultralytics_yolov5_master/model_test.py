from models.experimental import attempt_load
import os,glob,torch,cv2
from PIL import Image
from utils2 import s
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    if torch.backends.mps.is_available():
        device = torch.device('mps')

model_path = 'WPdata_weightsWMterasaki.pt'
yolomodel = attempt_load(model_path, map_location='cuda')

img_file_paths = glob.glob('*.png')

a = cv2.imread(img_file_paths[0])
a = cv2.resize(a,(512,512))
a = torch.permute(torch.tensor(a),(2,0,1)).to(torch.float32)/255
a = a.to(device).unsqueeze(0)

b = torch.concatenate((a,a),0)

out = yolomodel(a)[0]

conf = out[0][:,4]

temp = out[0][conf>0.6,:].cpu() ############# i think its x,y,w,h,conf,class0,class1

import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

points = temp[:,0:2]

print("SHOWING:", temp.shape[0], " ITEMS")
s(a)
for i in range(temp.shape[0]):
    plt.plot(temp[i][0],temp[i][1],'ro')

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=96, random_state=0, n_init="auto").fit(points)

cluster_centers = kmeans.cluster_centers_

print("SHOWING:", temp.shape[0], " ITEMS")
s(a)
for i in range(cluster_centers.shape[0]):
    plt.plot(cluster_centers[i][0],cluster_centers[i][1],'bo')
plt.show()

print(yolomodel)


# imgs = []
# img_names = []
# for filename in img_file_paths: #assuming gif
#     # print('Opening image: ', filename)
#     base = os.path.basename(filename)
#     img_names.append(os.path.splitext(base)[0])
#     im=Image.open(filename)
#     imgs.append(im)
