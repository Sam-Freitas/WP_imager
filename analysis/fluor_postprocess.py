import numpy as np
import cv2, pathlib, os, glob, matplotlib, imagesize, tqdm
from natsort import natsorted
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
        if len(img_shape) > 2:
            if np.argmin(img_shape) == 0:
                img = np.moveaxis(img,0,-1)
    plt.figure()
    plt.imshow(img)
    plt.title(title)

current_path = pathlib.Path(__file__).parent.resolve()
path_to_dir = r'output\GLS371_testing\t_1\2023-11-21\fluorescent_data'
file_format = '.png'

output_dir = os.path.join(current_path,'output')
os.makedirs(output_dir, exist_ok=True)

img_paths = natsorted(glob.glob(os.path.join(path_to_dir,'*' + file_format)))
width, height = imagesize.get(img_paths[0])

stack = np.zeros((len(img_paths),width,height), dtype = np.float64)
mean_img = np.zeros((width,height), dtype = np.float64)
# stack_resized = np.zeros(shape=(len(img_paths),128,128), dtype = np.float64)

for i,this_img_path in enumerate(tqdm.tqdm(img_paths)):

    stack[i,::] = cv2.imread(this_img_path,0)
    mean_img += stack[i,::]

mean_img = mean_img/len(img_paths)

# # temp = stack_resized-
# temp = stack - mean_img
# temp[temp<0] = 0
# temp = temp.astype(np.uint8)

# mean_img_large = cv2.resize(mean_img,(width,height))

for i,this_img in enumerate(tqdm.tqdm(stack)):

    out_name = os.path.split(img_paths[i])[-1]

    a = this_img - mean_img
    a[a<0] = 0

    cv2.imwrite(os.path.join(output_dir,out_name),a.astype(np.uint8))
