import numpy as np
import cv2, pathlib, os, glob, matplotlib, imagesize, tqdm
from natsort import natsorted
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def normalize_img(input_img):
    input_img = input_img - input_img.min()
    input_img = input_img/input_img.max()
    return input_img

def norm_5_to_95(img):

    img_mean = np.mean(img)
    img_std = np.std(img)

    img2 = img-(img_mean + (2*img_std))

    img_mean2 = np.mean(img2)
    img_std2 = np.std(img2)

    img3 = img2/(np.abs(img_mean2) + np.abs(2*img_std2))
    img3 = (np.clip(img3,-1,1)+1)/2

    return img3

def norm_0_to_95(img):

    img_mean = np.mean(img)
    img_std = np.std(img)

    img2 = img-(img_mean + (2*img_std))

    img_mean2 = np.mean(img2)
    img_std2 = np.std(img2)

    img3 = img2/(np.abs(img_mean2) + np.abs(2*img_std2))

    num = 5
    img3 = np.clip(img3,img3.min(),num)
    img3 = norm(img3)

    return img3

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
path_to_dir = r'output\GLS371_testing\t_1\2023-11-22\fluorescent_data'
file_format = '.png'

output_dir = os.path.join(current_path,'output')
# output_dir2 = os.path.join(current_path,'output_median')
os.makedirs(output_dir, exist_ok=True)
# os.makedirs(output_dir2, exist_ok=True)

img_paths = natsorted(glob.glob(os.path.join(path_to_dir,'*' + file_format)))
width, height = imagesize.get(img_paths[0])

for i,this_img in enumerate(tqdm.tqdm(stack)):

    img = cv2.imread(this_img,-1) # or make grayscale with 0 instead of -1


    img2 = norm_0_to_95(img2)
    # img2 = norm(clahe.apply(img2.astype(np.uint8)))
    # # img2 = norm(img2)
    img2 = (img2*255).astype(np.uint8)

    # a = this_img - median_img
    # a[a<0] = 0
    # a = (a/a.max())*255
    # cv2.imwrite(os.path.join(output_dir2,out_name),a.astype(np.uint8))
