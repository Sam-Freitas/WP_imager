from fastsam import FastSAM, FastSAMPrompt
from natsort import natsorted
import os,glob,tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cv2 import imread, imwrite
from skimage.measure import regionprops, regionprops_table #, label
from skimage.filters import gaussian

from utils.tools import imclearborder, overlay_masks_on_image

def del_dir_contents(path_to_dir):
    files = glob.glob(os.path.join(path_to_dir,'*'))
    for f in files:
        os.remove(f)

def combine_two_dicts(d1,d2):

    d3 = d1.copy()

    for each_key in list(d1.keys()):

        a = np.concatenate((d1[each_key], d2[each_key]))
        d3[each_key] = a

    return d3

imgs_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data\SICKO_testing_dataset_png"
imgs = list(natsorted(glob.glob(os.path.join(imgs_path,'*.png'))))
imgs_path = r"C:\Users\LabPC2\Documents\GitHub\python_to_GRBL\captured_images"
imgs = list(natsorted(glob.glob(os.path.join(imgs_path,'*.jpg'))))

output_path = './output/'
os.makedirs(output_path,exist_ok=True)
del_dir_contents(output_path)

model = FastSAM('FastSAM-x.pt')

region_propert_table = []
min_area = 1000

for i,each_img in enumerate(tqdm.tqdm(imgs)):
    IMAGE_PATH = each_img
    DEVICE = 'cuda'

    image = imread(each_img)
    # image = gaussian(image,sigma = 1,truncate = 2,channel_axis=-1)

    everything_results = model(image, device=DEVICE, retina_masks=True, imgsz=640, conf=0.05, iou=0.01,)
    prompt_process = FastSAMPrompt(image, everything_results, device=DEVICE)
    # everything prompt
    ann = prompt_process.everything_prompt()

    idx = []
    if len(ann) > 0:
        for j,each_ann in enumerate(ann.data):
            temp = imclearborder(np.asarray(each_ann.cpu()).astype(np.uint8),5)
            if (np.asarray(each_ann.sum().cpu()) == np.sum(temp)) and (np.sum(temp) > min_area):
                idx.append(j)

        ann = ann[idx]

        if ann.shape[0]>0:
            labeled_image, mask = overlay_masks_on_image(ann,image, individual=True)
            props = regionprops_table(mask.astype(np.int64), properties=['area','area_convex','perimeter','bbox'])

            if i==0:
                region_propert_table = props.copy()
            else:
                region_propert_table = combine_two_dicts(region_propert_table,props)
            imwrite('./output/' + str(i) + 'e.jpg', labeled_image[:,:,::-1])

            # props = skimage.measure.regionprops_table(mask)
            # region_propert_table.append(props)
    #     else:
    #         labeled_image = image   
    #         props = []
    # else:
    #     labeled_image = image   

    # imwrite('./output/' + str(i) + 'e.jpg', labeled_image[:,:,::-1])

    if i%25 == 0:
        out = pd.DataFrame(region_propert_table)
        out.to_csv('out.csv')
    # prompt_process.plot(annotations=ann,output_path='./output/' + str(i) + 'e.jpg',)

out = pd.DataFrame(region_propert_table)
out.to_csv('out.csv')