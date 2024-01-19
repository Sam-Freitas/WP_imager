import cv2, os, tqdm, glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted
from skimage import filters, morphology, measure, segmentation
import scipy

def get_terasaki_positions():

    print('Getting Terasaki positions')
    path = os.path.join(r'Y:\Users\Sam Freitas\WP_imager\settings','settings_terasaki_positions.csv')
    df = pd.read_csv(path, delimiter = ',',index_col=False)
    df = df.to_dict()

    return df

plt.ioff()

def imshow(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

def imshow_resize(frame, frame_name="img", resize_size=[640, 640]):
    frame = cv2.resize(frame, dsize=resize_size)
    cv2.imshow(frame_name, frame)
    return True

def del_dir_contents(path_to_dir):
    files = glob.glob(os.path.join(path_to_dir, "*"))
    for f in files:
        os.remove(f)

def bwareafilt(mask, n=1, area_range=(0, np.inf)):
    """Extract objects from binary image by size """
    # For openCV > 3.0 this can be changed to: areas_num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    labels = measure.label(mask.astype('uint8'), background=0)
    area_idx = np.arange(1, np.max(labels) + 1)
    areas = np.array([np.sum(labels == i) for i in area_idx])
    inside_range_idx = np.logical_and(areas >= area_range[0], areas <= area_range[1])
    area_idx = area_idx[inside_range_idx]
    areas = areas[inside_range_idx]
    keep_idx = area_idx[np.argsort(areas)[::-1][0:n]]
    kept_areas = areas[np.argsort(areas)[::-1][0:n]]
    if np.size(kept_areas) == 0:
        kept_areas = np.array([0])
    if n == 1:
        kept_areas = kept_areas[0]
    kept_mask = np.isin(labels, keep_idx)
    return kept_mask, kept_areas


s_terasaki_positions = get_terasaki_positions()
t_names = list(s_terasaki_positions['name'].values())

img_file_type = "png"
imgs_dir = r"output\autofocus_calibration"  # os.path.join(os.getcwd(),'captured_images')
imgs = natsorted(glob.glob(os.path.join(imgs_dir, "*." + img_file_type)))

output_dir = (
    os.path.dirname(os.path.realpath(__file__))
)
# os.makedirs(output_dir, exist_ok=True)

first_img = cv2.imread(imgs[0], -1)

img_groups = np.reshape(np.asarray(imgs),(-1,33)) # by rows

# group_results_mean = np.zeros(shape = img_groups.shape)
group_results_max = np.zeros(shape = img_groups.shape)
# group_results_min = np.zeros(shape = img_groups.shape)

well_size_pixels_approx = 1000
grid_size = 9

for i in range(img_groups.shape[0]):

    print(t_names[i])
    print(i)

    idx = np.char.find(img_groups[:,0],t_names[i])>0
    group = img_groups[idx,:].squeeze()

    # imgs = np.zeros(shape = (first_img.shape[0],1))

    for j,this_img in enumerate(tqdm.tqdm(group)):

        img = cv2.imread(this_img, -1).astype(np.float32)

        ret,thresh = cv2.threshold((filters.gaussian(cv2.threshold(img,img.mean(),255,cv2.THRESH_BINARY)[1],5)>0).astype(np.float32),0,255,cv2.THRESH_BINARY)
        thresh, areas = bwareafilt(thresh,1)

        M = cv2.moments(thresh.astype(np.float32))
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        arrays = np.meshgrid(np.linspace(cX-well_size_pixels_approx,cX+well_size_pixels_approx,grid_size),
                            np.linspace(cY-well_size_pixels_approx,cY+well_size_pixels_approx,grid_size))

        LoG_img = scipy.ndimage.gaussian_laplace(img,5)

        center_measurement = LoG_img[arrays[0].ravel().astype(np.int64),arrays[1].ravel().astype(np.int64)]

        # group_results_mean[i,j] = LoG_img.mean() 
        group_results_max[i,j] =  center_measurement.max() #######################LoG_img.max()
        # group_results_min[i,j] =  LoG_img.min()

    # if i > 10:
    #     break


temp = np.zeros(img_groups.shape[1])
counter = 1
plt.figure(figsize = [10,10])
for k in range(img_groups.shape[0]):
    a = np.max(group_results_max[k,:])
    if a > 0:
        plt.plot(group_results_max[k,:]/a)
        temp += group_results_max[k,:]/a
        counter += 1
plt.xticks(ticks = np.arange(group_results_max[k,:].shape[0]), fontsize = 6)
plt.grid(axis='x', color='0.95')
plt.plot(np.arange(img_groups.shape[1]), temp/counter, c='k', linewidth = 5)
plt.savefig(os.path.join(output_dir,'focus_measurements.png'))
plt.show()


# temp_img1 = filters.sobel(temp_img)
# t1.append(np.max(temp_img1[400:800,800:1000]))

# temp_img2 = filters.scharr(temp_img)
# t2.append(np.max(temp_img2[400:800,800:1000]))

# temp_img3 = filters.prewitt(temp_img)
# t3.append(np.max(temp_img3[400:800,800:1000]))

# temp_img4 = filters.farid(temp_img)
# t4.append(np.max(temp_img4[400:800,800:1000]))

# plt.subplot(5,2,1)
# arg_max1 = np.argmax(t1)
# plt.title('Max of sobel filter, img: ' + str(arg_max1))
# plt.plot(t1)
# plt.plot(arg_max1,t1[arg_max1],'ro')
# plt.subplot(5,2,2)
# plt.imshow(cv2.imread(imgs[arg_max1])[:,:,-1])

# plt.subplot(5,2,3)
# arg_max2 = np.argmax(t2)
# plt.title('Max of scharr filter, img: ' + str(arg_max2))
# plt.plot(t2)
# plt.plot(arg_max2,t2[arg_max2],'ro')
# plt.subplot(5,2,4)
# plt.imshow(cv2.imread(imgs[arg_max2])[:,:,-1])

# plt.subplot(5,2,5)
# arg_max3 = np.argmax(t3)
# plt.title('Max of prewitt filter, img: ' + str(arg_max3))
# plt.plot(t3)
# plt.plot(arg_max3,t3[arg_max3],'ro')
# plt.subplot(5,2,6)
# plt.imshow(cv2.imread(imgs[arg_max3])[:,:,-1])

# plt.subplot(5,2,7)
# arg_max4 = np.argmax(t4)
# plt.title('Max of farid filter, img: ' + str(arg_max4))
# plt.plot(t4)
# plt.plot(arg_max4,t4[arg_max4],'ro')
# plt.subplot(5,2,8)
# plt.imshow(cv2.imread(imgs[arg_max4])[:,:,-1])
