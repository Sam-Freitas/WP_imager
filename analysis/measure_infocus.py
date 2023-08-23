import cv2, os, tqdm, glob
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from skimage import filters

plt.ioff()

def imshow_resize(frame_name = "img", frame = 0, resize_size = [640,480]):

    frame = cv2.resize(frame, dsize=resize_size)
    cv2.imshow(frame_name, frame)
    return True

def del_dir_contents(path_to_dir):
    files = glob.glob(os.path.join(path_to_dir,'*'))
    for f in files:
        os.remove(f)

img_file_type = 'jpg'
imgs_dir = os.path.join(os.getcwd(),'captured_images')
imgs = natsorted(glob.glob(os.path.join(imgs_dir,'*.' + img_file_type)))

os.makedirs('captured_transformed_images', exist_ok=True)
del_dir_contents('captured_trasformed_images')

first_img = cv2.imread(imgs[0],-1)

transformed_imgs = []
t = []
t1 = []
t2 = []
t3 = []
t4 = []
t5 = []

posx1y1x2y2 = [500,800,800,1000] # x down y across

for i,file in enumerate(tqdm.tqdm(imgs)):
    
    this_img = cv2.imread(file)[:,:,-1].astype(np.float32) # assume grayscale
    temp_img = this_img
    temp_img = filters.gaussian(temp_img, sigma = 2)

    temp_img5 = filters.roberts(temp_img)
    t5.append(np.max(temp_img5[posx1y1x2y2[0]:posx1y1x2y2[2],posx1y1x2y2[1]:posx1y1x2y2[3]]))
    
    transformed_imgs.append(temp_img5[posx1y1x2y2[0]:posx1y1x2y2[2],posx1y1x2y2[1]:posx1y1x2y2[3]])
    # t.append(np.max(temp_img))

    # print(i)

t_max = np.max(t5)
for i,img in enumerate(tqdm.tqdm(transformed_imgs)):

    norm = (transformed_imgs[i]/t_max)*255

    cv2.imwrite(os.path.join('captured_transformed_images',str(i) + '.jpg'), norm)

plt.figure(1)

plt.subplot(1,2,1)
arg_max5 = np.argmax(t5)
plt.title('Max of roberts filter, img: ' + str(arg_max5))
plt.plot(t5)
plt.plot(arg_max5,t5[arg_max5],'ro')
plt.subplot(1,2,2)
plt.title('most in-focus image -- filtered image')
img = cv2.imread(imgs[arg_max5])
img3 = filters.roberts(filters.gaussian(img[:,:,-1], sigma = 1))
img = cv2.rectangle(img, (posx1y1x2y2[1],posx1y1x2y2[0]),(posx1y1x2y2[3],posx1y1x2y2[2]),(255, 0, 0), 5)
img3 = cv2.cvtColor((255*(img3/np.max(img3))).astype(np.uint8), cv2.COLOR_GRAY2RGB)
img4 = np.concatenate((img,img3),axis = 1)
plt.imshow(img4)

plt.show()
print('EOF')


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