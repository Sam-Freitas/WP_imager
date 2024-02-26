import numpy as np
import scipy
import matplotlib.pyplot as plt
import os, time
import analysis.fluor_postprocess


def sq_grad(img,thresh = 50,offset = 10):

    img = analysis.fluor_postprocess.norm(img.astype(np.float32))
    thresh = thresh/255

    shift = int(0-offset)
    offset = int(offset)

    img1 = img[:,0:shift].astype(np.float32)
    img2 = img[:,offset:].astype(np.float32)

    diff = np.abs(img2-img1)
    mask = diff > thresh
    squared_gradient = diff*diff*mask

    return squared_gradient

def p(x):

    plt.plot(x)
    sigma = 1
    filt = scipy.ndimage.gaussian_filter1d(x,sigma)
    plt.plot(filt)
    extrema = scipy.signal.argrelextrema(filt, np.greater)
    extrema_minama = scipy.signal.argrelextrema(filt,np.less)
    for e in extrema:
        plt.plot(e,filt[e],'go')
    for e_n in extrema_minama:
        plt.plot(e_n,filt[e_n],'ro')
    print(extrema)

images = np.load('autofocus_stack.npy')
test_signal = np.random.random(9)

offsets = list(range(0,50,10))
threshes = list(range(0,25,5))

plt.figure()
plt.subplot(2,len(offsets),1)
plt.title('test signal')
p(test_signal)

for i,val in enumerate(zip(offsets,threshes)):
    # offset = 5*(i+1) # this is for the autofocus algorithm how many pixels apart is the focus to be measures
    # thresh = 5*(i+1) # same as above but now ignores all the values under thresh

    offset = val[0]+1 # this is for the autofocus algorithm how many pixels apart is the focus to be measures
    thresh = 5 #val[1]+1 # same as above but now ignores all the values under thresh

    uncalib_fscore = []

    for img in images:
        temp = sq_grad(img,thresh = thresh,offset = offset)
        uncalib_fscore.append(np.sum(temp))

    plt.subplot(2,len(offsets),i+1+len(offsets))
    plt.title("Offset: %s ---- Thresh: %s" % (offset, thresh))
    print("Offset: %s ---- Thresh: %s" % (offset, thresh))
    p(uncalib_fscore)
plt.show()

print('eof')