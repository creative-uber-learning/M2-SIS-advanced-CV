import os
import sys
import time
from marking_tools import *

from tp3_sujet import *


def test():
    plt.ion()

    im1 = np.mean(np.array(imread('Image1.png')).astype(np.float), axis=2)
    im2 = np.mean(np.array(imread('Image2.png')).astype(np.float), axis=2)

    m = marking()
    im_x, im_y = smoothedGradient(im1, sigma=2)
    m.add(1, im_x[200:205, 300:302], [[-0.67794561, -3.19281727],
                                      [-0.03868455, -1.62560241],
                                      [0.22736612, -0.30635884],
                                      [0.36763735,  0.77305092],
                                      [0.70369259,  1.80483468]])

    m.add(1, im_y[200:205, 300:302], [[-5.56381571, -4.35629625],
                                      [-6.46047, -5.52624641],
                                      [-7.64421368, -6.98094653],
                                      [-7.79410418, -7.17904077],
                                      [-6.19673306, -5.43011379]])

    R = HarrisScore(im1, sigma1=2, sigma2=3, k=0.06)
    m.add(2, R[200:205, 300:302], [[239.81975235,  259.09744995],
                                   [199.59635893,  223.4601744],
                                   [176.86886897,  207.17943751],
                                   [172.95245279,  210.87816299],
                                   [184.14220583,  230.85600882]])

    plt.imshow(R, cmap=plt.cm.Greys_r)
    # imsave('harris_response.png',(R-R.min())/(np.max(R)-np.min(R)))

    corners1 = HarrisCorners(im1, sigma1=2, sigma2=3, k=0.06)
    m.add(2, corners1[:10, :], [[46, 223],
                                [51, 216],
                                [56, 166],
                                [56, 175],
                                [56, 316],
                                [61, 159],
                                [63, 250],
                                [65, 256],
                                [67, 303],
                                [69, 283]])

    displayPeaks(im1, corners1)

    corners2 = HarrisCorners(im2, sigma1=2, sigma2=3, k=0.06)
    # corners2[:10,:]
    # array([[ 56,  64],
    #[ 67,  51],
    #[ 68,  28],
    #[ 70, 216],
    #[ 71, 166],
    #[ 73, 186],
    #[ 73, 641],
    #[ 74, 107],
    #[ 74, 115],
    # [ 77, 240]])
    displayPeaks(im2, corners2)

    N = 21
    patches1 = extractPatches(im1, corners1, N)
    # displayPatch(im1,corners1,patches1)

    patch_test = extractPatches(im1, np.array([[150, 300], [200, 270]]), 3)

    # m.add(0,patch_test,[[[ 37.66666667,  41.66666667,  44.33333333],
    #[ 35.66666667,  38.66666667,  40.66666667],
    # [ 37.33333333,  38.        ,  38.66666667]],
    # [[ 76.        ,  83.        ,  79.        ],
    #[ 70.66666667,  73.33333333,  74.66666667],
    # [ 91.33333333,  92.33333333,  92.33333333]]])

    patches2 = extractPatches(im2, corners2, N)

    t1 = np.arange(0, 4*5*5).reshape(4, 5, 5)
    t2 = t1-5
    tab = SSDTable(t1, t2)

    m.add(2, tab, [[625.,   10000.,   50625.,  122500.],
                   [22500.,     625.,   10000.,   50625.],
                   [75625.,   22500.,     625.,   10000.],
                   [160000.,   75625.,   22500.,     625.]])

    t1 = patches1[[1, 20, 50, 60], :, :]
    t2 = patches2[[1, 20, 50, 60], :, :]
    tab = NCCTable(t1, t2)

    m.add(2, tab, [[0.52881566,  0.87617929,  0.69917479,  0.81774953],
                   [0.74317605,  0.98377706,  0.72340192,  1.09927029],
                   [0.32002431,  0.54738743,  0.25827641,  1.06443083],
                   [0.48417473,  0.74272221,  0.68854655,  1.12421821]])

    table = SSDTable(patches1, patches2)

    m.print_note()

    matches1, matches2 = extractMatches(table, threshold=0.7)
    p1 = corners1[matches1, :]
    p2 = corners2[matches2, :]
    displayMatches(im1, im2, p1, p2)
    plt.ioff()
    displayMatches2(im1, im2, p1, p2)


if __name__ == "__main__":
    test()
