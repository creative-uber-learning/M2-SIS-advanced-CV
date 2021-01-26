import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import cv2
from scipy import ndimage
from skimage.io import imread
from skimage.io import imsave
import pickle
from scipy import sparse
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve, cg


def pause():
    plt.draw()
    plt.pause(0.001)
    raw_input("Press Enter to continue...")


def smoothGaussian(im, sigma):
    N = 4*sigma
    x = np.arange(-N, N+1).astype(float).reshape(-1, 1)
    weights = np.exp(-(x**2)/(2*sigma**2))
    weights = weights/np.sum(weights)
    im2 = scipy.ndimage.convolve(im.astype(float), weights)
    im_smooth = scipy.ndimage.convolve(im2.astype(float), weights.transpose())
    return im_smooth


def gradient(im_smooth):
    gradient_x = scipy.ndimage.convolve(
        im_smooth.astype(float), 0.5*np.array([[1, 0, -1]]))
    gradient_y = scipy.ndimage.convolve(im_smooth.astype(
        float), 0.5*np.array([[1, 0, -1]]).transpose())
    return gradient_x, gradient_y


def smoothedGradient(im, sigma):
    """this function computes a smooth version of the image image using the convolution by a gaussian with standard deviation sigma
    and then compue the gradient in the x and y direction using kernels [1,0,-1] ans its transpose"""
    im_smooth = smoothGaussian(im, sigma)
    im_x, im_y = gradient(im_smooth)
    return im_x, im_y


'''
def HarrisScore(im, sigma1, sigma2, k=0.06):
    """this function compute the harris score for each pixel in the image and return the result as an image"""
    # cornerness score will be stored in this image
    R = np.zeros(im.shape)
    offset = int(sigma1/2)
    y_range = im.shape[0] - offset
    x_range = im.shape[1] - offset
    dx, dy = smoothedGradient(im, sigma1)
    for r in range(offset, y_range):
        for c in range(offset, x_range):
            M = np.zeros((2, 2))
            for i in range(r - offset, r + offset + 1):
                for j in range(c - offset, c + offset + 1):
                    if i == 0 and j == 0:
                        continue
                    # sum up all the Ix and Iy values in the matrix M
                    M[0, 0] += dx[i, j] * dx[i, j]
                    M[0, 1] += dx[i, j]*dy[i, j]
                    M[1, 0] += dx[i, j]*dy[i, j]
                    M[1, 1] += dy[i, j]*dy[i, j]
            # computing lambda values using by singular value decomposition
            u, s, v = np.linalg.svd(M)
            [lmda1, lmda2] = s

            # computing the R value for (r,c) using lambda values
            lambda_product = lmda1*lmda2
            lambda_sum = lmda1 + lmda2
            R[r, c] = lambda_product - k*(lambda_sum**2)
    R = (R > 5*abs(np.mean(R)))*R
    # Non maximum suppression
    for r in range(1, im.shape[0]-1):
        for c in range(1, im.shape[1]-1):
            flag = 0
            for i in [r-1, r+1]:
                for j in [c-1, c+1]:
                    if(R[r, c] < R[i, j]):
                        R[r, c] = 0
                        flag = 1
                        break
                if(flag == 1):
                    break

    (X, Y) = np.where(R != 0)
    return np.array(R[X, Y])
'''


def HarrisScore(im, sigma1, sigma2, k=0.06):
    """this function compute the harris score for each pixel in the image and return the result as an image"""

    R = np.zeros(im.shape)
    offset = int(sigma1/2)
    height = im.shape[0]
    width = im.shape[1]
    dx, dy = smoothedGradient(im, sigma1)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    img_float32 = np.float32(im)
    print("Finding Corners...")
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)
            if r > sigma2:
                R = [x, y]
                img_float32.itemset((y, x), 0)
                img_float32.itemset((y, x), 0)
    return img_float32


def HarrisCorners(im, sigma1, sigma2, k=0.06):
    """this function extract local maximums ion the harris score image that are above 0.005 times the maxium of R and a local miximum in a region a radius 2"""
    R = HarrisScore(im, sigma1=sigma1, sigma2=sigma2, k=0.06)
    import skimage.feature
    peaks = skimage.feature.peak_local_max(R)
    return peaks


def displayPeaks(im, peaks):
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.imshow(im, cmap=plt.cm.Greys_r)
    plt.plot(peaks[:, 1], peaks[:, 0], '.')
    plt.axis((0, im.shape[1], im.shape[0], 0))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.subplots_adjust(left=0.01, right=0.99, top=1, bottom=0)
    plt.show()


def extractPatches(im, points, N):
    """this function extracts patches of size N by N centered around each point privded in the matrix points
    and return a nb_points by N by N 3D array """
    assert(N % 2 == 1)
    radius = (N-1)/2
    patches = np.zeros((points.shape[0], N, N))
    for i, p in enumerate(points):
        if p[0]-radius >= 0 and p[0]+radius < im.shape[0] and p[1]-radius >= 0 and p[1]+radius < im.shape[1]:
            patches[i, :, :] = im[p[0]-radius:p[0] +
                                  radius+1, p[1]-radius: p[1]+radius+1]

    assert(patches.shape[1] == N)
    assert(patches.shape[2] == N)
    return patches


def SSDTable(patches1, patches2):
    """this function computes the sum of square differences between each pair of patches"""
    # TODO implement this function
    dif = patches1.ravel() - patches2.ravel()
    table = np.dot(dif, dif)
    return table


def displayPatch(im, corners, patches):
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.imshow(im, cmap=plt.cm.Greys_r)
    plt.axis((0, im.shape[1], im.shape[0], 0))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.subplots_adjust(left=0.01, right=0.99, top=1, bottom=0)
    plt.plot(corners[:, 1], corners[:, 0], '.')
    p = np.fliplr(plt.ginput(1))
    i = np.argmin(np.sum((p-corners)**2, axis=1))
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.imshow(patches[i, :, :], cmap=plt.cm.Greys_r)


def displayMatches(im1, im2, p1, p2):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im1, cmap=plt.cm.Greys_r)
    for i in range(p1.shape[0]):
        plt.plot([p1[i, 1], p2[i, 1]], [p1[i, 0], p2[i, 0]])
    plt.plot(p1[:, 1], p1[:, 0], '.')
    plt.subplot(1, 2, 2)
    plt.imshow(im2, cmap=plt.cm.Greys_r)
    for i in range(p1.shape[0]):
        plt.plot([p1[i, 1], p2[i, 1]], [p1[i, 0], p2[i, 0]])
    plt.plot(p2[:, 1], p2[:, 0], '.')
    plt.show()


def displayMatches2(im1, im2, p1, p2):
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.imshow(np.row_stack((im1, im2)), cmap=plt.cm.Greys_r)
    for i in range(p1.shape[0]):
        plt.plot([p1[i, 1], p2[i, 1]], [p1[i, 0], p2[i, 0]+im1.shape[0]])
    plt.plot(p1[:, 1], p1[:, 0], '.')
    plt.plot(p2[:, 1], p2[:, 0]+im1.shape[0], '.')
    plt.axis((0, im1.shape[1], im1.shape[0]+im2.shape[0], 0))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.subplots_adjust(left=0.01, right=0.99, top=1, bottom=0)
    plt.show()


def extractMatches(table, threshold=0.7):
    bestscore = np.min(table, axis=1)
    bestmatch1 = np.argmin(table, axis=1)
    bestmatch2 = np.argmin(table, axis=0)
    t = bestmatch2[bestmatch1]
    valid1 = t == np.arange(0, t.size)

    table_copy = table.copy()
    table_copy[np.arange(0, table.shape[0]), bestmatch1] = np.max(table)
    bestscore2 = np.min(table_copy, axis=1)
    valid2 = bestscore < 0.9*bestscore2

    valid = valid1 & valid2
    return np.nonzero(valid)[0], bestmatch1[valid]


def main():
    plt.ion()
    im1 = np.mean(np.array(imread('Image1.jpg')).astype(np.float), axis=2)
    im2 = np.mean(np.array(imread('Image2.jpg')).astype(np.float), axis=2)

    im_x, im_y = smoothedGradient(im1, sigma=2)
    # im_x[200:205,300:302]
    # array([[-0.73459394, -3.24114919],
    # [-0.09954611, -1.66888352],
    # [ 0.18028972, -0.33038019],
    # [ 0.34290241,  0.77111416],
    # [ 0.69410558,  1.81250375]])

    # im_y[200:205,300:302]
    # array([[-5.57636412, -4.37612   ],
    # [-6.52248546, -5.5795648 ],
    # [-7.72358858, -7.04014021],
    # [-7.84553211, -7.21222404],
    # [-6.19201956, -5.42149942]])

    R = HarrisScore(im1, sigma1=2, sigma2=3, k=0.06)
    # R[200:205,300:302]
    # array([[ 241.9235239 ,  261.08986283],
    # [ 201.38571146,  225.27802866],
    # [ 178.14703746,  208.63364944],
    # [ 173.60252361,  211.83960978],
    # [ 184.0906068 ,  231.22448418]])

    plt.imshow(R, cmap=plt.cm.Greys_r)
    # imsave('harris_response.png',(R-R.min())/(np.max(R)-np.min(R)))

    corners1 = HarrisCorners(im1, sigma1=2, sigma2=3, k=0.06)
    # corners1[:10,:]
    # array([[ 46, 223],
    # [ 51, 216],
    # [ 56, 166],
    # [ 56, 175],
    # [ 56, 316],
    # [ 61, 159],
    # [ 63, 250],
    # [ 65, 256],
    # [ 67, 303],
    # [ 69, 283]])
    displayPeaks(im1, corners1)

    corners2 = HarrisCorners(im2, sigma1=2, sigma2=3, k=0.06)
    # corners2[:10,:]
    # array([[ 56,  64],
    # [ 67,  51],
    # [ 68,  28],
    # [ 70, 216],
    # [ 71, 166],
    # [ 73, 186],
    # [ 73, 641],
    # [ 74, 107],
    # [ 74, 115],
    # [ 77, 240]])
    displayPeaks(im2, corners2)

    N = 21
    patches1 = extractPatches(im1, corners1, N)
    # displayPatch(im1,corners1,patches1)

    extractPatches(im1, np.array([[150, 300], [200, 270]]), 3)
    # array([[[ 37.66666667,  41.66666667,  44.33333333],
    # [ 36.33333333,  38.66666667,  40.66666667],
    # [ 37.33333333,  38.        ,  38.66666667]],

    # [[ 75.66666667,  82.66666667,  79.        ],
    # [ 70.66666667,  73.33333333,  73.        ],
    # [ 91.33333333,  92.33333333,  91.33333333]]])

    patches2 = extractPatches(im2, corners2, N)

    t1 = np.arange(0, 4*5*5).reshape(4, 5, 5)
    t2 = t1-5
    SSDTable(t1, t2)
    # array([[    625.,   10000.,   50625.,  122500.],
    # [  22500.,     625.,   10000.,   50625.],
    # [  75625.,   22500.,     625.,   10000.],
    # [ 160000.,   75625.,   22500.,     625.]])

    table = SSDTable(patches1, patches2)

    matches1, matches2 = extractMatches(table, threshold=0.7)
    p1 = corners1[matches1, :]
    p2 = corners2[matches2, :]
    displayMatches(im1, im2, p1, p2)
    plt.ioff()
    displayMatches2(im1, im2, p1, p2)


if __name__ == "__main__":
    main()
