#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy import ndimage
from skimage.io import imread
from skimage.io import imsave
import pickle
from scipy import sparse
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve, cg
from xorshift import *
from numpy import sin, cos, ceil, floor
#import cv2
from math import sin, cos, radians, pi
from numba import njit, prange


def pause():
    plt.draw()
    plt.pause(0.001)
    raw_input("Press Enter to continue...")


def plotLine(rho, theta, xmin, xmax, ymin, ymax, color='b'):
    """display a line given its angle and its distance to the origin"""
    points = []
    for px in [xmin, xmax]:
        py = (rho-px*np.cos(theta))/np.sin(theta)
        if py > ymin and py < ymax:
            points.append([px, py])

    for py in [ymin, ymax]:
        px = (rho-py*np.sin(theta))/np.cos(theta)
        if px > xmin and px < xmax:
            points.append([px, py])
    points = np.array(points)
    distancePointsLine(points, rho, theta)
    plt.plot(points[:, 0], points[:, 1], color)


def distancePointsLine(points, rho, theta):
    """this function computes the set of euclidian distances between a set of points and a line 
    parameterized with the angle theta (between the x axis and the vector joining the origin 
    and the projection of the origin on the line) and the distance to the origin rho"""
    # https://stats.stackexchange.com/questions/71614/distance-measure-of-angles-between-two-vectors-taking-magnitude-into-account
    distances = np.abs(points[:, 0] * np.cos(theta) +
                       points[:, 1] * np.sin(theta) - rho)
    return distances


@njit(parallel=True)
def parallel_nonzero_count(arr):
    flattened = arr.ravel()
    sum_ = 0
    for i in prange(flattened.size):
        sum_ += flattened[i] != 0
    return sum_


def countInliersLine(points, rho, theta, tau):
    """this function compute the number of inliers for a line given a distance threshold tau"""
    distances = distancePointsLine(points, rho, theta)
    inliers_count = parallel_nonzero_count(distances < tau)
    return inliers_count


def h(x, tau):
    """this function computes the smooth function h given in the slides that goes from 1 at location 0 to 0 at location tau"""
    # TODO implement this function, code it for x a vector
    # you can use if/else with a loop over the elements of the vector , or
    # alternatively use a vectorized formulation by creating first a boolean vector b=x<tau that you then multiply by the polynome

    return hx


def smoothLineScore(points, rho, theta, tau):
    """this function implements the smoothed score function of a line"""
    # TODO implement this function

    return smooth_score


def getRhos(points, theta):
    """given an angle theta, this function computes the signed distance rho of the line with angle
    theta passing through each point"""
    # TODO implement this function

    return rhos


def rhoIdsFromRhos(rhos, min_rho, max_rho, nb_rhos):
    """given rhos, this function get the index location of these rhos in the discrete rho grid ,
    instead of returning an integer index of the nearest rho in rho_grid, it provides a float number
    such that the user of the function can decide to use round, floor or ceil afterward"""
    delta_rho = (max_rho-min_rho)/(nb_rhos-1)
    return (rhos-min_rho)/delta_rho


def scoresBruteForce(points, tau, nb_thetas, min_rho, max_rho, nb_rhos):
    """this function loops over all angle/rho pairs and fill the score array using the inliers count"""

    rho_grid, theta_grid = getRhoAndThetaGrid(
        nb_thetas, min_rho, max_rho, nb_rhos)
    # TODO finish coding this function

    return scores


def scoresSmoothBruteForce(points, tau, nb_thetas, min_rho, max_rho, nb_rhos):
    """this function loops over all angle/rho pairs and fill the score array using the smoothed score,
    the column corresponds to theta , the line to rho"""
    rho_grid, theta_grid = getRhoAndThetaGrid(
        nb_thetas, min_rho, max_rho, nb_rhos)

    # TODO finish coding this function

    # if you enjoy using numpy broadcasting you can avoid loops...but you should probably keep that for the end

    return scores


def getRhoAndThetaGrid(nb_thetas, min_rho, max_rho, nb_rhos):
    theta_grid = np.pi*np.arange(0, nb_thetas)/nb_thetas
    # we de not use linspace as we do not want theta_grid[0] to be equal to theta_grid[-1]
    # otherwise the last column would identical to the first column flipped upside-down
    rho_grid = np.linspace(min_rho, max_rho, nb_rhos)
    assert(np.max(np.abs(rhoIdsFromRhos(rho_grid, min_rho,
                                        max_rho, nb_rhos)-np.arange(nb_rhos))) < 1e-10)
    return rho_grid, theta_grid


def scoresHough(points, tau, nb_thetas, min_rho, max_rho, nb_rhos):
    """this function compute the same score table as the scoresBruteForce function (inliers count)
    but avoid looping over all rhos for each theta by computing an interval of valid rhos (see slides)"""

    rho_grid, theta_grid = getRhoAndThetaGrid(
        nb_thetas, min_rho, max_rho, nb_rhos)
    # TODO: implement this function, you can call getRhos
    # avoiding loops seems quite difficult here so go for loops
    # be careful with the "range" function the range(m,n) will give number ranging from m to n-1 and not to n

    return scores


def scoresHoughSmooth(points, tau, nb_thetas, min_rho, max_rho, nb_rhos):
    """this function compute the same score table as the scoresSmoothBruteForce function 
    but avoid looping over all rhos for each theta by computing an interval of valid rhos"""
    rho_grid, theta_grid = getRhoAndThetaGrid(
        nb_thetas, min_rho, max_rho, nb_rhos)

    # TODO: implement this function, you can call getRhos  and h

    return scores


def findPeaks(scores, nb_thetas, min_rho, max_rho, nb_rhos, display_peaks=False, threshold_rel=0.8):
    # we extend the scores array to an range of angle from 0 to 2*pi, which will help in
    # dealing with the border of the image on the left and right boundaries of the image
    # when get the local maxima
    rho_grid, theta_grid = getRhoAndThetaGrid(
        nb_thetas, min_rho, max_rho, nb_rhos)
    # the last columns is equal to the first column
    scores_extended = np.column_stack((scores, np.flipud(scores)))
    theta_grid_extended = np.hstack((theta_grid, theta_grid+np.pi))
    import skimage.feature
    peaks = skimage.feature.peak.peak_local_max(
        scores_extended, threshold_rel=0.8, min_distance=2)

    if display_peaks:
        plt.figure()
        plt.imshow(scores_extended, cmap=plt.cm.Greys_r)
        plt.plot(peaks[:, 1], peaks[:, 0], '.')
        plt.axis((0, scores_extended.shape[1], 0, scores_extended.shape[0]))
    peak_rhos = rho_grid[peaks[:, 0]]
    peak_thetas = theta_grid_extended[peaks[:, 1]]
    return peak_rhos, peak_thetas


def displayResult(points, peak_rhos, peak_thetas, tau):
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1])
    xmin, xmax, ymin, ymax = (min(points[:, 0]), max(
        points[:, 0]), min(points[:, 1]), max(points[:, 1]))
    for i in range(len(peak_rhos)):
        plotLine(peak_rhos[i], peak_thetas[i], xmin, xmax, ymin, ymax, 'b')
        plotLine(peak_rhos[i]-tau, peak_thetas[i],
                 xmin, xmax, ymin, ymax, 'b:')
        plotLine(peak_rhos[i]+tau, peak_thetas[i],
                 xmin, xmax, ymin, ymax, 'b:')
    plt.axis((xmin, xmax, ymin, ymax))


def main():
    plt.ion()

    with open('tp4_hough.pkl', 'rb') as f:
        points,\
            scores_brute_force,\
            scores_hough,\
            peak_rhos, peak_thetas,\
            scores_smoothed_brute_force,\
            scores_hough_smooth,\
            peak_rhos_smooth,\
            peak_thetas_smooth = pickle.load(f)

    n = 100
    sigma = 0.03
    random = xorshift()  # use custom pseudo reandom to get repeatable results
    points1 = random.rand(
        n, 1)*np.array([0, 4])+np.array([2, 0])+random.normal(0, sigma, n, 2)
    points2 = random.rand(
        n, 1)*np.array([4, 4])+np.array([0, 0])+random.normal(0, sigma, n, 2)
    points3 = random.rand(n, 2)*np.array([4, 4])

    points = np.row_stack((points1, points2, points3))
    plt.scatter(points[:, 0], points[:, 1])
    plt.axis('equal')
    plt.draw()
    plt.show()

    min_theta = 0
    max_theta = np.pi
    max_rho = np.max(np.linalg.norm(points, axis=1))
    min_rho = -max_rho

    nb_rhos = 100
    nb_thetas = 150
    tau = 10*sigma

    distancePointsLine(points[1:5, :], 2, 0.3)
    #array([ 0.02298727,  0.60614286,  0.87389925,  0.04036134])
    countInliersLine(points,  2, 0.3, tau)  # 68
    h(np.linspace(0, 1, 11), 0.5)
    # array([ 1.      ,  0.884736,  0.592704,  0.262144,  0.046656,  0.      ,
    #   -0.      , -0.      , -0.      , -0.      , -0.      ])
    smoothLineScore(points, 2, 0.3, tau)  # 35.398607765521561
    getRhos(points[1:5, :], 0.3)
    #array([ 2.02298727,  2.60614286,  2.87389925,  2.04036134])

    # computes the lines score table using the brute force method from slide 7
    scores_brute_force = scoresBruteForce(
        points, tau, nb_thetas, min_rho, max_rho, nb_rhos)
    plt.figure()
    plt.imshow(scores_brute_force, cmap=plt.cm.Greys_r)
    plt.show()
    plt.title('brute force score table')
    pause()

    # computes the line score table using the hough transform approach that do not use the slide sum from slide 10
    scores_hough = scoresHough(
        points, tau, nb_thetas, min_rho, max_rho, nb_rhos)
    plt.figure()
    plt.imshow(scores_hough, cmap=plt.cm.Greys_r)
    plt.show()
    plt.title('hough based  score table')
    pause()

    # the difference should be less than 1e-5
    print('difference between the two approaches : ' +
          str(np.max(np.abs(scores_hough-scores_brute_force))))

    peak_rhos, peak_thetas = findPeaks(
        scores_hough, nb_thetas, min_rho, max_rho, nb_rhos, display_peaks=True, threshold_rel=0.5)

    displayResult(points, peak_rhos, peak_thetas, tau)
    pause()

    scores_smoothed_brute_force = scoresSmoothBruteForce(
        points, tau, nb_thetas, min_rho, max_rho, nb_rhos)
    plt.figure()
    plt.imshow(scores_smoothed_brute_force, cmap=plt.cm.Greys_r)
    plt.show()
    plt.title('smoothed brute force score table')
    pause()

    scores_hough_smooth = scoresHoughSmooth(
        points, tau, nb_thetas, min_rho, max_rho, nb_rhos)
    plt.figure()
    plt.imshow(scores_hough_smooth, cmap=plt.cm.Greys_r)
    plt.show()
    plt.title('smoothed hough based  score table')
    pause()

    peak_rhos_smooth, peak_thetas_smooth = findPeaks(
        scores_hough_smooth, nb_thetas, min_rho, max_rho, nb_rhos, display_peaks=True)
    plt.ioff()
    displayResult(points, peak_rhos_smooth, peak_thetas_smooth, tau)
    pause()

    with open('tp4_hough.pkl', 'wb') as f:
        l = [points.astype(np.float16),
             scores_brute_force.astype(np.uint8),
             scores_hough.astype(np.uint8),
             peak_rhos, peak_thetas,
             scores_smoothed_brute_force.astype(np.float16),
             scores_hough_smooth.astype(np.float16),
             peak_rhos_smooth,
             peak_thetas_smooth]
        pickle.dump(l, f)


if __name__ == "__main__":
    main()
