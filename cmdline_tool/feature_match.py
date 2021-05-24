import numpy as np
import cv2 as cv
import os
from skimage.measure import compare_ssim
import imutils



def feature_matching(f1,f2, f3):
    i1 = cv.imread(f1)
    i2 = cv.imread(f2)
    img1 = cv.cvtColor(i1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(i2, cv.COLOR_BGR2GRAY)
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    img3 = cv.drawMatches(i1, kp1, i2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(f3, img3)
    return len(matches)


def feature_diff(f1, f2, o1, o2):
    i1 = cv.imread(f1)
    i2 = cv.imread(f2)
    img1 = i1 #cv.cvtColor(i1, cv.COLOR_BGR2GRAY)
    img2 = i2 #cv.cvtColor(i2, cv.COLOR_BGR2GRAY)
    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(img1, img2, full=True, multichannel=True)
    #print(diff)
    # diff = (diff * 255).astype("uint8")
    #print(diff)
    diff = diff[:,:,0] * 100 + diff[:,:,1] * 10 + diff[:,:,2]
    diff = diff.astype("uint8")
    # np.savetxt("diff.txt", diff, fmt='%3d')
    print("SSIM: {}".format(score))

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv.threshold(diff, 0, 255,
        cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    #cv.imwrite("threshold.png", thresh)
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(i1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.rectangle(i2, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # show the output images
    cv.imwrite(o1, i1)
    cv.imwrite(o2, i2)
    return score