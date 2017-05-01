import cv2
import numpy as np
import math

def ComputeImageHistogram(img):
    height, width = img.shape
    histogram = np.zeros(256)

    for i in range(height):
        for j in range(width):
            intensity = img[i,j]
            histogram[intensity] += 1



    # do we need to normalize?
    # image_size = height*width
    # for i in range(len(histogram)):
    #     histogram[i] = histogram[i] / image_size

    return histogram


def HistogramEqulization(in_hist):
    trans = np.zeros(256)
    eq_hist = np.zeros(256)
    cdf = np.zeros(256)

    for k in range(len(in_hist)):
        for l in range(k+1):
            cdf[k] += in_hist[l]

    cdf_min = np.amin(cdf)

    for b in range(len(eq_hist)):
        trans[b] = math.ceil((cdf[b] - cdf_min) * (255/float(image_size)))-1

    for c in range(len(trans)):
        index = int(trans[c])
        eq_hist[index] += in_hist[c]

    return eq_hist, trans


def MatchHistogram(src_hist, dst_hist):
    eq_src_hist, src_trans = HistogramEqulization(src_hist)
    eq_dst_hist, dst_trans = HistogramEqulization(dst_hist)
    inverse_trans = np.zeros(256, dtype = np.uint8)
    closest_index = 0

    for m in range(len(src_trans)):
        min_diff = np.inf
        for n in range(len(dst_trans)):
            diff = abs(src_trans[m]-dst_trans[n])
            if diff < min_diff:
                min_diff = diff
                closest_index = n
        inverse_trans[m] = dst_trans[closest_index]

    return inverse_trans

def EqulizeHistogram(src_img):
    height2, width2 = src_img.shape
    src__hist = ComputeImageHistogram(src_img)
    dst__hist, trans = HistogramEqulization(src__hist)
    dst_img = np.zeros((height2, width2), dtype = np.uint8)

    for f in range(height2):
        for g in range(width2):
            dst_img[f, g] = int(trans[src_img[f, g]])

    return dst_img


def MatchHistogram1(src_img, hist):
    height1, width1 = src_img.shape
    dst_img = np.zeros((height1, width1), dtype=np.uint8)

    src___hist = ComputeImageHistogram(src_img)
    inverse_trans = MatchHistogram(src___hist, hist)

    for f in range(height1):
        for g in range(width1):
            dst_img[f, g] = int(inverse_trans[src_img[f, g]])

    return dst_img

# tests

if __name__ == '__main__':
    global image_size

    img = cv2.imread('example.jpg', 0)
    height, width = img.shape
    image_size = height * width
    print 'showing image',img
    cv2.imshow('original image', img)
    hist = ComputeImageHistogram(img)
    eq_hist, trans = HistogramEqulization(hist)
    new_img = MatchHistogram1(img, eq_hist)
    cv2.imshow('new image', new_img)
    cv2.waitKey(0)
