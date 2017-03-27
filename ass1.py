import numpy as np
import math
import cv2


class RemoveElementsFromImage:
    def BuildImagePyramid(self, image, n):
        print 'damn'
        print image
        smallerImage = image
        # height=smallerImage.shape[0]
        # width=smallerImage.shape[1]

        pyramid = np.zeros(n)

        while (n > 0):
            image = smallerImage
            smallerImage = np.zeros((image.shape[0] / 2, image.shape[1] / 2), dtype=np.uint8)
            for i in range(smallerImage.shape[0]):
                for j in range(smallerImage.shape[1]):
                    orig_i = 2 * i
                    orig_j = 2 * j
                    averageOfPixels = (int(image[orig_i][orig_j]) + int(image[orig_i + 1][orig_j]) + int(
                        image[orig_i][orig_j + 1]) + int(image[orig_i + 1][orig_j + 1])) / 4
                    smallerImage[i][j] = averageOfPixels
                    # print smallerImage[i][j]
            pyramid[n] = smallerImage
            n = n - 1

        return pyramid

    def ComputeCubicInterpolationCoefficients(self, top, left, i, j, img):
        right_distances = [-1.25, -0.25, 0.75, 1.75]
        left_distances = [-1.75, -0.75, 0.25, 1.25]
        top_distances = [-1.25, -0.25, 0.75, 1.75]
        bottom_distances = [-1.75, -0.75, 0.25, 1.25]

        # compute Cx
        if (left):
            x = left_distances
        else:
            x = right_distances
        if (top):
            y = top_distances
        else:
            y = bottom_distances

        # TODO image bounderies
        bx = [img[math.ceil(i + x[0]), j], img[math.ceil(i + x[1]), j], img[math.ceil(i + x[2]), j],
              img[math.ceil(i + x[3]), j]]
        by = [img[i, math.ceil(j + y[0])], i, img[math.ceil(j + y[1])], i, img[math.ceil(j + y[2])],
              img[i, math.ceil(j + y[3])]]

        cx = np.polyfit(x, bx, 3)
        cy = np.polyfit(y, by, 3)

        m = np.outer(cx, cy)

        s = m.sum()

        return m / float(s)

    def RemoveElements(self, image, colors):
        length, width = cv2.GetSize(image)
        pyramid=self.BuildImagePyramid(image,4)
        for level in range(len(pyramid)):
            img=pyramid[level]
            for i in range(length):
                coffes=np.zeros(4)
                for j in range(width):
                    # top/left??
                    coffes[0]= self.ComputeCubicInterpolationCoefficients(self, True, True, i, j, img)
                    coffes[1] = self.ComputeCubicInterpolationCoefficients(self, True, False, i, j, img)
                    coffes[3] = self.ComputeCubicInterpolationCoefficients(self, False, False, i, j, img)
                    coffes[4] = self.ComputeCubicInterpolationCoefficients(self, True, False, i, j, img)
                    bigger_img=pyramid[level+1]
                    new_i=i*2
                    new_j=j*2
                    bigger_img[new_i,new_j]=coffes[3]
                    bigger_img[new_i+1,new_j]=coffes[0]
                    bigger_img[new_i,new_j+1]=coffes[2]
                    bigger_img[new_i+1,new_j+1]=coffes[1]



if __name__ == '__main__':
# img=cv2.imread('prettyGirl.jpg',0)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# removeElm=RemoveElementsFromImage()
# smaller=removeElm.BuildImagePyramid(img,1)
# print 'finished'
# cv2.imshow('smaller',smaller)
# cv2.waitKey(0)

