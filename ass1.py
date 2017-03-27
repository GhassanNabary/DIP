import numpy as np
import math
import cv2


class RemoveElementsFromImage:
    def BuildImagePyramid(self, image, n):
        smallerImage = image
        rows, columns = image.shape
        pyramid = []

        while (n > 0):
            image = smallerImage
            smallerImage = np.zeros((image.shape[0] / 2, image.shape[1] / 2), dtype=np.uint8)
            pyramid.append(smallerImage)

            for i in range(smallerImage.shape[0]):
                for j in range(smallerImage.shape[1]):
                    orig_i = 2 * i
                    orig_j = 2 * j
                    averageOfPixels = (int(image[orig_i][orig_j]) + int(image[orig_i + 1][orig_j]) + int(
                        image[orig_i][orig_j + 1]) + int(image[orig_i + 1][orig_j + 1])) / 4
                    smallerImage[i][j] = averageOfPixels

            n = n - 1
        pyramid.reverse()
        return pyramid

    def ComputeCubicInterpolationCoefficients(self, top, left, i, j, image):

        right_distances = [-1.25, -0.25, 0.75, 1.75]
        left_distances = [-1.75, -0.75, 0.25, 1.25]
        top_distances = [-1.25, -0.25, 0.75, 1.75]
        bottom_distances = [-1.75, -0.75, 0.25, 1.25]

        if (left):
            x = left_distances
        else:
            x = right_distances
        if (top):
            y = top_distances
        else:
            y = bottom_distances

        rows, columns = image.shape

        bx = [img[int(np.clip(math.ceil(i + x[0]),0,rows)), j], img[int(np.clip(math.ceil(i + x[1]),0,rows)), j], img[int(np.clip(math.ceil(i + x[2]),0,rows)), j],img[int(np.clip(math.ceil(i + x[3]),0,rows)), j]]
        by = [img[i, int(np.clip(math.ceil(j + y[0]),0,columns))], img[i,int(np.clip(math.ceil(j + y[1]),0,columns))], img[i,int(np.clip(math.ceil(j + y[2]),0,columns))],img[i, int(np.clip(math.ceil(j + y[3]),0,columns))]]

        cx = np.polyfit(x, bx, 3)
        cy = np.polyfit(y, by, 3)

        m = np.outer(cx, cy)

        s = m.sum()

        return m / float(s)


    def sum_mul(self, mat1,mat2):
        sum = 0
        for k in range(len(mat1)):
            for h in range(len(mat2)):
                sum += mat1[k][h] * mat2[k][h]

        print 'wiiiiiiinnnnnnnnnnnnnnnnnnnnnndoooooooooooooow', sum

        return sum


    def RemoveElements(self, image, colors):
        rows, columns = image.shape
        pyramid = self.BuildImagePyramid(image, 200)
        bigger_img = image
        for level in range(len(pyramid)-1):

            img = pyramid[level]
            rows, columns = img.shape
            for i in range(rows):
                coffes = np.zeros(4, dtype=object)
                for j in range(columns):
                  #  print 'Intensity',img[i, j]
                    if img[i, j] in colors:
                     #   print 'here',i,j,level
                        coffes[0] = self.ComputeCubicInterpolationCoefficients(True, True, i, j, img)
                        coffes[1] = self.ComputeCubicInterpolationCoefficients(True, False, i, j, img)
                        coffes[2] = self.ComputeCubicInterpolationCoefficients(False, False, i, j, img)
                        coffes[3] = self.ComputeCubicInterpolationCoefficients(False, True, i, j, img)
                        # sum of multiplications

                        # dealing with image bounderies
                        if i < 2:
                            i = 2
                        elif i > pyramid[level].shape[0] - 4:
                            i = pyramid[level].shape[0] - 4
                        if j < 2:
                            j = 2
                        elif j > pyramid[level].shape[1] - 4:
                            j = pyramid[level].shape[1] - 4

                        print 'wiiiiiiinnnnnnnnnnnnnnnnnnnnnndoooooooooooooow', coffes[0]

                        sum0 = self.sum_mul(coffes[0], img[i - 2:i + 2, j - 1:j + 3])

                        sum1 = self.sum_mul(coffes[1], img[i - 1:i + 3, j - 1:j + 3])

                        sum2 = self.sum_mul(coffes[2], img[i - 1:i + 3, j - 2:j + 2])

                        sum3 = self.sum_mul(coffes[3], img[i - 2:i + 2, j - 2:j + 2])

                        # window0 = np.clip(np.dot(coffes[0], img[i - 2:i + 2, j - 1:j + 3]),0,255)
                        # window1 = np.clip(np.dot(coffes[1], img[i - 1:i + 3, j - 1:j + 3]),0,255)
                        # window2 = np.clip(np.dot(coffes[2], img[i - 1:i + 3, j - 2:j + 2]),0,255)
                        # window3 = np.clip(np.dot(coffes[3], img[i - 2:i + 2, j - 2:j + 2]),0,255)

                        window0 = np.clip(sum0, 0, 255)
                        window1 = np.clip(sum1, 0, 255)
                        window2 = np.clip(sum2, 0, 255)
                        window3 = np.clip(sum3, 0, 255)

                        bigger_img = pyramid[level + 1]

                        # dealing with image bounderies
                        if i < 2:
                            i = 2
                        elif i > pyramid[level].shape[0] - 4:
                            i = pyramid[level].shape[0] - 4
                        if j < 2:
                            j = 2
                        elif j > pyramid[level].shape[1] - 4:
                            j = pyramid[level].shape[1] - 4

                        new_i = i * 2
                        new_j = j * 2
                        bigger_rows, bigger_columns = bigger_img.shape
                        print 'wiiiiiiinnnnnnnnnnnnnnnnnnnnnndoooooooooooooow',window0
                        bigger_img[np.clip(new_i+ 1,0,bigger_rows-1) , np.clip(new_j,0,bigger_columns-1)] = window0
                        bigger_img[np.clip(new_i+ 1,0,bigger_rows-1), np.clip(new_j + 1,0,bigger_columns-1)] = window1
                        bigger_img[np.clip(new_i,0,bigger_rows-1), np.clip(new_j + 1,0,bigger_columns-1)] = window2
                        bigger_img[np.clip(new_i,0,bigger_rows-1), np.clip(new_j,0,bigger_columns-1)] = window3

                        pyramid[level+1]=bigger_img
                        # keep doing interpolation until no changes on that level

        return bigger_img


if __name__ == '__main__':
    print 'reading image'
    img = cv2.imread('1.png', 0)
    print 'showing image',img[255][200]
    cv2.imshow('original image', img)
    cv2.waitKey(0)
    removeElm = RemoveElementsFromImage()
    print 'removing element'
    removed = removeElm.RemoveElements(img, [195])
    print 'finished'
    cv2.imshow('removed', removed)
    cv2.waitKey(0)
