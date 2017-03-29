import numpy as np
import math
import cv2

class RemoveElementsFromImage:
    def ComputeCubicInterpolationCoefficients1(self,upper,left,row, column, image):
        if (left == 0):
            x = np.array([-1.75, -0.75, 0.25, 1.25])
        else:
            x = np.array([-1.25, -0.25, 0.75, 1.75])

        if (upper == 0):
            y = np.array([-1.75, -0.75, 0.25, 1.25])
        else:
            y = np.array([-1.25, -0.25, 0.75, 1.75])

        ix = np.rint(x)
        iy = np.rint(y)

        xx = np.zeros((16))
        yy = np.zeros((16))

        for i in range(4):
            for j in range(4):
                xx[4 * i + j] = x[i]
                yy[4 * i + j] = y[i]

        values = image[np.maximum(row + (int(ix[0])), 0):np.minimum(row + (int(ix[3])) + 1, image.shape[0] - 1),
                 np.maximum(column + (int(iy[0])), 0):np.minimum(column + (int(iy[3])), image.shape[1] - 1) + 1]
        n_values = np.zeros((4, 4))
        n_values[:values.shape[0], :values.shape[1]] = values  # **
        values = n_values

        fx = values[:, 1 + left]

        fy = values[1 + upper, :]

        cx = np.polyfit(x, fx, 3)
        cy = np.polyfit(y, fy, 3)

        coeffs = np.zeros((4, 4))

        for i in range(4):
            for j in range(4):
                coeffs[i, j] = cx[i] * cy[j]

        sum = np.sum(np.sum(coeffs, 1))

        if sum != 0:
            coeffs = coeffs / sum

        return coeffs

    def BuildImagePyramid(self, image, n,color):
        element_size=list()
        smallerImage = image
        rows, columns = image.shape
        element_pixels=[]
        for l in range(rows):
            for k in range(columns):
                if image[l,k]==color:
                    element_pixels.append((l,k))

        element_size.append(element_pixels)
        pyramid = list()
        pyramid.append(image)
        while len(pyramid) < n:
            element_pixels=[]
            image = smallerImage
            smallerImage = np.zeros((image.shape[0] / 2, image.shape[1] / 2), dtype=np.uint8)
            for i in range(smallerImage.shape[0]):
                for j in range(smallerImage.shape[1]):
                    orig_i = 2 * i
                    orig_j = 2 * j
                    if (orig_i, orig_j) in element_size[len(pyramid) - 1] or (orig_i+1, orig_j+1) in element_size[len(pyramid) - 1] or (orig_i+1, orig_j) in element_size[len(pyramid) - 1] or (orig_i, orig_j+1) in element_size[len(pyramid) - 1]:
                        element_pixels.append((i,j))
                    averageOfPixels = (int(image[orig_i][orig_j]) + int(image[orig_i + 1][orig_j]) + int(
                        image[orig_i][orig_j + 1]) + int(image[orig_i + 1][orig_j + 1])) / 4
                    smallerImage[i][j] = averageOfPixels
            element_size.append(element_pixels)

           # n = n - 1
            pyramid.append(smallerImage)
           # cv2.imshow('pyramid level {:d}'.format(n),smallerImage)
            cv2.waitKey(0)

            #small to big
        pyramid.reverse()
        element_size.reverse()
        # print 'PYRAMID', len(pyramid)
        return (pyramid,element_size)

    def ComputeCubicInterpolationCoefficients(self, top, left, i, j, image):
        right_distances = [-1.25, -0.25, 0.75, 1.75]
        left_distances = [-1.75, -0.75, 0.25, 1.25]
        top_distances = [-1.25, -0.25, 0.75, 1.75]
        bottom_distances = [-1.75, -0.75, 0.25, 1.25]

        if left:
            x = left_distances
        else:
            x = right_distances
        if top:
            y = top_distances
        else:
            y = bottom_distances

        rows, columns = image.shape

        bx = [img[int(np.clip(math.ceil(i + x[0]), 0, rows)), j], img[int(np.clip(math.ceil(i + x[1]), 0, rows)), j], img[int(np.clip(math.ceil(i + x[2]), 0, rows)), j], img[int(np.clip(math.ceil(i + x[3]), 0, rows)), j]]
        by = [img[i, int(np.clip(math.ceil(j + y[0]), 0, columns))], img[i, int(np.clip(math.ceil(j + y[1]), 0, columns))], img[i, int(np.clip(math.ceil(j + y[2]), 0, columns))],img[i, int(np.clip(math.ceil(j + y[3]), 0, columns))]]

        cx = np.polyfit(x, bx, 3)
        cy = np.polyfit(y, by, 3)

        # print 'bx,by',bx,by
        m = np.zeros((4,4))
        for k in range(len(cx)):
            for h in range(len(cy)):
                m[k,h] = cx[k]*cy[h]

        s = m.sum()

        if s != 0:
            ans = m / float(s)
        else:
            ans = m
        return ans

    def sum_mul(self, mat1,mat2):
        print 'mat2',mat2
        print 'mat1', mat1
        sum = 0
        for k in range(mat1.shape[0]):
            for h in range(mat2.shape[0]):
                sum += mat1[k][h] * mat2[k][h]
        print 'sum',int(sum)
        return int(sum)

    def RemoveElements(self, image, colors):
        rows, columns = image.shape
        for color in colors:
            (pyramid,element_size) = self.BuildImagePyramid(image,8,color)
            bigger_img = image
            for level in range(len(pyramid)-1):
            #for level in range(4):
                img = pyramid[level]
                print 'img RemoveElements',img
                rows, columns = img.shape
                print 'rows,cols',rows,columns
                for i in range(rows):
                    coeffs = np.zeros(4, dtype=object)
                    for j in range(columns):
                        # print 'i,j in remove',i,j
                        # if img[i, j] != 255:
                             # print 'Intensity', img[i, j]

                        bigger_img = pyramid[level + 1]
                       # cv2.imshow('pyramid level {:d}'.format(level + 1), bigger_img)
                        if (i,j) in element_size[level] and i>4 and i<img.shape[0] and j>4 and j<img.shape[1] :

                            avg = -1
                            max_loop = 0
                            while img[i,j] != avg and max_loop < 10:
                                max_loop += 1

                                # print 'img avg before',img[i,j],avg

                                coeffs[0] = self.ComputeCubicInterpolationCoefficients(True, True, i, j, img)
                                coeffs[1] = self.ComputeCubicInterpolationCoefficients(True, False, i, j, img)
                                coeffs[2] = self.ComputeCubicInterpolationCoefficients(False, False, i, j, img)
                                coeffs[3] = self.ComputeCubicInterpolationCoefficients(False, True, i, j, img)

                                # dealing with image boundaries
                                # i = np.clip(i, 2, rows - 4)
                                # j = np.clip(j, 2, columns - 4)

                                # print 'img window 0', img[i - 2:i + 2, j - 1:j + 3]

                                sum0 = self.sum_mul(coeffs[0], img[i - 2:i + 2, j - 1:j + 3])
                                print 'sum0',sum0
                                sum1 = self.sum_mul(coeffs[1], img[i - 1:i + 3, j - 1:j + 3])
                                print 'sum1', sum1
                                sum2 = self.sum_mul(coeffs[2], img[i - 1:i + 3, j - 2:j + 2])
                                print 'sum2', sum2
                                sum3 = self.sum_mul(coeffs[3], img[i - 2:i + 2, j - 2:j + 2])
                                print 'sum3', sum3
                                quarter0 = np.clip(sum0, 0, 255)
                                quarter1 = np.clip(sum1, 0, 255)
                                quarter2 = np.clip(sum2, 0, 255)
                                quarter3 = np.clip(sum3, 0, 255)

                                bigger_img = pyramid[level + 1]
                                print 'img pyramid[level + 1]', bigger_img
                                avg = int(math.ceil((quarter0+quarter1+quarter2+quarter3)/4))

                                # print 'img avg after', img[i, j], avg
                                img[i, j] = avg
                                new_i = i * 2
                                new_j = j * 2
                                bigger_rows, bigger_columns = bigger_img.shape
                                print 'i,j',i,j
                                bigger_img[np.clip(new_i + 1, 0, bigger_rows-1), np.clip(new_j, 0, bigger_columns - 1)] = quarter0
                                bigger_img[np.clip(new_i + 1, 0, bigger_rows-1), np.clip(new_j + 1,0,bigger_columns-1)] = quarter1
                                bigger_img[np.clip(new_i, 0, bigger_rows-1), np.clip(new_j + 1,0,bigger_columns-1)] = quarter2
                                bigger_img[np.clip(new_i, 0, bigger_rows-1), np.clip(new_j,0,bigger_columns-1)] = quarter3
                               # cv2.imshow('pyramid level after {:d}'.format(level + 1), bigger_img)
                            # keep doing interpolation until no changes on that level

        return bigger_img


if __name__ == '__main__':
    print 'reading image'
    #img = cv2.imread('3.png', 0)
    size = 400
    img = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if not (50 < i < 70 and 50 < j <70):
                img[i, j] = 195
            # else:
            #     img[i,j]=255

    print 'showing image',img
    cv2.imshow('original image', img)
    # cv2.waitKey(0)
    removeElm = RemoveElementsFromImage()
    print 'removing element'
    after_removed = removeElm.RemoveElements(img,[195])
    # after_removed = removeElm.RemoveElements(img, [0, 127,254,190,126,253,109,191,110,223,63,135,95,155,219,177,86,246,235,212,250,249])

    print 'finished'
    cv2.imshow('after removed', after_removed)
    cv2.waitKey(0)

    # img = cv2.imread('1.png',0)
    # mask = cv2.imread('1mask.png', 0)
    # dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)

