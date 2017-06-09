import cv2
import numpy as np
import math

def ImagePadding(img1, filter1):
    # get filter dimensions
    (filter_height, filter_width) = filter1.shape[:2]
    # find filter center
    pad = (filter_width - 1) / 2
    # pad image
    # should we pad with 0's or replicate?
    # cv2.BORDER_CONSTANT, value=0
    # padded_img = cv2.copyMakeBorder(img1, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    padded_img = np.lib.pad(img1, (pad, pad), 'constant')

    return padded_img

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x.real, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    # print 'M',M.shape
    # print 'x', x.shape
    return np.dot(M, x)

def FFT_helper(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x.real, dtype=float)
    N = x.shape[0]
    if N <= 100:  # this cutoff should be optimized
        return DFT_slow(x)
    else:
        X_even = FFT_helper(x[::2])
        X_odd = FFT_helper(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        #first half
        X_odd_pad1 = X_odd
        factor_pad1 = factor[:N / 2]
        diff_odd_factor1 = abs(factor[:N / 2].shape[0] - X_odd.shape[0])
        if factor[:N / 2].shape[0] < X_odd.shape[0]:
            factor_pad1 = np.pad(factor[:N / 2], (0, diff_odd_factor1), 'constant', constant_values=1)
        elif factor[:N / 2].shape[0] > X_odd.shape[0]:
            X_odd_pad1 = np.pad(X_odd, (0, diff_odd_factor1), 'constant', constant_values=1)
        else:
            factor_pad1 =  factor[:N / 2]
            X_odd_pad1 = X_odd

        first_half_mul_result =factor_pad1 * X_odd_pad1
        diff_even_res1 = abs(X_even.shape[0] - first_half_mul_result.shape[0])
        X_even_pad1 = X_even
        if first_half_mul_result.shape[0] < X_even.shape[0] :
            first_half_mul_result = np.pad(first_half_mul_result, (0, diff_even_res1), 'constant', constant_values=1)
        elif first_half_mul_result.shape[0] > X_even.shape[0]:
            X_even_pad1 = np.pad(X_even, (0, diff_even_res1), 'constant', constant_values=1)
        else:
            X_even_pad1 = X_even
            first_half_mul_result = factor_pad1 * X_odd_pad1
        # second half
        X_odd_pad2 = X_odd
        factor_pad2 = factor[N / 2:]
        diff_odd_factor2 = abs(factor[N / 2:].shape[0] - X_odd.shape[0])
        if factor[N / 2:].shape[0] < X_odd.shape[0]:
            factor_pad2 = np.pad(factor[:N / 2], (0, diff_odd_factor2), 'constant', constant_values=1)
        elif factor[N / 2:].shape[0] > X_odd.shape[0]:
            X_odd_pad2 = np.pad(X_odd, (0, diff_odd_factor2), 'constant', constant_values=1)
        else:
            factor_pad2 =  factor[N / 2:]
            X_odd_pad2 = X_odd
        sec_half_mul_result = factor_pad2 * X_odd_pad2
        diff_even_res2 = abs(X_even.shape[0] - sec_half_mul_result.shape[0])
        X_even_pad2 = X_even
        if sec_half_mul_result.shape[0] < X_even.shape[0]:
            sec_half_mul_result = np.pad(sec_half_mul_result, (0, diff_even_res2), 'constant', constant_values=1)
        elif sec_half_mul_result.shape[0] > X_even.shape[0]:
            X_even_pad2 = np.pad(X_even, (0, diff_even_res2), 'constant', constant_values=1)
        else:
            X_even_pad2 = X_even
            sec_half_mul_result = factor_pad2 * X_odd_pad2

        first_half = X_even_pad1 + first_half_mul_result
        sec_half = X_even_pad2 + sec_half_mul_result

        return np.concatenate([first_half,
                               sec_half])
#2d fft
def FFT(samples):
        fftRows = np.array([FFT_helper(row) for row in samples])
        return np.array([FFT_helper(row) for row in fftRows.transpose()]).transpose()
def IFFT_helper(x):
    fft = FFT_helper(x)
    ifft = np.divide(fft[-np.arange(fft.shape[0])], x.shape[-1])
    return ifft
#2d ifft
def IFFT(samples):
    ifftRows = np.array([IFFT_helper(row) for row in samples])
    return np.array([IFFT_helper(row) for row in ifftRows.transpose()]).transpose()

def Convolve(image, filt):
    print 'filt', filt.shape
    m, n = filt.shape
    if (m == n):
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        cimage = np.zeros((y,x)).astype(np.float)
        print cimage.shape
        for i in range(y):
            for j in range(x):


                # # get area to convolve
                # area = image[i:i+m, j:j+m]
                # # convolve
                # k = (area * filt).sum()
                # # store convolved value in result image
                # cimage[i][j] = k



                cimage[i][j] = np.sum(image[i:i+m, j:j+m]*filt)
    print 'vcvv',cimage
    print 'cimage', cimage.shape
    return cimage




# def Convolve(img, filter):
#     print 'filte',filter.shape
#
#     # get img dimensions
#     (img_height, img_width) = img.shape[:2]
#     # get filter dimensions
#     (filter_height, filter_width) = filter.shape[:2]
#     # find filter center
#     pad = (filter_width - 1) / 2
#     # pad image
#     image = ImagePadding(img, filter)
#     print 'img',image.shape
#     # allocate result
#     result = np.zeros((img_height, img_width), dtype="float32")
#     # loop over input image, sliding filter across pixel
#     for y in np.arange(pad, img_height + pad):
#         for x in np.arange(pad, img_width + pad):
#             # get area to convolve
#             print 'y - pad:y + pad + 1',y - pad,y + pad + 1
#             print 'x - pad:x + pad + 1', x - pad,x + pad + 1
#             area = image[y - pad:y + pad + 1+1, x - pad:x + pad + 1+1]
#             # convolve
#             k = (area * filter).sum()
#             # store convolved value in result image
#             result[y - pad, x - pad] = k.real
#
#     # rescale the output image to be in the range [0, 255]
#     # result = rescale_intensity(result, in_range=(0, 255))
#     result = (result * 255).astype("uint8")
#
#     return result

def distance(x0, y0, x1, y1):

    dist = np.sqrt(np.square(x1 - x0) + np.square(y1 - y0))

    return dist


def generateGaussianFilter(image, d0, isHigh):

    c_x = int(image.shape[0]/2)

    c_y = int(image.shape[1]/2)

    filter = [[np.exp(-1*(np.square(distance(c_x,c_y,i,j)))/(2*np.square(d0))) for j in range(image.shape[1])] for i in range(image.shape[0])]



    filter = np.asarray(filter)

    if isHigh:

        filter = 1 - filter

    filter = np.resize(filter,(27,27))

    return filter

def generateIdealFilter(image, d0, isHigh):
    filter = np.zeros((27,27))
    filter[12][12] = 1


    return filter
# tests
if __name__ == '__main__':
    img = cv2.imread("InputImage.png",0)
    # padded_img = ImagePadding(img)
    # filters
    F1 = generateGaussianFilter(img, 30, False)
    # cv2.imshow('Filter : Frequency Domain', np.uint8(255*F1))
    img = FFT(img)
    convolved_img = Convolve(img, F1)
    convolved_img = IFFT(convolved_img)
    # convolved_img = np.fft.ifft2(convolved_img)
    magnitude = np.abs(convolved_img)
    magnitude_spectrum = 20 * np.log(magnitude)
    cv2.imshow('Image : Filtered Frequency Domain',  np.uint8(magnitude_spectrum))
    # fq_filter = FFT(filter)
    # fq_filter_shifted = np.fft.fftshift(fq_filter)

    cv2.waitKey(0)