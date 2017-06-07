import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def oneDimFFT(samples):

    n = len(samples)

    if n == 1:

        return samples

    w_n = np.exp(-2j * np.pi / n)

    w = 1

    a_even = samples[0::2]

    a_odd = samples[1::2]



    y_0 = oneDimFFT(a_even)

    y_1 = oneDimFFT(a_odd)

    y = np.zeros(n) + 0 * 1j

    m = int(n / 2)

    for k in range(m):

        y[k] = y_0[k] + (w * y_1[k])

        y[k + m] = y_0[k] - (w * y_1[k])

        w *= w_n

    return y





def twoDimFFT_test(samples):

    fftRows = np.array([oneDimFFT(row) for row in samples])

    return np.array([oneDimFFT(row) for row in fftRows.transpose()]).transpose()


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

def FFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x.real, dtype=float)
    N = x.shape[0]
    if N <= 100:  # this cutoff should be optimized
        return DFT_slow(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
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
def fftpad(f):
    m, n = f.shape
    M, N = 2 ** int(math.ceil(math.log(m, 2))), 2 ** int(math.ceil(math.log(n, 2)))
    pf =np.zeros(shape=(M,N))
    for i in range(0, m):
        for j in range(0, n):
            pf[i][j] = f[i][j]
    return pf


def twoDimFFT(samples):
    fftRows = np.array([FFT(row) for row in samples])
    return np.array([FFT(row) for row in fftRows.transpose()]).transpose()

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


def Convolve(img2, filter2):
    # get img dimensions
    (img_height, img_width) = img2.shape[:2]
    # get filter dimensions
    (filter_height, filter_width) = filter2.shape[:2]
    # find filter center
    pad = (filter_width - 1) / 2
    # pad image
    image = ImagePadding(img2, filter2)
    # allocate result
    result = np.zeros((img_height, img_width), dtype="float32")

    # loop over input image, sliding filter across pixel
    for y in np.arange(pad, img_height + pad):
        for x in np.arange(pad, img_width + pad):
            # get area to convolve
            area = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            # convolve
            k = (area * filter2).sum()
            # store convolved value in result image
            result[y - pad, x - pad] = k.real

    # rescale the output image to be in the range [0, 255]
    # result = rescale_intensity(result, in_range=(0, 255))
    result = (result * 255).astype("uint8")

    return result


# def rescale_intensity(image, in_range='image', out_range='dtype'):
#     dtype = image.dtype.type
#
#     imin, imax = intensity_range(image, in_range)
#     omin, omax = intensity_range(image, out_range, clip_negative=(imin >= 0))
#
#     image = np.clip(image, imin, imax)
#
#     image = (image - imin) / float(imax - imin)
#
#     return dtype(image * (omax - omin) + omin)
#
#
# def intensity_range(image, range_values='image', clip_negative=False):
#     if range_values == 'dtype':
#         range_values = image.dtype.type
#
#     if range_values == 'image':
#         i_min = np.min(image)
#         i_max = np.max(image)
#     elif range_values in DTYPE_RANGE:
#         i_min, i_max = DTYPE_RANGE[range_values]
#         if clip_negative:
#             i_min = 0
#     else:
#         i_min, i_max = range_values
#     return i_min, i_max
# from cmath import exp,pi
# def FFT(X):
#     n = len(X)
#     w = exp(-2*pi*1j/n)
#     if n > 1:
#         X = FFT(X[::2]) + FFT(X[1::2])
#         for k in xrange(n/2):
#             xk = X[k]
#             X[k] = xk + w**k*X[k+n/2]
#             X[k+n/2] = xk - w**k*X[k+n/2]
#     return X

# tests
if __name__ == '__main__':
    # create filter
    filter = np.zeros((3, 3))
    filter[0,1] = 1
    filter[1,0] = 1
    filter[1,1] = 1
    filter[1,2] = 1
    filter[2,1] = 1
    # read image
    # img =  np.random.uniform(0,255,(200,200))
    # img = np.zeros(5,dtype=np.uint8)
    # for m in range(img.shape[0]):
    #     for n in range(img.shape[1]):
    #         if m < n :
    #             img[m,n] = 1
    # img = cv2.imread('InputImage.png', 0)
    img = cv2.imread('lena.jpg',0)
    # img = cv2.imread('InputImage.png', 0)
    # print 'img', img[::2]
    fq_img = twoDimFFT(img)
    # fq_img = np.fft.fft2(img)
    fshift = np.fft.fftshift(fq_img)
    # magnitude_spectrum = 20 * np.log(np.abs(fshift))
    # cv2.imwrite('test.png',magnitude_spectrum)
    # convolve image with filter
    convolved_img = Convolve(fshift, filter)
    spatial_img = np.fft.ifft2(convolved_img)
    # fshift = np.fft.ifftshift(spatial_img)
    magnitude_spectrum1 = 20 * np.log(np.abs(spatial_img))
    print 'conv ',np.uint8(magnitude_spectrum1)

    cv2.imwrite('res.png',np.uint8(magnitude_spectrum1))
    # spatial_imgshift, = np.array([spatial_img.real])
    # show original image
    # cv2.imshow('original image', img)
    # show convolved image
    # cv2.imshow('convolved image', spatial_img.real)
    # test = np.fft.fft2(img)
    # test = twoDimFFT(img)
    # fshift = np.fft.fftshift(test)
    # magnitude_spectrum = 20 * np.log(np.abs(fshift))
    # cv2.imwrite('test.png',magnitude_spectrum)
    # plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.show()
    # wait for exit
    #------------------------------
    # img = cv2.imread('image4.jpg', 0)
    # cv2.imshow(' image4', img)
    # f = np.fft.fft2(img)
    # fshift = np.fft.fftshift(f)
    # f_ishift = np.fft.ifftshift(fshift)
    # d_shift = np.array(np.dstack([f_ishift.real, f_ishift.imag]))
    # # img_back = cv2.idft(d_shift)
    # img_back = np.fft.ifft2(d_shift)
    # d_shift1 = np.array(np.dstack([img_back.real, img_back.imag]))
    # # img = cv2.magnitude(d_shift1[:, :, 0], d_shift1[:, :, 1])
    # cv2.imshow(' image', img)
    # cv2.waitKey(0)
