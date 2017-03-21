import numpy as np
import cv2


class RemoveElementsFromImage:
    print cv2.__version__

    def BuildImagePyramid(self,Image,n):
        print 'damn'
        print Image
        smallerImage=Image
        # height=smallerImage.shape[0]
        # width=smallerImage.shape[1]
        while(n>0):
            Image = smallerImage
            smallerImage = np.zeros((Image.shape[0] / 2, Image.shape[1] / 2), dtype=np.uint8)
            for i in range(smallerImage.shape[0]):
                for j in range(smallerImage.shape[1]):
                    orig_i=2*i
                    orig_j=2*j
                    averageOfPixels= (int(Image[orig_i][orig_j])+ int(Image[orig_i+1][orig_j])+ int(Image[orig_i][orig_j+1]) +int(Image[orig_i+1][orig_j+1]))/4
                    smallerImage[i][j]=averageOfPixels
                    #print smallerImage[i][j]
            n=n-1


        return smallerImage


    def  ComputeCubicInterpolationCoefficients(self):
        print 'here'


if __name__ == '__main__':

    img=cv2.imread('prettyGirl.jpg',0)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    removeElm=RemoveElementsFromImage()
    smaller=removeElm.BuildImagePyramid(img,3)
    print 'finished'
    cv2.imshow('smaller',smaller)
    cv2.waitKey(0)