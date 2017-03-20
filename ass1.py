import numpy as np
import cv2


class RemoveElementsFromImage:
    print cv2.__version__

    def BuildImagePyramid(self,Image,n):
        print 'damn'


if __name__ == '__main__':

    img=cv2.imread('prettyGirl.jpg',0)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    removeElm=RemoveElementsFromImage()
    removeElm.BuildImagePyramid(img,0)