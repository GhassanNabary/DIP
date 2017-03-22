import numpy as np
import math
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


    def  ComputeCubicInterpolationCoefficients(self,top,left,i,j,img):
        print 'here'
        right_distances=[-1.25,-0.25,0.75,1.75]
        left_distances = [-1.75, -0.75, 0.25, 1.25]
        top_distances=[-1.25, -0.25, 0.75, 1.75]
        bottom_distances=[-1.75, -0.75, 0.25, 1.25]

        #compute Cx
        if(left):
            x=left_distances
        else:
            x=right_distances
        if(top):
            y=top_distances
        else:
            y=bottom_distances



#TODO image bounderies
        bx=[img[ math.ceil(i+x[0]),j],img[math.ceil(i+x[1]),j],img[math.ceil(i+x[2]),j],img[math.ceil(i+x[3]),j]]
        by=[img[ i,math.ceil(j+y[0])],i,img[math.ceil(j+y[1])],i,img[math.ceil(j+y[2])],img[i,math.ceil(j+y[3])]]

        cx=np.polyfit(x,bx,3)
        cy=np.polyfit(y,by,3)

        m = np.outer(cx, cy)

        s = m.sum()

        return m/float(s)

    def  RemoveElements(self,Image,colors):



if __name__ == '__main__':

    # img=cv2.imread('prettyGirl.jpg',0)
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # removeElm=RemoveElementsFromImage()
    # smaller=removeElm.BuildImagePyramid(img,1)
    # print 'finished'
    # cv2.imshow('smaller',smaller)
    # cv2.waitKey(0)
    cx=[1,1,1,1]
    cy=[1,1,1,1]
    # vec = [(i * j) for (i, j) in zip(cx, cy)]
    a = np.outer(cx, cy)
    # b = np.matrix(a)
    s = a.sum()
    print 1
    print type(a)
    b = a / float(s)
    print b

