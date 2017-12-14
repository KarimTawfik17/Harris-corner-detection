from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import cv2 as cv
import math


# calculating gaussian
def gaussian(sigma,x):
    a= 1/(np.sqrt(2*np.pi)*sigma)
    b=math.exp(-(x**2)/(2*(sigma**2)))
    c = a*b
    return a*b

#calculating gaussian kernel
def gaussian_kernel(sigma):
    a=gaussian(sigma, -1)
    b=gaussian(sigma, 0)
    c=gaussian(sigma, 1)
    sum=a+b+c
    if sum!=0:
        a=a/sum
        b=b/sum
        c=c/sum
    return np.reshape(np.asarray([a,b,c]), (1,3))

# calculating 1st order derivative of gaussian
def gaussian_1derivative(sigma,x):
    a = -x/ (np.sqrt(2 * np.pi)*sigma**3)
    b = math.exp(-(x ** 2) / (2 * (sigma ** 2)))
    c = a * b
    return a * b

# calcualating kernel for 1st order derivative
def gaussian_1derivative_kernel(sigma):
    a=gaussian_1derivative(sigma, -1)
    b=gaussian_1derivative(sigma, 0)
    c=gaussian_1derivative(sigma, 1)
    sum=a+b+c
    if sum!=0:
        a=a/sum
        b=b/sum
        c=c/sum
    return np.reshape(np.asarray([a,b,c]), (1,3))


#calculating gaussian 2nd order derivative
def gaussian_2derivative(sigma,x):
    a = (x**2) / (np.sqrt(2 * np.pi) * sigma**5)
    b = math.exp(-(x ** 2) / (2 * (sigma ** 2)))
    c = a * b
    d=-1/(np.sqrt(2 * np.pi) * sigma**3)
    return c+d

#kernel for 2nd order derivative
def gaussian_2derivative_kernel(sigma):
    return np.reshape(np.asarray([gaussian_1derivative(sigma, -1), gaussian_1derivative(sigma, 0), gaussian_1derivative(sigma, 1)]),(1,3))


def Hessian(filepath):
    I = cv.imread(filepath,0)

    I=I.astype(np.float64)
    sigma=1

    G = gaussian_kernel(sigma)  # one dimensional gaussian mask to convolve with I [-1,0,1]
    Gx = gaussian_1derivative_kernel(sigma)  # calculating gaussian 1d derivative with sigma as 1 and x =-1,0,1
    Gy = np.transpose(Gx)
    Gxx=gaussian_2derivative_kernel(sigma)
    Gyy=np.transpose(Gxx)
    sIx = cv.filter2D(I, -1, G)

    sIy =cv.filter2D(I,-1,np.transpose(G))
    Ix=cv.filter2D(sIx, -1, Gx)
    Iy = cv.filter2D(sIy, -1, Gy)

    Ixx = cv.filter2D(Ix, -1, Gxx)   # calculating gaussian with second order derivative of image
    Iyy = cv.filter2D(Iy, -1, Gyy)
    Ixy = cv.filter2D(Ix, -1, Gyy)

    # H = np.matrix([[Ixx,Ixy],[Ixy,Iyy]])
    # w, v = LA.eig(H)
    #x,y=[]
    evalue=I.copy()

    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            H = ([Ixx[i][j], Ixy[i][j]], [Ixy[i][j], Iyy[i][j]])  #hessian matrix
            [e1, e2], _ = LA.eig(H)
            evalue[i,j]=e1+e2
    return evalue

def plot_corners(evalue,image):
    output = cv.cvtColor(I, cv.COLOR_GRAY2RGB)
    #
    output[evalue > 0.5 * np.max(evalue)] = [255, 255, 0]  # detecting corners
    plt.figure()

    plt.title("Hessian Matrix Corner Detection")
    plt.imshow(output)

    plt.show()

# i have kept outputs in output folder inside question 3 part 1

I = cv.imread("input1.png", 0)
evalue=Hessian("input1.png")
plot_corners(evalue,I)

I = cv.imread("input2.png", 0)
evalue=Hessian("input2.png")
plot_corners(evalue,I)

I = cv.imread("input3.png", 0)
evalue=Hessian("input3.png")
plot_corners(evalue,I)

