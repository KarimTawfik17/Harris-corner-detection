from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import cv2 as cv
import math
import matplotlib.cm as cm
import time


#calculating gaussian
def gaussian(sigma,x):
    a= 1/(np.sqrt(2*np.pi)*sigma)
    b=math.exp(-(x**2)/(2*(sigma**2)))
    c = a*b
    return a*b

# getting gaussian kernel
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

# 1st order gaussian derivative
def gaussian_1derivative(sigma,x):
    a = -x/ (np.sqrt(2 * np.pi)*sigma**3)
    b = math.exp(-(x ** 2) / (2 * (sigma ** 2)))
    c = a * b
    return a * b
# 1st order gaussian derivative kernel
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


def plot_image(image,title):
	plt.figure()

	plt.title(title)
	plt.imshow(image,cmap = 'gray')

	plt.show()


def Harris(filepath,alpha):

    start = time.time()
    I = cv.imread(filepath,0)

    # I=I.astype(np.float64)
    G = gaussian_kernel(1)  # one dimensional gaussian mask to convolve with I [-1,0,1]

    Gx = gaussian_1derivative_kernel(1)  # calculating gaussian 1d derivative with sigma as 1 and x =-1,0,1

    Gy = np.transpose(Gx)
    Ix = cv.filter2D(I, -1, Gx)
    Iy = cv.filter2D(I, -1, Gy)

    sIx=np.square(Ix)  # square of x direction smoothen image
    sIy=np.square(Iy)  # square of y direction smoothen image

    Ixy=Ix*Iy

    L2x = cv.filter2D(sIx,-1,G+np.transpose(G))  # convolution
    L2y = cv.filter2D(sIy,-1,G+np.transpose(G))
    Lxy = cv.filter2D(Ixy,-1,G+np.transpose(G))



    cornerness=np.zeros(I.shape, np.float64)

    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            H = np.array([[L2x[i,j], Lxy[i,j]], [Lxy[i,j], L2y[i,j]]])  #harris corner detection

            cornerness[i,j]= LA.det(H) - (alpha * (np.trace(H)))


    output = cv.cvtColor(I, cv.COLOR_GRAY2RGB)
    output[cornerness > 0.5 * np.max(cornerness)] = [255, 255, 0]

    return output,time.time()-start

# Plotting all images
def plot_image(img1,title1,img2,title2,img3,title3,img4,title4,img5,title5,img6,title6,img7,title7,img8,title8,img9,title9):

    plt.subplot(331)
    plt.imshow(img1)
    plt.title(title1)


    plt.subplot(332)
    plt.imshow(img2)
    plt.title(title2)


    plt.subplot(333)
    plt.imshow(img3)
    plt.title(title3)


    plt.subplot(334)
    plt.imshow(img4)
    plt.title(title4)

    plt.subplot(335)
    plt.imshow(img5)
    plt.title(title5)


    plt.subplot(336)
    plt.imshow(img6)
    plt.title(title6)


    plt.subplot(337)
    plt.imshow(img7)
    plt.title(title7)


    plt.subplot(338)
    plt.imshow(img8)
    plt.title(title8)

    plt.subplot(339)
    plt.imshow(img9)
    plt.title(title9)
    plt.show()


# i have kept outputs in output folder inside question 3
# By comparing the output and time in question 3 part 2 and question3 part 3 we can say that accuracy is same but efficiency has changed.
#  Efficiency is better in part 2 compare to part 3

# time taken in question 3 part 3 for input1.png image is 10.55 sec
# time taken in question 3 part 2 for input1.png image is  6.66 sec
# so we can say in part2 time taken is less than in part3 so the efficiency in part 2 was better than part 3
# Execution time for other images can be seen in the output image
# Also if we decrease alpha noise adds up more whereas if we increase alpha noise decreases
alpha1=1/25
output1,Time1=Harris("input1.png",alpha1)
output2,Time2=Harris("input2.png",alpha1)
output3,Time3=Harris("input3.png",alpha1)

alpha2=1/35
output4,Time4=Harris("input1.png",alpha2)
output5,Time5=Harris("input2.png",alpha2)
output6,Time6=Harris("input3.png",alpha2)

alpha3=1/15
output7,Time7=Harris("input1.png",alpha3)
output8,Time8=Harris("input2.png",alpha3)
output9,Time9=Harris("input3.png",alpha3)

plot_image(output1,"Alpha ="+str(alpha1)+" Time = "+str(Time1),output2,"Alpha ="+str(alpha1)+" Time = "+str(Time2),output3,"Alpha ="+str(alpha1)+" Time = "+str(Time3),output4,"Alpha ="+str(alpha2)+" Time = "+str(Time4),output5,"Alpha ="+str(alpha2)+" Time = "+str(Time5),output6,"Alpha ="+str(alpha2)+" Time = "+str(Time6),output7,"Alpha ="+str(alpha3)+" Time = "+str(Time7),output8,"Alpha ="+str(alpha3)+" Time = "+str(Time8),output9,"Alpha ="+str(alpha3)+" Time = "+str(Time9))
