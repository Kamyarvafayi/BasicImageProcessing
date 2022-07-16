from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
imagecolor = plt.imread(r'C:\Users\totian\Desktop\tempcodes\sample picture.jpg')
imagecolor = np.array(imagecolor[:,:,0])
plt.imshow(imagecolor,cmap='gray')
# In[]: creating gray image
def color2gray(imagecolor):
    R, G, B = imagecolor[:,:,0], imagecolor[:,:,1], imagecolor[:,:,2]
    imgGray = np.floor(0.2989 * R + 0.5870 * G + 0.1140 * B)
    plt.figure()
    plt.imshow(imgGray, cmap = 'gray')
    return imgGray
grayimage = color2gray(imagecolor)

# In[]: zero Cdf
zerocdf = 0
maingray = grayimage
imageshape = np.shape(maingray)
for i in range(imageshape[0]):
    for j in range(imageshape[1]):
        if maingray[i,j]==0:
            zerocdf += 1
# In[]: finding image CDF

count = np.zeros([255,1])
Pr = np.zeros([255,1])
for i in range(1,255):
     count[i] = (zerocdf+sum(grayimage[grayimage==i]))/i
     Pr[i] = count[i]/np.size(grayimage)
# Cdf
Cdf = np.zeros([255,1])
for i in range(1,255):
     Cdf[i] = sum(count[:i+1])
# Equalization
for i in range(imageshape[0]):
    for j in range(imageshape[1]):
        if maingray[i,j]!=255:
            maingray[i,j] = np.floor((Cdf[maingray[i,j].astype(int)]-zerocdf)/(np.size(maingray)-zerocdf)*255)-1
plt.figure()
plt.imshow(maingray,cmap='gray')
plt.figure()
plt.imshow(grayimage,cmap='gray')
# In[]:
plt.hist(maingray)
plt.figure()
plt.hist(grayimage)
