from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
# In[]:
imagecolor = plt.imread(r'C:\Users\totian\Desktop\tempcodes\sample picture.jpg')
imagecolor = np.array(imagecolor[:,:,0])
plt.figure()
plt.imshow(imagecolor,cmap='gray')
# In[]: Sobel Edge Detector
SobelX = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
SobelY = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
Gx = cv2.filter2D(imagecolor,6,SobelX)
Gy = cv2.filter2D(imagecolor,6,SobelY)
EdgeImage = np.round(np.sqrt(Gx**2+Gy**2)/255)
plt.figure()
plt.imshow(EdgeImage,cmap='gray')
# In[]: 
Theta = np.arange(0,np.pi,np.pi/180)
r = np.arange(0, np.sqrt(EdgeImage.shape[0]**2 + EdgeImage.shape[1]**2),1)
Accumulator = np.zeros([Theta.shape[0],r.shape[0]*2])
# In[]:
for i in range(EdgeImage.shape[0]):
    for j in range(EdgeImage.shape[1]):
        if EdgeImage[i,j] == 1:
           for k in range(Theta.shape[0]):
               r  = np.round(i*np.cos(Theta[k])+j*np.sin(Theta[k])) 
               Accumulator[k, int(r + 1230)] += 1
                   