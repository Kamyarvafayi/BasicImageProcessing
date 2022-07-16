from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
# In[]: Creating rect
s = 200
backimage = np.zeros([s,s],dtype = int)
center1 = np.array([50,50])
center2 = np.array([150,150])
for i in range(s):
    for j in range(s):
        if j<=center2[0] and j>=center1[0] and i<=center2[1] and i>=center1[1] :
            backimage[j,i] = 255
grayimage = np.array(backimage, dtype = np.uint8)
plt.imshow(grayimage,cmap='gray')
# In[]: drawing circle
s = 200
backimage = np.zeros([s,s],dtype = int)
center = np.array([100,100])
radius = 50
for i in range(s):
    for j in range(s):
        if (j-center[0])**2+(i-center[1])**2 == radius**2:
            backimage[j,i] = 255
grayimage = np.array(backimage, dtype = np.uint8)
plt.imshow(grayimage,cmap='gray')
# In[]:
imagecolor = plt.imread(r'C:\Users\totian\Desktop\tempcodes\sample picture.jpg')
imagecolor = np.array(imagecolor[:,:,0])
plt.figure()
plt.imshow(imagecolor,cmap='gray')
# In[]: SUSAN
s1 = 100
s2 = 100
T1 = 5
T2 = 3
Threshold = 1
gthreshold = 8
cornerdetector = np.zeros([s1,s2],dtype = np.uint8)
finaldetector = np.zeros([s1,s2],dtype = np.uint8)
n = np.zeros([s1,s2],dtype = float)
backimage = imagecolor
for i in range(s1):
    for j in range(s2):
        count = 0
        if (i > 1 and j > 1) and (i< s1-1 and j < s2-1) :
            for t in [-1,0,1]:
                for k in [-1,0,1]:
                    c = np.exp(-((backimage[i,j]-backimage[i+t,j+k])/Threshold)**6)
                    n[i,j] += c
        cornerdetector[i,j] = max (gthreshold-n[i,j],0)
finaldetector[cornerdetector>T1] = 255
plt.figure()
plt.imshow(finaldetector,cmap='gray')
plt.figure()
plt.imshow(imagecolor[:s1,:s2],cmap='gray')
