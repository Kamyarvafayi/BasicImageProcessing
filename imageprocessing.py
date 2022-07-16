from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
img = Image.open(r'C:\Users\totian\Desktop\tv 2.jpg')
imgplt = plt.imread(r'C:\Users\totian\Desktop\tv 2.jpg')
imagecolor = plt.imread(r'C:\Users\totian\Desktop\فایل برگزیدگان جشنواره 1400 کمیته داوری\فایل پاورپوینت منتخبین نهایی\پژوهشگر جوان نمونه\علی خنجری.jpg')
plt.imshow(imagecolor)
imgmtx = np.array(img, dtype=float)
imgmtx2 = np.array(imgmtx,dtype = np.uint8)
img2 = Image.fromarray(imgmtx2,'RGB')
imgbw = np.average(imgmtx,axis = 2)
imgmtx3 = np.array(imgbw,dtype = np.uint8)
img3 = Image.fromarray(imgmtx3)
imgplt2 = imgplt
plt.imshow(imgplt[:,:,:]/255,cmap='gray')
plt.imshow(imgplt2,cmap='gray')

sharpenkernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
imgfiltered = cv2.filter2D(imagecolor,-1,sharpenkernel)
plt.imshow(imgfiltered)                    
edgekernel1 =  np.array([[-1,0,1],[-2,0,2],[-1,0,1]])                   
imgfilterededge1 = cv2.filter2D(imagecolor,-1,edgekernel1)
plt.imshow(imgfilterededge1)    
edgekernel2 =  np.array([[-1,-2,-1],[0,0,0],[1,2,1]])                   
imgfilterededge2 = cv2.filter2D(imagecolor,-1,edgekernel2)
plt.imshow(imgfilterededge2)
edgeimage = imgfilterededge1 + imgfilterededge2
plt.imshow(edgeimage)
newimage = imagecolor - edgeimage
plt.imshow(newimage)
PositiveLaplaciankernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
imgfilteredLap1 = cv2.filter2D(imagecolor,-1,PositiveLaplaciankernel)
plt.imshow(imgfilteredLap)
plt.figure()
NegativeLaplaciankernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
imgfilteredLap2 = cv2.filter2D(imagecolor,-1,NegativeLaplaciankernel)
plt.imshow(imgfilteredLap)
# In[]: white and gray image
edgekernel1 =  np.array([[-1,0,1],[-2,0,2],[-1,0,1]])                   
imgfilterededge1gray = cv2.filter2D(imagecolor[:,:,0],-1,edgekernel1)
plt.figure()
plt.imshow(imgfilterededge1gray, cmap = 'gray')    
edgekernel2 =  np.array([[-1,-2,-1],[0,0,0],[1,2,1]])                   
imgfilterededge2gray = cv2.filter2D(imagecolor[:,:,0],-1,edgekernel2)
plt.figure()
plt.imshow(imgfilterededge2gray, cmap = 'gray')
edgeimagegray = imgfilterededge1gray + imgfilterededge2gray
plt.figure()
plt.imshow(edgeimagegray, cmap = 'gray')
# In[]: 
#plt.figure()
#redhist = plt.hist(imagecolor[:,:,0])
#greenhist, greenbins = plt.hist(imagecolor[:,:,1],20)
#bluehist = plt.hist(imagecolor[:,:,2])
# In[]: drawing circle
s = 200
backimage = np.zeros([s,s],dtype = int)
center = np.array([100,100])
radious = 50
for i in range(s):
    for j in range(s):
        if (j-center[0])**2+(i-center[1])**2 <= radious**2:
            backimage[j,i] = 255
newcircle = np.array(backimage, dtype = np.uint8)
plt.imshow(newcircle,cmap='gray')
hist = plt.hist(newcircle)
# In[]: creating rectangle
s = 200
backimage = np.zeros([s,s],dtype = int)
center1 = np.array([50,50])
center2 = np.array([150,150])
for i in range(s):
    for j in range(s):
        if j<=center2[0] and j>=center1[0] and i<=center2[1] and i>=center1[1] :
            backimage[j,i] = 255
newrect = np.array(backimage, dtype = np.uint8)
plt.imshow(newrect,cmap='gray')
# In[]: creating gray scale image
R, G, B = imagecolor[:,:,0], imagecolor[:,:,1], imagecolor[:,:,2]
imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
plt.imshow(imgGray, cmap='gray')
plt.show()
