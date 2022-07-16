from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
imagecolor = plt.imread(r'C:\Users\totian\Desktop\فایل برگزیدگان جشنواره 1400 کمیته داوری\فایل پاورپوینت منتخبین نهایی\پژوهشگر جوان نمونه\علی خنجری.jpg')
imagecolor = np.array(imagecolor[:,:,:3])
plt.imshow(imagecolor)
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
        if (j-center[0])**2+(i-center[1])**2 <= radius**2:
            backimage[j,i] = 255
grayimage = np.array(backimage, dtype = np.uint8)
plt.imshow(grayimage,cmap='gray')
# In[]: creating gray image
def color2gray(imagecolor):
    R, G, B = imagecolor[:,:,0], imagecolor[:,:,1], imagecolor[:,:,2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    plt.figure()
    plt.imshow(imgGray, cmap = 'gray')
    return imgGray
grayimage = color2gray(imagecolor)
maingray = grayimage
# In[]: Guassian filter for smoothing
grayimage = cv2.GaussianBlur(grayimage,[7,7], 1.4)
plt.imshow(grayimage, cmap = 'gray')
# In[]: Laplacian of Gaussian (LoG)
grayimage = cv2.Laplacian(cv2.GaussianBlur(grayimage,[7,7],1.4),5, 7)
plt.imshow(grayimage, cmap = 'gray')
plt.figure()
smoothedimage = maingray - grayimage
plt.imshow(smoothedimage, cmap = 'gray')
# In[]: Ixx and Iyy and Ixy 
depth = 6
def sobel(Image,side = 0):
    if side == 0:
#       Ixkernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
       Ixkernel = np.array([[3,10,3],[0,0,0],[-3,-10,-3]])
       Ix = cv2.filter2D(Image, depth, Ixkernel)
       plt.figure()
       plt.imshow(Ix, cmap = 'gray')
       return Ix
    elif side == 1:
#       Iykernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
       Iykernel = np.array([[3,0,-3],[10,0,-10],[3,0,-3]])
       Iy = cv2.filter2D(Image, depth, Iykernel)
       plt.figure()
       plt.imshow(Iy, cmap = 'gray')
       return Iy
    elif side == 2:
#       Ixkernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
#       Iykernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
       Ixkernel = np.array([[3,10,3],[0,0,0],[-3,-10,-3]])
       Iykernel = np.array([[3,0,-3],[10,0,-10],[3,0,-3]])
       Ix = cv2.filter2D(Image, depth, Ixkernel)
       Iy = cv2.filter2D(Image, depth, Iykernel)
       plt.figure()
       plt.imshow(Ix, cmap = 'gray')
       plt.figure()
       plt.imshow(Iy, cmap = 'gray')
       return Ix, Iy
Ix, Iy = sobel(grayimage, side = 2)
Ixx = sobel(Ix, side = 0)
Iyy = sobel(Iy, side = 1)
Ixy = sobel(Ix, side = 1)
Iyx = sobel(Iy, side = 0)
# In[]: Haris
HarisDet = Ixx*Iyy - Ixy**2
HarisTrace = Ixx + Iyy
alpha = 0.8
Haris = np.floor(HarisDet - alpha*HarisTrace)
plt.figure()
plt.imshow(Haris, cmap = 'gray')
#plt.imshow(cv2.Sobel(grayimage,6,1,0,ksize=3), cmap = 'gray')
# In[]: salt and pepper noise
def addSaPeNoise(image, Number):
    sign = 1
    x , y = np.shape(image)
    for i in range(Number):
        noisex = np.random.randint(x)
        noisey = np.random.randint(y)
        if sign >0:
            image[noisex][noisey] = 0
        else:
            image[noisex][noisey] = 255
        sign = sign*(-1)
    return image
noisyimage = addSaPeNoise(grayimage, 2000)
plt.figure()
plt.imshow(noisyimage, cmap='gray')        
# In[]: Using Haris
grayimage = np.array(grayimage, dtype=np.uint8)
plt.imshow(cv2.cornerHarris(grayimage,5,3,0.2), cmap='gray')
# In[]: Mean Filter
meankernel = 1/25 * np.ones([5,5])
grayimage = cv2.filter2D(grayimage,6, meankernel)
plt.figure()
plt.imshow(grayimage, cmap='gray') 
# In[]: Dilation
DilatedImage = cv2.dilate(noisyimage,np.ones([3,3]))
plt.imshow(noisyimage,cmap='gray')
plt.figure()
plt.imshow(DilatedImage,cmap='gray')
# In[]: erosion
ErodedImage = cv2.erode(noisyimage,np.ones([3,3]))
plt.imshow(noisyimage,cmap='gray')
plt.figure()
plt.imshow(ErodedImage,cmap='gray')
# In[]: Closing
def Closing(image):
    DilatedImage = cv2.dilate(image,np.ones([5,5]))
    ErodedImage = cv2.erode(DilatedImage,np.ones([5,5]))
    return ErodedImage
#Open = Opening(noisyimage)
# In[]: Opening
def Opening(image):
    ErodedImage = cv2.erode(image,np.ones([5,5]))
    DilatedImage = cv2.dilate(ErodedImage,np.ones([5,5]))
    return DilatedImage
#Close = Closing(noisyimage)
# In[]: Opening and Closing
Open = Opening(Closing(noisyimage))
plt.figure()
plt.imshow(Open, cmap = 'gray')
Close = Closing(Opening(noisyimage))
plt.figure()
plt.imshow(Close, cmap = 'gray')
# In[]:
openedimage = Opening(grayimage)
WhiteTophat = grayimage - openedimage 
plt.imshow(WhiteTophat, cmap = 'gray')
plt.figure()
BlackTophat = Closing(grayimage) - grayimage
plt.imshow(BlackTophat, cmap = 'gray')
# In[]: Top and bottom hat with noisy image
openedimage = Opening(noisyimage)
Tophatnoisy = noisyimage - openedimage
plt.imshow(Tophatnoisy, cmap = 'gray')
plt.figure()
Bottomhatnoisy = Closing(noisyimage) - noisyimage
plt.imshow(Bottomhatnoisy, cmap = 'gray')
# In[]: Rotate
angle = np.pi
kernel = np.array([[np.cos(angle), np.sin(angle), 0],[-1*np.sin(angle),np.cos(angle),0],[0,0,1]])
rotatedimage =cv2.filter2D(grayimage, 6, kernel)
plt.figure()
plt.imshow(rotatedimage,cmap='gray')