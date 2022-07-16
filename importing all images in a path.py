import glob
import cv2
import matplotlib.pyplot as plt
collection = []
path = glob.glob(r"C:\Users\totian\Desktop\فایل برگزیدگان جشنواره 1400 کمیته داوری\فایل پاورپوینت منتخبین نهایی\پژوهشگر جوان نمونه\*.jpg")
for images in path:
    img = plt.imread(images)
    collection.append(img)