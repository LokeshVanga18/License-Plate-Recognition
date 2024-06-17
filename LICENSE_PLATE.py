import cv2
import requests
import base64
import time
import torch
import matplotlib.pyplot as plt 
import easyocr
import imutils
import numpy as np


img = cv2.imread("car.jpeg")
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)


#    APPLYING FILTERS HERE

bfilter = cv2.bilateralFilter(gray , 11 , 17 , 17)
edge = cv2.Canny(bfilter , 30 , 200)#edge detection

#   FINDING CONTOURS AND APPLYING MASK

key = cv2.findContours(edge.copy() , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
con = imutils.grab_contours(key)
con = sorted(con , key=cv2.contourArea , reverse=True)[:10]

location = None
for c in con:
    a = cv2.approxPolyDP(c , 10 , True)
    if len(a) == 4:
        location=a
        break

mask = np.zeros(gray.shape , np.uint8)
new_i = cv2.drawContours(mask , [location] , 0 , 255 , -1)
new_i = cv2.bitwise_and(img , img , mask=mask)

while True:
    cv2.imshow("loke" , new_i)
    key = cv2.waitKey(0)
    if key == ord('l'):
        break
cv2.destroyAllWindows()


(x , y) = np.where(mask==255)
(x1 , y1) = (np.min(x) , np.min(y))
(x2 , y2) = (np.max(x) , np.max(y))
c_img = gray[x1:x2+1 , y1:y2+1]

reader = easyocr.Reader(['en'])
result = reader.readtext(c_img)
print(result[0][-2])