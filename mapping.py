import matplotlib
import numpy as np
import cv2


cords = np.array([[0,208],[416,208]])

image = cv2.imread('frame1.png')

rez = cv2.resize(image,(416,416),interpolation = cv2.INTER_AREA)


#cv2.line(image, (0, 180), (235, 10), (0, 0, 128), 1)

cv2.namedWindow('pic', cv2.WINDOW_NORMAL)
cv2.resizeWindow('pic', 1920, 1080) 

cv2.imshow("pic", rez)



# provide points from image 1
pts_src = np.array([[154, 174], [702, 349], [702, 572],[1, 572], [1, 191]])
# corresponding points from image 2 (i.e. (154, 174) matches (212, 80))
d2map = np.array([[212, 80],[489, 80],[505, 180],[367, 235], [144,153]])
 
# calculate matrix H
h, status = cv2.findHomography(pts_src, d2map)
 
# provide a point you wish to map from image 1 to image 2
a = np.array([[154, 174]], dtype='float32')
a = np.array([a])
 
# finally, get the mapping
pointsOut = cv2.perspectiveTransform(a, h)

cv2.waitKey(0)
print(pointsOut)
#print(cords)