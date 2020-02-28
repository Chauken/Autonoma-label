import cv2



cap = cv2.VideoCapture(0)
ret,img=cap.read()
#while True:
_, frame = cap.read()



cv2.imwrite('frame.png',frame)

print("done")

cv2.waitKey(0)

