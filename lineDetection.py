import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import time

height = 0
width = 0


def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, mask)

    
def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.1, rows*0.95]
    top_left     = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right    = [cols*0.6, rows*0.6] 

    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]

    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)


# images showing the region of interest only

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#image = mpimg.imread('/home/martin/Desktop/keras-yolo3-master/git roadlines/CarND-LaneLines-P1/test_images/solidWhiteCurve.jpg')
#image = cv2.imread("/home/martin/Desktop/keras-yolo3-master/Train00001.png")



'''

cropped_image = region_of_interest(
    image,
    np.array([region_of_interest_vertices], np.int32),
)
'''
#white mask
lower = np.uint8([0, 0, 0])
upper = np.uint8([20, 255, 100])





    #'outpy8.avi'
cap = cv2.VideoCapture(0)
ret,img=cap.read()
while True:
    _, frame = cap.read()


    
    # note that [:,:,0] is blue, [:,:,1] is green, [:,:,2] is red
    #frame[:,:,0] = 0
    # write the image

    #frame = convert_hls(frame)
    mask = frame
   # mask = cv2.inRange(frame, lower, upper)
    #mask = convert_gray_scale(frame)

    #cv2.imshow("masked",mask)
    
    
    #select region of interest
    #frame = select_region(frame)
    time.sleep(0.05)

  

    cannyed_image = cv2.Canny(mask, 75, 150)
    lines = cv2.HoughLinesP(cannyed_image, 1, np.pi/180, 50, maxLineGap=40)
    #Colour = (0, 0, 255)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            print(x1,y1,x2,y2)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 128), 1)
            

            #cv2.rectangle(frame, (x1,y1),(x2,y2),(0, 0, 255),1)

    cv2.namedWindow('linesDetected', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('linesDetected', 1920, 1080) 

    cv2.imshow("linesEdges", cannyed_image)
    cv2.imshow("linesDetected", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.waitKey(0)
cv2.destroyAllWindows()
#plt.figure()
#plt.imshow(cannyed_image)
plt.show()


