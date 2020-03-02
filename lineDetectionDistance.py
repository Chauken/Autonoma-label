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




x1n = 0
x2n = 0
x3n = 0
y1n = 0
y2n = 0
y3n = 0


    #'outpy8.avi'
cap = cv2.VideoCapture(0)
ret,img=cap.read()
while True:
    _, frame = cap.read()


    axll = np.array([[0, 0]])
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
    kordx = [0,0,0]
    kordy = [0,0,0]
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x3 = int((x1+x2)/2)
            y3 = int((y1+y2)/2)
            kordx[0] = x1
            kordx[1] = x2
            kordx[2] = x3
            kordy[0] = y1
            kordy[1] = y2
            kordy[2] = y3
             
            #print(x1,y1,x2,y2,x3,y3)
            #cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 128), 1)
            
            leftAngle1 = (float(x1)/640)*(math.pi/2)
            leftAngle2 = (float(x2)/640)*(math.pi/2)
            leftAngle3 = (float(x3)/640)*(math.pi/2)

            if y1 > 50 and y2:
                #cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 128), 1)
                cv2.circle(frame,(x1,y1),1,(255,0,0),2)
                cv2.circle(frame,(x2,y2),1,(255,0,0),2)
                cv2.circle(frame,(x3,y3),1,(0,255,0),2)
                
                for p in range(3):
                    # y < 120
                    if kordy[p] <= 120: 
                        f = 550.6786 - 0.27671*kordx[p] - 12.0046*kordy[p] + 0.00010684*kordx[p]**(2) + 0.0057381*kordx[p]*kordy[p] + 0.11022*kordy[p]**(2) + 1.0173*10**(-6)*kordx[p]**(3) - 8.9658*10**(-6)*kordx[p]**(2)*kordy[p] - 2.7778*10**(-5)*kordx[p]*kordy[p]**(2) - 0.0003642*kordy[p]**(3) - 7.9473*10**(-10)*kordx[p]**(4) - 4.081*10**(-23)*kordx[p]**(3)*kordy[p] + 4.3403*10**(-8)*kordx[p]**(2)*kordy[p]**(2) - 1.9384*10**(-22)*kordx[p]*kordy[p]**(3)
                        cv2.putText(frame, text=str(round(f)), org=(kordx[p]+3,kordy[p]+3), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.35, color=(255,0,0), thickness=1)
                    
                    # y > 120
                    elif kordy[p] > 120:
                    
                        f = 134.5 - 0.04422*kordx[p] - 0.7315*kordy[p] + 9.766*10**(-5)*kordx[p]**2 + 2.593*10**(-5)*kordx[p]*kordy[p] +       0.001649*kordy[p]**2 - 6.104*10**(-8)*kordx[p]**3 - 1.163*10**(-8)*kordx[p]**(2)*kordy[p] - 1.085*10**(-8)*kordx[p]*kordy[p]**2 - 1.331*10**(-6)*kordy[p]**3
                        cv2.putText(frame, text=str(round(f)), org=(kordx[p]+3,kordy[p]+3), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.35, color=(0,255,0), thickness=1)
                    
                    x1n = round(f*math.sin(leftAngle1), 2)
                    y1n = round(f*math.cos(leftAngle1), 2)

                    x2n = round(f*math.sin(leftAngle2), 2)
                    y2n = round(f*math.cos(leftAngle2), 2)
                    
                    x3n = round(f*math.sin(leftAngle3), 2)
                    y3n = round(f*math.cos(leftAngle3), 2)

                    


            elif y1 and y2 > 50:
                #cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 128), 1)
                cv2.circle(frame,(x1,y1),1,(255,0,0),2)
                cv2.circle(frame,(x2,y2),1,(255,0,0),2)
                cv2.circle(frame,(x3,y3),1,(0,255,0),2)

                for p in range(3):
                    # y < 120
                    if kordy[p] <= 120: 
                        f = 550.6786 - 0.27671*kordx[p] - 12.0046*kordy[p] + 0.00010684*kordx[p]**(2) + 0.0057381*kordx[p]*kordy[p] + 0.11022*kordy[p]**(2) + 1.0173*10**(-6)*kordx[p]**(3) - 8.9658*10**(-6)*kordx[p]**(2)*kordy[p] - 2.7778*10**(-5)*kordx[p]*kordy[p]**(2) - 0.0003642*kordy[p]**(3) - 7.9473*10**(-10)*kordx[p]**(4) - 4.081*10**(-23)*kordx[p]**(3)*kordy[p] + 4.3403*10**(-8)*kordx[p]**(2)*kordy[p]**(2) - 1.9384*10**(-22)*kordx[p]*kordy[p]**(3)
                        cv2.putText(frame, text=str(round(f)), org=(kordx[p]+3,kordy[p]+3), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.35, color=(255,0,0), thickness=1)
                    
                    # y > 120
                    elif kordy[p] > 120:
                    
                        f = 134.5 - 0.04422*kordx[p] - 0.7315*kordy[p] + 9.766*10**(-5)*kordx[p]**2 + 2.593*10**(-5)*kordx[p]*kordy[p] +       0.001649*kordy[p]**2 - 6.104*10**(-8)*kordx[p]**3 - 1.163*10**(-8)*kordx[p]**(2)*kordy[p] - 1.085*10**(-8)*kordx[p]*kordy[p]**2 - 1.331*10**(-6)*kordy[p]**3
                        cv2.putText(frame, text=str(round(f)), org=(kordx[p]+3,kordy[p]+3), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.35, color=(0,255,0), thickness=1)
                    
                    x1n = round(f*math.sin(leftAngle1), 2)
                    y1n = round(f*math.cos(leftAngle1), 2)

                    x2n = round(f*math.sin(leftAngle2), 2)
                    y2n = round(f*math.cos(leftAngle2), 2)
                    
                    x3n = round(f*math.sin(leftAngle3), 2)
                    y3n = round(f*math.cos(leftAngle3), 2)

            axll = np.append(axll, [[x1n, y1n]], axis=0)
            axll = np.append(axll, [[x2n, y2n]], axis=0)
            axll = np.append(axll, [[x3n, y3n]], axis=0)
            
            #cv2.rectangle(frame, (x1,y1),(x2,y2),(0, 0, 255),1)
    
    axll = np.delete(axll, 0, 0)
    plt.cla()

    plt.axis([0, 150, 0, 150])
    plt.scatter(axll[:, 0], axll[:,1])

    plt.pause(0.05)
    
    plt.ion()
    plt.show()

    cv2.namedWindow('linesDetected', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('linesDetected', 640, 480) 

    #cv2.imshow("linesEdges", cannyed_image)
    cv2.imshow("linesDetected", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.waitKey(0)
cv2.destroyAllWindows()
#plt.figure()
#plt.imshow(cannyed_image)
plt.show()


