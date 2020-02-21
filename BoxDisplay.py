import cv2
import re

print('Annotation text file:')
folderPath = input()

f = open(folderPath, 'r+')

Tlines = f.readlines()

PathPattern = re.compile(r'jpg|png')
CoordPattern = re.compile(r'(\d{1,3}),')
LabelPattern = re.compile(r',(\d) ')

print('0    :   Go to next')

for pp in range(len(Tlines)):

    BigBoy = Tlines[pp]

    for i in PathPattern.finditer(BigBoy):
        path_index_len = i.end()

    PathString = BigBoy[0:path_index_len]

    image1 = cv2.imread(PathString)

    window_name = 'Image'

    q = 0
    for t in range(int(len(CoordPattern.findall(BigBoy))/4)):
        if (int(LabelPattern.findall(BigBoy)[t]) == 1):
            Colour = (0, 255, 0)
        else:
            Colour = (0, 0, 255)
        start = (int(CoordPattern.findall(BigBoy)[q]),int(CoordPattern.findall(BigBoy)[q+1]))
        end= (int(CoordPattern.findall(BigBoy)[q+2]),int(CoordPattern.findall(BigBoy)[q+3]))
        cv2.rectangle(image1,start,end,Colour,2)
        cv2.putText(image1, text=str(LabelPattern.findall(BigBoy)[t]),
                    org=(int(CoordPattern.findall(BigBoy)[q]), int(CoordPattern.findall(BigBoy)[q + 1]) - 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.60, color=(255,0,0), thickness=2)
        q = q + 4

    cv2.imshow('Iteration ' + str(pp+1),image1)

    cv2.waitKey(0)

    cv2.destroyAllWindows()