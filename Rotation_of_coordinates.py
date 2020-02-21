from create import create
import sys
import argparse
import os
list=[]
def main():
    '''picpath=input('please enter pictures direction as "C:\\folder" format:')
    annpath = input('please enter annotation file path as "C:\\folder" format:')'''
    '''for file in os.listdir(picpath):
        list.append(os.path.join(picpath, file))
    for path in list:
        create(path,annpath) Lable_Shag3'''
    picpath = input('please enter the direction of the file you want to change as for example (text.txt)')
    annpath = input('please enter name of new file, example (text.txt)')
    rotation = input('please enter counter clockwise rotation of image where 90,180 and 270 is supported')
    filee = open(picpath, 'r+')
    f = open(annpath, "w+")
    content = filee.readlines()
    degre = int(rotation)
    '''with open("Lable_Shag3.txt") as filee:"'''
    i = 0
    p = 0
    final = ""
    cordinates = ["", "", "", "", ""]
    cordinate = 0
    '''for x in content:'''
    for x in content:
        "content = [x.strip() for x in content]"

        print(x)
        Ree = len(x)
        while True:
            if x[i] == ',':
                if x[i-2] == ' ':
                    i = i-1
                elif x[i-3] == ' ':
                    i = i-2
                elif x[i-4] == ' ':
                    i = i-3
                if p == 0:
                    p = 1
                    for u in range(i):
                        final = final + x[u]
                while True:
                    if x[i] == ' ':
                        cordinate = 0
                        '''print(cordinates[0])
                        print(cordinates[1])
                        print(cordinates[2])
                        print(cordinates[3])
                        print(cordinates[4])'''
                        if int(cordinates[0]) > int(cordinates[2]):
                            temp1 = cordinates[0]
                            cordinates[0] = cordinates[2]
                            cordinates[2] = temp1
                        if int(cordinates[1]) > int(cordinates[3]):
                            temp1 = cordinates[1]
                            cordinates[1] = cordinates[3]
                            cordinates[3] = temp1

                        if degre == 90:
                            temp1 = cordinates[0]
                            temp2 = cordinates[2]
                            temp3 = 416 - int(temp1)
                            temp4 = 416 - int(temp2)
                            cordinates[0] = cordinates[1]
                            cordinates[2] = cordinates[3]
                            cordinates[1] = str(temp4)
                            cordinates[3] = str(temp3)
                        elif degre == 180:
                            temp1 = cordinates[0]
                            temp2 = cordinates[1]
                            temp3 = cordinates[2]
                            temp4 = cordinates[3]
                            cordinates[0] = str(416-int(temp3))
                            cordinates[1] = str(416-int(temp4))
                            cordinates[2] = str(416-int(temp1))
                            cordinates[3] = str(416-int(temp2))
                        elif degre == 270:
                            temp1 = cordinates[1]
                            temp2 = cordinates[3]
                            temp3 = 416 - int(temp1)
                            temp4 = 416 - int(temp2)
                            cordinates[1] = cordinates[0]
                            cordinates[3] = cordinates[2]
                            cordinates[0] = str(temp4)
                            cordinates[2] = str(temp3)


                        final = final + cordinates[0] + "," + cordinates[1] + "," + cordinates[2] + "," + cordinates[3] + "," + cordinates[4] + " "

                        '''f.write(path + final)
                        file.close()'''
                        cordinates.clear()
                        cordinates = ["", "", "", "", ""]
                        break
                    elif x[i] == ',':

                        cordinate = cordinate + 1
                    else:
                        cordinates[cordinate] = cordinates[cordinate] + x[i]
                    i = i + 1
            i = i + 1
            if Ree == i:
                i = 0
                p = 0
                final = final + "\n"
                f.write(final)
                final = ""
                break




    '''for x in filee:
        linee = filee.readlines()
        linee = [linee.rstrip('\n') for linee in filee]
        print(linee)'''




    '''image = cv2.imread(path)
    file = open(annpath, 'a+')
    while True:
        f.readlines()


        file.write(path + ' ')
        file.close()
        clone = image.copy()
        cv2.namedWindow("image")
        image = clone.copy()
        file = open(annpath, 'a+')
        for i in coordinates:
            print(i + ':')
            temp = input()
            if temp=='x':
                continue
            else:
                file.write(i + ',' + temp + ' ')
        file.write('\n')
        if event == cv2.EVENT_LBUTTONDOWN:
            ref_point = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            ref_point.append((x, y))
            cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
            cv2.imshow("image", image)
            pr = str(ref_point)
            pr = pr.replace('[', '')
            pr = pr.replace(']', '')
            pr = pr.replace('(', '')
            pr = pr.replace(')', '')
            pr = pr.replace(' ', '')
            print(pr)'''
if __name__=='__main__':
    main()