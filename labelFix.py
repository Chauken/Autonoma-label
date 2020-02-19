# example of brighting image augmentation
import re
from numpy import expand_dims
#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array
#from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import glob
import cv2
import os
from matplotlib import pyplot

def Convert(string): 
    li = list(string.split(" ")) 
    return li




path = 'Lable_Shag4.txt'

str2 = " ("

pattern = re.compile(r"\((\d+)\)")

f=open(path,'r+')
fku =f.readlines()

fl = [0]*1000

lenku = len(fku)

for u in range(lenku):
	fl[u*6] = fku[u]

j= 0
for x in range (lenku):
	#print(x)	

	for i in range (6):
		fl_temp = fl[j]

		fl_temp2_pos = fl_temp.find(")")
		fl_temp2 = fl_temp[fl_temp2_pos:]


		list1 = pattern.findall(fl_temp)

		list1 = [int(i) for i in list1]
		list1[0] = j + i + 1
		list1 = str(list1)

		list1 = list1[1:-1]

		pos_x = fl_temp.find(str2)
		fl_temp = fl_temp[:pos_x+2] +list1 + fl_temp2

		fl[j + i] = fl_temp
	
	j=j+6


while fl[len(fl)-1] == 0:
	fl.pop()

for x in fl:

	print(x)


f.close







