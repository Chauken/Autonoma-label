# example of brighting image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import glob
import cv2
import os
import re
from matplotlib import pyplot
import natsort



datagen = ImageDataGenerator(
    brightness_range=[0.5,1.5],
    )

pattern = re.compile(r"\((\d+)\)")


img_dir = "/home/martin/Desktop/shaggycam Train/ShaggyCam3/" # Enter Directory of all images 
img_saveDir = "/home/martin/Desktop/shaggycam Train/ShaggyCam3/done/"


data_path = os.path.join(img_dir,'*.jpg')
files = glob.glob(data_path)

files.sort(key=lambda x: int(pattern.findall(x)[0]))


prefix = "  ("
filename = "1"
filetype = ").jpg"




for f1 in files:


	# load the image
	img = cv2.imread(f1)
	

	
	# convert to numpy array
	data = img_to_array(img)
	# expand dimension to one sample
	samples = expand_dims(data, 0)
	# prepare iterator
	it = datagen.flow(samples, batch_size=1)



	for i in range(6):
		imgToSave = img_saveDir+prefix+filename+filetype
		batch = it.next()
		
		


		#print(len(batch))
		cv2.imwrite(imgToSave,batch[0])


		filename_temp = int(filename)
		#print("",filename)
		filename_temp = filename_temp + 1
		
		filename = str(filename_temp)
		
		#print("",filename)


print("Done")


