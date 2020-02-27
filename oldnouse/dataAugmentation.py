# example of brighting image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import glob
import cv2
import os
from matplotlib import pyplot

datagen = ImageDataGenerator(
    brightness_range=[0.5,1.5],
    )


img_dir = "/home/martin/Desktop/ShaggyCam4/" # Enter Directory of all images 
img_saveDir = "/home/martin/Desktop/ShaggyCam4/done/"
data_path = os.path.join(img_dir,'*jpg')
files = glob.glob(data_path)

filename = "1"
filetype = ".png"






for f1 in files:


	# load the image
	img = cv2.imread(f1)
	# convert to numpy array
	data = img_to_array(img)
	# expand dimension to one sample
	samples = expand_dims(data, 0)
	# prepare iterator  
	it = datagen.flow(samples, batch_size=1)
	# generate samples and plot



	for i in range(6):
		imgToSave = img_saveDir+filename+filetype
		batch = it.next()

		cv2.imwrite(imgToSave,batch[0])

		filename_temp = int(filename)
		#print("",filename)
		filename_temp = filename_temp + 1
		
		filename = str(filename_temp)
		
		print("",filename)





