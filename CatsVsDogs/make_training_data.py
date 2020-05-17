import os
import cv2
import numpy as np
from tqdm import tqdm

class DogsVSCats():

	def __init__(self):		
	    self.IM_SIZE = 50
	    # dataset path
	    self.CATS = "PetImages/Cat"
	    self.DOGS = "PetImages/Dog"
	    self.LABELS = {self.CATS:0,self.DOGS:1}
	    self.training_data = []
	    self.cat_count = 0
	    self.dog_count = 0
	    self.damaged_images = 0

	def make_training_data(self):
	    for label in self.LABELS:
	        for f in tqdm(os.listdir(label)):
	            try:
	                # gray scale and resize images
	                path = os.path.join(label,f)
	                img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
	                img = cv2.resize(img,(self.IM_SIZE,self.IM_SIZE))
	                '''
	                label images with one-hot vector, like [0, 1] or [1, 0]
	                training_data: [[image data, one hot vector],......]
	                '''
	                self.training_data.append([np.array(img),np.eye(2)[self.LABELS[label]]])

	                if label == self.CATS:
	                    self.cat_count += 1

	                elif label == self.DOGS:
	                    self.dog_count += 1
	            except Exception as e:
	                self.damaged_images += 1
	                print("{} destroyed.".format(path))
	                continue
	                
	    #save data in npy file, not reload data again in the future
	    np.random.shuffle(self.training_data)
	    np.save("training_data.npy",self.training_data)
	    #print all results
	    print(f"cat has {self.cat_count} images.")
	    print(f"dog has {self.dog_count} images.")
	    print(f"{self.damaged_images} images destroyed")


