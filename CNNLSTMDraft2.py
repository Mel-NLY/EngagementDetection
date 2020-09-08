import tensorflow as tf
from keras import applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D,Conv3D, MaxPooling2D, TimeDistributed
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image 
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.layers import LSTM
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
import pandas as pd
from glob import glob
from tqdm import tqdm
from scipy.misc import imread, imresize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms,datasets, models
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
from datetime import datetime
from swa.keras import SWA


import requests
import json
 
def send_notification_via_pushbullet(title, body):
    """ Sending notification via pushbullet.
        Args:
            title (str) : title of text.
            body (str) : Body of text.
    """
    data_send = {"type": "note", "title": title, "body": body}
 
    ACCESS_TOKEN = 'o.A174Gezu4JEGQgO1q8ChR7Xy40H2WbG9'
    resp = requests.post('https://api.pushbullet.com/v2/pushes', data=json.dumps(data_send),
                         headers={'Authorization': 'Bearer ' + ACCESS_TOKEN, 'Content-Type': 'application/json'})
    if resp.status_code != 200:
        raise Exception('Something wrong')
    else:
        print ('complete sending')

use_gpu = torch.cuda.is_available()
if use_gpu:
    pinMem = True
else:
    pinMem = False

batch_size = 128

path=r"/media/mel/MEL/Documents/Jobs/NP Industry Project - Coddie/Research on Behavioral Teaching/Face Image Datasets/DAiSEE"
traindf=pd.read_csv(path+"/Labels/TrainFrameLabels.csv",dtype=str)
valdf=pd.read_csv(path+"/Labels/ValidationFrameLabels.csv",dtype=str)

# creating an empty list
images = glob(path+"/DataSet/Train/TrainFrames/*.jpg")
trainpart_image = []
train_image = []
trainpart_label = []
train_label = []

# for loop to read and store frames
for i in tqdm(range(traindf.shape[0]-1500000)):
#	try:
	if i % 5 == 0:
		# loading the image and keeping the target size as (224,224,3)
		img = image.load_img(path+"/DataSet/Train/TrainFrames/"+traindf['FrameID'][i], target_size=(50,50,3))
		# converting it to array
		img = image.img_to_array(img)
		# normalizing the pixel value
		img = img/255
		# appending the image to the train_image list
		trainpart_image.append(img)
		trainpart_label.append(float(traindf['Engagement'][i]))
	if i % 300 == 0:
		train_image.append(trainpart_image)
		train_label.append(trainpart_label)
		trainpart_image=[]
		trainpart_label=[]
#	except:
#		if i % 10 == 0:
#			trainpart_image.append(np.zeros((50,50,3)))
#			trainpart_label.append(float(0))
#		if i % 300 == 0:
#			train_image.append(trainpart_image)
#			train_label.append(trainpart_label)
#			trainpart_image=[]
#			trainpart_label=[]
#		continue
        
# converting the list to numpy array
X_train = np.stack(train_image[1:-1])
y_train = np.array(train_label[1:-1])
y_train = to_categorical(y_train)

# shape of the array
print(X_train.shape)
print(y_train.shape)

# Sending a notification when the sets are prepared
send_notification_via_pushbullet("Time: "+str(datetime.now())+"\nTraining & Validation Set", "X Training Images shape: "+str(X_train.shape)+"\nY Training Labels shape: "+str(y_train.shape))

# creating an empty list
images = glob(path+"/DataSet/Validation/ValidationFrames/*.jpg")
validationpart_image = []
validation_image = []
validationpart_label = []
validation_label = []

# for loop to read and store frames
for i in tqdm(range(valdf.shape[0]-450000)):
#	try:
	if i % 5 == 0:
	    # loading the image and keeping the target size as (224,224,3)
	    img = image.load_img(path+"/DataSet/Validation/ValidationFrames/"+valdf['FrameID'][i], target_size=(50,50,3))
	    # converting it to array
	    img = image.img_to_array(img)
	    # normalizing the pixel value
	    img = img/255
	    # appending the image to the train_image list
	    validationpart_image.append(img)
	    validationpart_label.append(float(valdf['Engagement'][i]))
	if i % 300 == 0:
	    validation_image.append(validationpart_image)
	    validation_label.append(validationpart_label)
	    validationpart_image=[]
	    validationpart_label=[]
#	except:
#		if i % 10 == 0:
#			validationpart_image.append(np.zeros((50,50,3)))
#			validationpart_label.append(float(0))
#		if i % 300 == 0:
#			validation_image.append(validationpart_image)
#			validation_label.append(validationpart_label)
#			validationpart_image=[]
#			validationpart_label=[]
#		continue
        
# converting the list to numpy array
X_val = np.stack(validation_image[1:-1])
y_val = np.array(validation_label[1:-1])
y_val = to_categorical(y_val)

# shape of the array
print(X_val.shape)
print(y_val.shape)

# Sending a notification when the sets are prepared
send_notification_via_pushbullet("Time: "+str(datetime.now())+"\nTraining & Validation Set", "X Val Images shape: "+str(X_val.shape)+"\nY Val Labels shape: "+str(y_val.shape))

# Input (No. Video Samples, Timesteps, Dimension of Image, Channels)
input_shape = (60, 50, 50, 3)

model = Sequential()
model.add(TimeDistributed(Conv2D(32, (5, 5), padding='same',activation='relu'),input_shape=input_shape))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Dropout(0.3)))
model.add(TimeDistributed(Conv2D(16, (5, 5), padding='same', activation='relu')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Dropout(0.3)))
model.add(TimeDistributed(Conv2D(16, (5, 5),kernel_regularizer=l2(0.01),activation='relu')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Dropout(0.3)))
model.add(TimeDistributed(Conv2D(16, (5, 5),kernel_regularizer=l2(0.01),activation='relu')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Dropout(0.3)))
model.add(TimeDistributed(Flatten()))

model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), return_sequences=True))
model.add(Dense(4, activation='softmax'))

print(model.summary())

sgd = SGD(lr=0.002, decay = 1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

#callbacks = [ EarlyStopping(monitor='val_loss', patience=10, verbose=0), ModelCheckpoint('video_1_LSTM_1_1024.h5', monitor='val_loss', save_best_only=True, verbose=0) ]
swa = SWA(start_epoch=2, lr_schedule='cyclic', swa_lr=0.001, swa_lr2=0.005,swa_freq=3,verbose=1)
nb_epoch = 30

model.fit(X_train,y_train,validation_data=(X_val,y_val),batch_size=1, epochs=nb_epoch,callbacks=[swa],verbose=1, shuffle=True)
#model.fit_generator(generator=train_generator(size=size, batch_size=batch_size), steps_per_epoch = 1645782 // 30,epochs = 10, validation_data=validation_generator, validation_steps = 1516784 // 30)

# Sending a notification when model finishes training
send_notification_via_pushbullet("Time: "+str(datetime.now())+"\nModel_3.h5 Finished Training"," ")

model.save("model_3.h5")

model = tf.keras.models.load_model('model_3.h5')

path=r"/media/mel/MEL/Documents/Jobs/NP Industry Project - Coddie/Research on Behavioral Teaching/Face Image Datasets/DAiSEE"
testdf=pd.read_csv(path+"/Labels/TestFrameLabels.csv",dtype=str)

# creating an empty list
images = glob(path+"/DataSet/Test/TestFrames/*.jpg")
testpart_image = []
test_image = []
testpart_label = []
test_label = []

# for loop to read and store frames
for i in tqdm(range(testdf.shape[0]-450000)):
    if i % 5 == 0:
        # loading the image and keeping the target size as (224,224,3)
        img = image.load_img(path+"/DataSet/Test/TestFrames/"+testdf['FrameID'][i], target_size=(50,50,3))
        # converting it to array
        img = image.img_to_array(img)
        # normalizing the pixel value
        img = img/255
        # appending the image to the train_image list
        testpart_image.append(img)
        testpart_label.append(float(testdf['Engagement'][i]))
    if i % 300 == 0:
        test_image.append(testpart_image)
        test_label.append(testpart_label)
        testpart_image=[]
        testpart_label=[]

# converting the list to numpy array
X_test = np.stack(test_image[1:-1])
y_test = np.array(test_label[1:-1])
y_test = to_categorical(y_test)

# shape of the array
print(X_test.shape)
print(y_test.shape)

test_loss, test_acc = model.evaluate(X_test,  y_test, batch_size=1, verbose=2)

print("Test Accuracy: " + str(test_acc))
print("Test Loss: " + str(test_loss))

# Sending a notification when model finishes evaluation
send_notification_via_pushbullet("Time: "+str(datetime.now())+"\nModel_3.h5 Finished Evaluation", "Test Accuracy: "+str(test_acc)+"\nTest Loss: "+str(test_loss))
