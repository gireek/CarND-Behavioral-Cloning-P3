import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten , Dense , Lambda , Convolution2D , Dropout , Activation , Cropping2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.layers.pooling import MaxPooling2D

from random import shuffle
import sklearn

def augment_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    # Add a constant so that it prevents the image from being completely dark
    random_bright = .25+np.random.uniform()
    # Apply the brightness reduction to the V channel
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1
  
lines = []
with open("./data/driving_log.csv") as file:
    reader = csv.reader(file)
    for line in reader:
        lines.append(line)
        
#ignore the first row with column headings --> center,left,right,steering,throttle,brake,speed
lines = lines[1:]
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = augment_brightness(cv2.imread(name))
                image = augment_brightness(cv2.flip(center_image, 1))
                center_angle = float(batch_sample[3])
                steering = -1*center_angle
                images.append(center_image)
                images.append(image)
                angles.append(center_angle)
                angles.append(steering)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Cropping2D(cropping=((70,25), (0,0)) , input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255.0 -0.5))

model.add(Convolution2D(24, 5, 5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit_generator(train_generator, steps_per_epoch= len(train_samples*2),validation_data=validation_generator, 
#                     validation_steps=len(validation_samples), epochs=5, verbose = 1)
filepath="model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=3, mode='min') 
callbacks_list = [checkpoint, early_stop]

model.fit_generator(train_generator, samples_per_epoch= len(train_samples*2), validation_data=validation_generator, nb_val_samples=len(validation_samples*2), callbacks=callbacks_list ,nb_epoch=15)