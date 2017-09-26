
# coding: utf-8

# In[1]:

import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import sklearn
from sklearn.utils import shuffle


# In[2]:

def keras_model():
    
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    
    return model


# In[3]:

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #randomly choosing between center, left and right images to make the learning more robust
                i = int(np.random.choice(['0', '1', '2']))
                name = 'data1/IMG/'+batch_sample[i].split('/')[-1]
                image = cv2.imread(name)
                #The function below outputs 0 for i=0, 0.25 for i=1 and -0.25 for i=2
                steering_correction = 0.25*(3 - 2*i) if i else 0
                angle = float(batch_sample[3]) + steering_correction
                images.append(image)
                angles.append(angle)
                
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                #Augmenting only the images that have a non-zero steering angle, to compensate for the bias towards
                #straight driving in driving data
                if angle!=0:
                    augmented_images.append(image)
                    augmented_angles.append(angle)
                    augmented_images.append(cv2.flip(image,1))
                    augmented_angles.append(angle*-1.0)
                

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# In[ ]:

samples = []
with open('data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
del(samples[0])

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# In[ ]:

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = keras_model()

model.fit_generator(train_generator, validation_data=validation_generator, samples_per_epoch=14000, nb_epoch=3, nb_val_samples=len(validation_samples))



# In[ ]:

model.save('model.h5')

