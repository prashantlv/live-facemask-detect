import numpy
import cv2
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout

model = Sequential([
    Conv2D(128, (3,3), activation='relu', input_shape=()),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(2,activation='softmax')
])

model.compile(optimizer='adam', loss='binary_crossetropy', metrics='accuracy')
