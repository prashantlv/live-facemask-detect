import numpy
import cv2
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
model = Sequential([
    Conv2D(128, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Dropout(0.2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.2),

    Flatten(),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dense(2,activation='softmax')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')


TRAINING_DIR = './dataset/train'

train_datagen = ImageDataGenerator(
                rescale=1.0/255,
                rotation_range=30,
                width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
                    TRAINING_DIR,
                    batch_size=10,
                    target_size=(150,150))

VALIDATION_DIR = './dataset/test'
validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
                                                         batch_size=10, 
                                                         target_size=(150, 150))

history = model.fit_generator(train_generator,
                              epochs=4,
                              validation_data=validation_generator)

model.save('./model.h5')