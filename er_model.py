import numpy as np
import pandas as pd
import keras
from keras import layers
import keras.backend as K
from keras.initializers import glorot_uniform
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def inception_block(input_x, filters):
    f1, f2, f3, f4, f5, f6 = filters

    X = input_x

    conv1_1x1 = layers.Conv2D(f1, kernel_size=(1,1), padding='same', kernel_initializer=glorot_uniform())(input_x)
    conv1_1x1 = layers.Activation('relu')(conv1_1x1)

    conv2_1x1 = layers.Conv2D(f2, kernel_size=(1,1), padding='same', kernel_initializer=glorot_uniform())(input_x)
    conv2_1x1 = layers.Activation('relu')(conv2_1x1)
    conv2_3x3 = layers.Conv2D(f3, kernel_size=(3,3), padding='same', kernel_initializer=glorot_uniform())(conv2_1x1)
    conv2_3x3 = layers.Activation('relu')(conv2_3x3)

    conv3_1x1 = layers.Conv2D(f4, kernel_size=(1,1), padding='same', kernel_initializer=glorot_uniform())(input_x)
    conv3_1x1 = layers.Activation('relu')(conv3_1x1)
    conv3_5x5 = layers.Conv2D(f5, kernel_size=(5,5), padding='same', kernel_initializer=glorot_uniform())(conv3_1x1)
    conv3_5x5 = layers.Activation('relu')(conv3_5x5)

    pool = layers.MaxPool2D(pool_size=(3,3), strides=(1,1), padding="same")(input_x)
    conv_pool_1x1 = layers.Conv2D(f6, kernel_size=(1,1), padding='same', kernel_initializer=glorot_uniform())(pool)
    conv_pool_1x1 = layers.Activation('relu')(conv_pool_1x1)

    output = layers.Concatenate(axis=-1)([conv1_1x1, conv2_3x3, conv3_5x5, conv_pool_1x1])

    return output



def model_def(input_shape, n_classes):
    input_x = layers.Input(input_shape)

    X = input_x

    X = layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same", kernel_initializer=glorot_uniform())(X)
    X = layers.Activation('relu')(X)

    X = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="valid")(X)

    X = layers.Conv2D(128, kernel_size=(3,3), strides=(2,2), padding="same", kernel_initializer=glorot_uniform())(X)
    X = layers.Activation('relu')(X)

    X = inception_block(X, [64, 96, 128, 16, 32, 32])

    X = inception_block(X, [128, 128, 192, 32, 96, 64])

    X = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="valid")(X)

    X = inception_block(X, [192, 96, 208, 16, 48, 64])

    X = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="valid")(X)

    X = inception_block(X, [256, 160, 320, 32, 128, 128])

    X = inception_block(X, [384, 192, 384, 48, 128, 128])

    X = layers.MaxPool2D(pool_size=(3,3), strides=(1,1), padding="valid")(X)

    X = layers.Flatten()(X)

    X = layers.Dropout(0.4)(X)

    X = layers.Dense(n_classes, activation='softmax')(X)

    model = keras.Model(inputs=input_x, outputs=X)

    model.load_weights('models/er_model_1.h5')

    return model


class model_class:
    def __init__(self):
        self.em_model = model_def(input_shape = (48, 48, 1), n_classes=6)
        self.em_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def predict(self, img_array):
        return self.em_model.predict(img_array)
