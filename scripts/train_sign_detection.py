from os import listdir
import cv2
import sys
import pandas as pd
import os
import numpy as np

from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

#This scripts extracts traffic signs and their labels from
#http://www.cvl.isy.liu.se/research/datasets/traffic-signs-dataset/

#Input folder for data. Make sure, that only road images are in the folder
input_path = sys.argv[0]
output_path1 = sys.argv[1]
output_path2 = sys.argv[2]

#Read in image files
images_files = [f for f in listdir(input_path + 'dataset')]

images = {}

for image_file in images_files:
    images[image_file] = (cv2.imread(input_path + 'dataset/' + image_file))

imgs_mask = []
imgs = []

with open(input_path + 'labels.txt') as f:
    line = f.readline()
    while line != '':
        line = line.split(':')
        if line[1] == '' or line[1] == "\n":
            line = f.readline()
            continue
        name = line[0]
        signs = line[1].split(';')
        signs = signs[0:(len(signs)-1)]
        
        img_new = np.zeros((images[name].shape[0],images[name].shape[1]))
        
        for sign in signs:
            if sign == 'MISC_SIGNS':
                continue
            splitted = sign.split(", ")
            coord1 = (splitted[3],splitted[4])
            coord2 = (splitted[1],splitted[2])
            img = images[name][int(float(coord1[1])):int(float(coord2[1])),int(float(coord1[0])):int(float(coord2[0])),:]
            img_new[int(float(coord1[1])):int(float(coord2[1])),int(float(coord1[0])):int(float(coord2[0]))] = 1
        
        imgs_mask.append(img_new)
        imgs.append(images[name]/255)
        line = f.readline()
        
    f.close()


for i in range(len(imgs)):
    if not os.path.exists(output_path):
        os.makedirs(output_path + 'dataset/')
    cv2.imwrite(output_path1 + 'dataset/' + 'img' + str(i) + '.jpg' ,imgs[i])
	cv2.imwrite(output_path2 + 'dataset/' + 'img' + str(i) + '.jpg' ,imgs[i])

size = imgs[0].shape

train_images = imgs[0:1600]
train_labels = imgs_mask[0:1600]
test_images = imgs[1600:1827]
test_labels = imgs_mask[1600:1827]

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

train_batches = train_dataset.batch(64)
test_batches = test_dataset.batch(64)

base_model = tf.keras.applications.MobileNetV2(input_shape=[size[0], size[1], 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

OUTPUT_CLASSES = 2

model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
                               
EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_batches, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_batches,
                          callbacks=[DisplayCallback()])


model.save_weights('model.h5')