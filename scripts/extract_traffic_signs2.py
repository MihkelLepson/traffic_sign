from os import listdir
import cv2
import sys
import pandas as pd
import os

#This scripts extracts traffic signs and their labels from
#https://www.kaggle.com/andrewmvd/road-sign-detection

#Input folder for data. Make sure, that only road images are in the folder
input_path = sys.argv[0]
output_path = sys.argv[1]

#Read in image files
images_files = [f for f in listdir(input_path + 'dataset')]

images = []

for image_file in images_files:
    images.append(cv2.imread(input_path + 'dataset/' + image_file))
    
with open(input_path + 'labels.txt') as f:
        #TODO
        line = f.readline()
    f.close()