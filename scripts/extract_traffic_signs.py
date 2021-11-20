from os import listdir
import cv2
import sys
import pandas as pd
import os
import numpy as np

#This scripts extracts traffic signs and their labels from
#http://www.cvl.isy.liu.se/research/datasets/traffic-signs-dataset/

#Input folder for data. Make sure, that only road images are in the folder
input_path = sys.argv[0]
output_path = sys.argv[1]

#Read in image files
images_files = [f for f in listdir(input_path + 'dataset')]

images = {}

for image_file in images_files:
    images[image_file] = (cv2.imread(input_path + 'dataset/' + image_file))

imgs = []
label_visibility = []
label_type = []
label_sign = []

with open('labels.txt') as f:
    line = f.readline()
    while line != '':
        line = line.split(':')
        if line[1] == '' or line[1] == "\n":
            line = f.readline()
            continue
        name = line[0]
        signs = line[1].split(';')
        signs = signs[0:(len(signs)-1)]
        for sign in signs:
            if sign == 'MISC_SIGNS':
                continue
            splitted = sign.split(", ")
            coord1 = (splitted[3],splitted[4])
            coord2 = (splitted[1],splitted[2])
            img = images[name][int(float(coord1[1])):int(float(coord2[1])),int(float(coord1[0])):int(float(coord2[0])),:]
            imgs.append(img)
            label_visibility.append(splitted[0])
            if splitted[6] == "PRIORITY_ROAD": 
                label_type.append("PRIORITY")
            elif splitted[6] == "PEDESTRIAN_CROSSING":
                label_type.append("SPECIAL_REGULATIONS")
            else:
                label_type.append(splitted[5])
            
            label_sign.append(splitted[6])
        
        line = f.readline()
        
    f.close()

data = np.array([label_visibility,label_type,label_sign])
data = np.transpose(data)
df = pd.DataFrame(data=data, columns=["visibility", "type", "sign"])

if not os.path.exists(output_path):
    os.makedirs(output_path)

df.to_pickle(output_path + 'df.pkl')

for i in range(len(imgs)):
    if not os.path.exists(output_path):
        os.makedirs(output_path + 'dataset/')
    cv2.imwrite(output_path + 'dataset/' + 'img' + str(i) + '.jpg' ,imgs[i])