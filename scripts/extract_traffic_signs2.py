from os import listdir
import cv2
import sys
import pandas as pd
import os
import re
import numpy as np

#This scripts extracts traffic signs and their labels from
#https://www.kaggle.com/andrewmvd/road-sign-detection

#Input folder for data. Make sure, that only road images are in the folder
input_path = sys.argv[0]
output_path = sys.argv[1]

#Read in image files
images_files = [f for f in listdir(input_path + 'dataset')]

images = {}

for image_file in images_files:
    images[image_file] = (cv2.imread(input_path + image_file))
    
labels = []
imgs = []
for image_file in images_files:
    with open("./annotations/" + image_file.replace(".png", ".xml")) as f:
        line = f.read()
        f.close()
        x = re.findall(r'[xy]m[ai][nx]>[0-9]?[0-9]?[0-9]', line)
        names = re.findall(r'(<name>)(.*)(</name>)', line)
        outer = []
        for coord in x:
            outer.append(int(re.findall(r'[0-9]?[0-9]?[0-9]', coord)[0]))
        for i in range(int(len(x)/4)):
            img = images[image_file][outer[i*4+1]:outer[i*4+3],outer[i*4]:outer[i*4+2],:]
            imgs.append(img)
            labels.append(names[i][1].strip("<name>").strip("</name>"))

data = np.array(labels)
df = pd.DataFrame(data=data, columns=["sign"])

if not os.path.exists(output_path):
    os.makedirs(output_path)

df.to_pickle(output_path + 'df.pkl')

for i in range(len(imgs)):
    if not os.path.exists(output_path):
        os.makedirs(output_path + 'dataset/')
    cv2.imwrite(output_path + 'dataset/' + 'img' + str(i) + '.jpg' ,imgs[i])